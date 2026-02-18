import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import copy
from GNN import JobShopGNNPolicy, prepare_graph_for_gnn
from graph_generator import generate_graph
from generator import generate_general_instances
from jobshop import JobShopInstance
from schedule import Schedule


class JobShopEnv:
    """
    Job Shop Scheduling Environment for RL.
    
    State: Current partial schedule with available operations
    Action: Select which operation to schedule next
    Reward: Based on schedule quality (minimize makespan)
    """
    def __init__(self, instance: JobShopInstance):
        self.instance = instance
        self.reset()
        
    def reset(self):
        """Reset environment to initial state."""
        self.current_time = 0
        self.schedule = Schedule(self.instance)
        
        # Track job progress (next operation index for each job)
        self.job_progress = [0] * self.instance.num_jobs
        
        # Track machine availability (when each machine becomes free)
        self.machine_available = [0] * self.instance.num_machines
        
        # Track which operations are completed
        self.completed_ops = set()
        
        # Track makespan (max completion time)
        self.makespan = 0
        
        return self._get_state()
    
    def _get_state(self):
        """
        Get current state representation.
        
        Returns:
            dict with:
                - node_features: features for all operations
                - edge_index: graph structure
                - mask: boolean mask of available operations
                - done: whether episode is complete
        """
        # Generate graph structure
        adjacency_list = generate_graph(self.instance)
        node_features, edge_index, node_id_map = prepare_graph_for_gnn(adjacency_list)
        
        # Create mask for available operations
        mask = self._get_available_operations_mask(node_id_map)
        
        # Check if done
        done = len(self.completed_ops) == self.instance.num_operations
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'node_id_map': node_id_map,
            'mask': mask,
            'done': done
        }
    
    def _get_available_operations_mask(self, node_id_map):
        """
        Create boolean mask indicating which operations can be scheduled.
        An operation is available if:
        1. It hasn't been scheduled yet
        2. All preceding operations in its job are completed
        """
        mask = torch.zeros(len(node_id_map), dtype=torch.bool)
        
        for job_id, job in enumerate(self.instance.jobs):
            next_op_idx = self.job_progress[job_id]
            
            if next_op_idx < len(job):
                # Find the node index for this operation
                node_name = f"job{job_id}_op{next_op_idx}"
                if node_name in node_id_map:
                    node_idx = node_id_map[node_name]
                    mask[node_idx] = True
        
        return mask
    
    def step(self, action):
        """
        Execute action (schedule an operation).
        
        Args:
            action: index of operation to schedule
            
        Returns:
            next_state, reward, done, info
        """
        state = self._get_state()
        node_id_map = state['node_id_map']
        
        # Get operation from action
        reverse_map = {v: k for k, v in node_id_map.items()}
        node_name = reverse_map[action]
        
        # Parse job_id and op_id from node_name
        parts = node_name.split('_')
        job_id = int(parts[0].replace('job', ''))
        op_idx = int(parts[1].replace('op', ''))
        
        operation = self.instance.jobs[job_id][op_idx]
        
        # Calculate start time for this operation
        # Must wait for: 1) previous operation in job, 2) machine availability
        prev_op_end_time = 0
        if op_idx > 0:
            prev_op = self.instance.jobs[job_id][op_idx - 1]
            prev_start = self.schedule.get_operation_start_time(prev_op.operation_id)
            if prev_start is not None:
                prev_op_end_time = prev_start + prev_op.duration
        
        machine_avail_time = self.machine_available[operation.machine_id]
        start_time = max(prev_op_end_time, machine_avail_time)
        
        # Schedule the operation
        self.schedule.add_operation(operation.operation_id, start_time)
        
        # Update state
        completion_time = start_time + operation.duration
        self.machine_available[operation.machine_id] = completion_time
        self.job_progress[job_id] += 1
        self.completed_ops.add(operation.operation_id)
        self.makespan = max(self.makespan, completion_time)
        
        # Get next state
        next_state = self._get_state()
        done = next_state['done']
        
        # Calculate reward
        reward = self._calculate_reward(done)
        
        info = {
            'makespan': self.makespan,
            'scheduled_op': node_name,
            'start_time': start_time,
            'completion_time': completion_time
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, done):
        """
        Calculate reward for current step.
        
        Reward design:
        - Small negative reward for each step (encourage efficiency)
        - Large negative reward proportional to final makespan when done
        """
        if done:
            # Final reward: negative makespan (want to minimize)
            # Normalize by number of operations to make it scale-independent
            return -self.makespan / self.instance.num_operations
        else:
            # Step penalty to encourage finishing quickly
            return -0.1
    
    def get_makespan(self):
        """Return current makespan."""
        return self.makespan


class ExperienceBuffer:
    """
    Buffer for storing trajectories for PPO training.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def add(self, state, action, reward, value, log_prob, done):
        """Add experience to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        """Clear buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def get(self):
        """Get all experiences."""
        return (self.states, self.actions, self.rewards, 
                self.values, self.log_probs, self.dones)
    
    def __len__(self):
        return len(self.states)


class PPOScheduler:
    """
    Proximal Policy Optimization (PPO) for Job Shop Scheduling.
    
    Uses GNN to encode graph state and learns to schedule operations
    to minimize makespan through trial and error (unsupervised).
    """
    def __init__(self, 
                 policy: JobShopGNNPolicy,
                 lr=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_epsilon=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 ppo_epochs=4,
                 device='cpu'):
        """
        Initialize PPO trainer.
        
        Args:
            policy: GNN-based policy network
            lr: learning rate
            gamma: discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping parameter
            value_coef: value loss coefficient
            entropy_coef: entropy bonus coefficient
            max_grad_norm: gradient clipping
            ppo_epochs: number of epochs per update
            device: torch device
        """
        self.policy = policy.to(device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.device = device
        
        # Training statistics
        self.episode_rewards = []
        self.episode_makespans = []
        self.policy_losses = []
        self.value_losses = []
        
    def select_action(self, state, deterministic=False):
        """
        Select action using current policy.
        
        Args:
            state: environment state
            deterministic: if True, select argmax action
            
        Returns:
            action, log_prob, value
        """
        node_features = state['node_features'].to(self.device)
        edge_index = state['edge_index'].to(self.device)
        mask = state['mask'].to(self.device)
        
        with torch.no_grad():
            output = self.policy(node_features, edge_index, mask)
            action_logits = output['action_logits']
            value = output['value']
            
            # Create distribution over valid actions
            probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            
            if deterministic:
                action = torch.argmax(probs)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: list of rewards
            values: list of value estimates
            dones: list of done flags
            
        Returns:
            advantages, returns
        """
        advantages = []
        gae = 0
        
        # Work backwards through the episode
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        # Returns are advantages + values
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def update(self, buffer: ExperienceBuffer):
        """
        Update policy using PPO algorithm.
        
        Args:
            buffer: experience buffer with trajectories
        """
        states, actions, rewards, values, old_log_probs, dones = buffer.get()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Convert to tensors
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        for epoch in range(self.ppo_epochs):
            # Evaluate actions with current policy
            policy_loss_epoch = 0
            value_loss_epoch = 0
            entropy_epoch = 0
            
            for i in range(len(states)):
                state = states[i]
                action = actions[i]
                old_log_prob = old_log_probs[i]
                advantage = advantages[i]
                ret = returns[i]
                
                # Get current policy output
                node_features = state['node_features'].to(self.device)
                edge_index = state['edge_index'].to(self.device)
                mask = state['mask'].to(self.device)
                
                output = self.policy(node_features, edge_index, mask)
                action_logits = output['action_logits']
                value = output['value']
                
                # Compute new log prob and entropy
                probs = F.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                new_log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                
                # Compute policy loss (PPO clip)
                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
                policy_loss = -torch.min(surr1, surr2)
                
                # Compute value loss
                value_loss = F.mse_loss(value, ret)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_epoch += entropy.item()
            
            # Track losses
            if epoch == 0:  # Only log first epoch to avoid clutter
                self.policy_losses.append(policy_loss_epoch / len(states))
                self.value_losses.append(value_loss_epoch / len(states))
    
    def train_episode(self, instance: JobShopInstance, verbose=False):
        """
        Train on a single episode (one scheduling problem).
        
        Args:
            instance: JobShopInstance to schedule
            verbose: if True, print episode details
            
        Returns:
            episode_reward, makespan
        """
        env = JobShopEnv(instance)
        buffer = ExperienceBuffer()
        
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # Collect trajectory
        while not done:
            action, log_prob, value = self.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            buffer.add(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            state = next_state
            step += 1
            
            if verbose and step % 10 == 0:
                print(f"  Step {step}: scheduled {info['scheduled_op']}, "
                      f"starts at {info['start_time']:.0f}")
        
        makespan = env.get_makespan()
        
        # Update policy
        self.update(buffer)
        
        # Track statistics
        self.episode_rewards.append(episode_reward)
        self.episode_makespans.append(makespan)
        
        if verbose:
            print(f"Episode completed: {step} steps, makespan={makespan:.2f}, "
                  f"reward={episode_reward:.2f}")
        
        return episode_reward, makespan
    
    def train(self, instances, num_epochs=100, verbose=True):
        """
        Train on multiple instances for multiple epochs.
        
        Args:
            instances: list of JobShopInstance
            num_epochs: number of training epochs
            verbose: if True, print training progress
            
        Returns:
            training_stats: dict with training statistics
        """
        print("=" * 60)
        print("Starting PPO Training for Job Shop Scheduling")
        print("=" * 60)
        print(f"Training instances: {len(instances)}")
        print(f"Epochs: {num_epochs}")
        print(f"Policy parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        print("=" * 60)
        
        best_avg_makespan = float('inf')
        
        for epoch in range(num_epochs):
            epoch_makespans = []
            epoch_rewards = []
            
            # Train on all instances
            for inst_idx, instance in enumerate(instances):
                reward, makespan = self.train_episode(instance, verbose=False)
                epoch_makespans.append(makespan)
                epoch_rewards.append(reward)
            
            avg_makespan = np.mean(epoch_makespans)
            avg_reward = np.mean(epoch_rewards)
            
            if avg_makespan < best_avg_makespan:
                best_avg_makespan = avg_makespan
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"  Avg Makespan: {avg_makespan:.2f} (best: {best_avg_makespan:.2f})")
                print(f"  Avg Reward: {avg_reward:.2f}")
                if len(self.policy_losses) > 0:
                    print(f"  Policy Loss: {self.policy_losses[-1]:.4f}")
                    print(f"  Value Loss: {self.value_losses[-1]:.4f}")
        
        print("=" * 60)
        print("Training completed!")
        print(f"Best average makespan: {best_avg_makespan:.2f}")
        print("=" * 60)
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_makespans': self.episode_makespans,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'best_avg_makespan': best_avg_makespan
        }
    
    def evaluate(self, instance: JobShopInstance, verbose=True):
        """
        Evaluate policy on a single instance (deterministic).
        
        Args:
            instance: JobShopInstance to schedule
            verbose: if True, print details
            
        Returns:
            schedule, makespan
        """
        env = JobShopEnv(instance)
        state = env.reset()
        done = False
        step = 0
        
        if verbose:
            print(f"\nEvaluating on instance: {instance}")
        
        while not done:
            action, _, _ = self.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            
            if verbose:
                print(f"Step {step + 1}: {info['scheduled_op']} "
                      f"-> starts at {info['start_time']:.0f}")
            
            state = next_state
            step += 1
        
        makespan = env.get_makespan()
        
        if verbose:
            print(f"\nFinal makespan: {makespan:.2f}")
        
        return env.schedule, makespan


def main():
    """
    Demo: Train PPO on job shop scheduling.
    """
    print("Job Shop Scheduling with PPO + GNN")
    print()
    
    # Generate training instances (small problems to start)
    train_instances = generate_general_instances(
        num_instances=5,
        num_jobs_range=(3, 5),
        num_machines_range=(3, 4),
        num_op_range=(3, 5),
        seed=42
    )
    
    print("Training instances:")
    for i, inst in enumerate(train_instances):
        print(f"  {i+1}. {inst}")
    
    # Create GNN policy
    policy = JobShopGNNPolicy(
        node_feature_dim=4,
        hidden_dim=32,  # Smaller for faster training
        num_heads=2,
        num_layers=2
    )
    
    # Create PPO trainer
    ppo = PPOScheduler(
        policy=policy,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        ppo_epochs=4
    )
    
    # Train
    stats = ppo.train(train_instances, num_epochs=50, verbose=True)
    
    # Evaluate on first instance
    print("\n" + "=" * 60)
    print("Evaluation on first training instance:")
    print("=" * 60)
    schedule, makespan = ppo.evaluate(train_instances[0], verbose=True)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
