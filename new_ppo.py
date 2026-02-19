import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from GNN import JobShopGNNPolicy
from muxGNN import MuxJobShopGNNPolicy
from generator import generate_general_instances, load_instances_from_json
from jobshop import JobShopInstance
from schedule import Schedule


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Transition:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    precedence_edge_index: torch.Tensor
    machine_edge_index: torch.Tensor
    mask: torch.Tensor
    action: int
    reward: float
    value: float
    log_prob: float
    done: bool


class JobShopEnv:
    """
    Environment for dispatching operations in Job Shop Scheduling.

    Action space: all operations in the instance; only masked available ops are valid.
    State: graph node features + edge index + availability mask.
    """

    def __init__(self, instance: JobShopInstance):
        self.instance = instance
        self._build_static_graph()
        self.reset()

    def _build_static_graph(self) -> None:
        """
        Build fixed operation index and edge structure once per instance.
        Node id is operation.operation_id (global and unique per instance).
        """
        self.num_nodes = self.instance.num_operations
        src_list: List[int] = []
        dst_list: List[int] = []
        precedence_src: List[int] = []
        precedence_dst: List[int] = []
        machine_src: List[int] = []
        machine_dst: List[int] = []

        # Precedence edges (job sequence)
        for job in self.instance.jobs:
            for pos in range(len(job) - 1):
                src_list.append(job[pos].operation_id)
                dst_list.append(job[pos + 1].operation_id)
                precedence_src.append(job[pos].operation_id)
                precedence_dst.append(job[pos + 1].operation_id)

        # Machine edges (clique among operations on same machine)
        machine_to_ops: Dict[int, List[int]] = {}
        for job in self.instance.jobs:
            for op in job:
                machine_to_ops.setdefault(op.machine_id, []).append(op.operation_id)

        for ops in machine_to_ops.values():
            for i in range(len(ops)):
                for j in range(len(ops)):
                    if i != j:
                        src_list.append(ops[i])
                        dst_list.append(ops[j])
                        machine_src.append(ops[i])
                        machine_dst.append(ops[j])

        if src_list:
            self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)

        if precedence_src:
            self.precedence_edge_index = torch.tensor([precedence_src, precedence_dst], dtype=torch.long)
        else:
            self.precedence_edge_index = torch.zeros((2, 0), dtype=torch.long)

        if machine_src:
            self.machine_edge_index = torch.tensor([machine_src, machine_dst], dtype=torch.long)
        else:
            self.machine_edge_index = torch.zeros((2, 0), dtype=torch.long)

        self.op_meta: Dict[int, Tuple[int, int, int, int]] = {}
        for job_id, job in enumerate(self.instance.jobs):
            for pos, op in enumerate(job):
                self.op_meta[op.operation_id] = (job_id, pos, op.machine_id, op.duration)

    def reset(self):
        self.schedule = Schedule(self.instance)
        self.job_progress = [0] * self.instance.num_jobs
        self.machine_available = [0.0] * self.instance.num_machines
        self.job_last_completion = [0.0] * self.instance.num_jobs
        self.completed_ops = set()
        self.makespan = 0.0
        return self._get_state()

    def _available_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        for job_id, job in enumerate(self.instance.jobs):
            next_idx = self.job_progress[job_id]
            if next_idx < len(job):
                op = job[next_idx]
                mask[op.operation_id] = True
        return mask

    def _max_duration(self) -> float:
        max_d = 1
        for job in self.instance.jobs:
            for op in job:
                if op.duration > max_d:
                    max_d = op.duration
        return float(max_d)

    def _get_state(self):
        done = len(self.completed_ops) == self.num_nodes
        mask = self._available_mask()

        max_d = self._max_duration()
        time_scale = max(1.0, max(self.machine_available) if self.machine_available else 1.0)

        # Dynamic node features (8 dims):
        # [duration, machine_id, job_id, op_pos, is_completed, is_available, ready_time, est_start]
        node_features = torch.zeros((self.num_nodes, 8), dtype=torch.float32)

        for op_id in range(self.num_nodes):
            job_id, pos, machine_id, duration = self.op_meta[op_id]
            is_completed = 1.0 if op_id in self.completed_ops else 0.0
            is_available = 1.0 if mask[op_id].item() else 0.0

            ready_time = self.job_last_completion[job_id]
            est_start = max(ready_time, self.machine_available[machine_id])

            node_features[op_id, 0] = float(duration) / max_d
            node_features[op_id, 1] = float(machine_id) / max(1, self.instance.num_machines - 1)
            node_features[op_id, 2] = float(job_id) / max(1, self.instance.num_jobs - 1)
            node_features[op_id, 3] = float(pos) / max(1, len(self.instance.jobs[job_id]) - 1)
            node_features[op_id, 4] = is_completed
            node_features[op_id, 5] = is_available
            node_features[op_id, 6] = float(ready_time) / time_scale
            node_features[op_id, 7] = float(est_start) / time_scale

        return {
            "node_features": node_features,
            "edge_index": self.edge_index,
            "precedence_edge_index": self.precedence_edge_index,
            "machine_edge_index": self.machine_edge_index,
            "mask": mask,
            "done": done,
        }

    def step(self, action: int):
        state = self._get_state()
        if not state["mask"][action].item():
            # Invalid action protection; heavy penalty.
            return self._get_state(), -5.0, False, {"invalid_action": True, "makespan": self.makespan}

        job_id, pos, machine_id, duration = self.op_meta[action]

        start_time = max(self.job_last_completion[job_id], self.machine_available[machine_id])
        completion_time = start_time + duration

        self.schedule.add_operation(action, start_time)
        self.machine_available[machine_id] = completion_time
        self.job_last_completion[job_id] = completion_time
        self.job_progress[job_id] += 1
        self.completed_ops.add(action)
        self.makespan = max(self.makespan, completion_time)

        next_state = self._get_state()
        done = next_state["done"]

        # Dense shaping: incremental makespan growth penalty + final normalized makespan penalty.
        reward = -0.01
        if done:
            reward += -self.makespan / max(1, self.num_nodes)

        info = {
            "job_id": job_id,
            "position": pos,
            "start_time": start_time,
            "completion_time": completion_time,
            "makespan": self.makespan,
            "invalid_action": False,
        }
        return next_state, reward, done, info

    def get_makespan(self) -> float:
        return float(self.makespan)


class PPOScheduler:
    def __init__(
        self,
        policy: torch.nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.device = device

        self.episode_rewards: List[float] = []
        self.episode_makespans: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropies: List[float] = []

    def _forward_policy(self, state) -> Dict[str, torch.Tensor]:
        node_features = state["node_features"].to(self.device)
        edge_index = state["edge_index"].to(self.device)
        mask = state["mask"].to(self.device)

        if isinstance(self.policy, MuxJobShopGNNPolicy):
            edge_indices = {
                "precedence": state["precedence_edge_index"].to(self.device),
                "machine": state["machine_edge_index"].to(self.device),
            }
            return self.policy(node_features, edge_index=edge_index, mask=mask, edge_indices=edge_indices)

        return self.policy(node_features, edge_index, mask)

    def select_action(self, state, deterministic: bool = False):
        with torch.no_grad():
            out = self._forward_policy(state)
            logits = out["action_logits"]
            value = out["value"]
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item())

    def compute_gae(self, transitions: List[Transition]):
        rewards = [t.reward for t in transitions]
        values = [t.value for t in transitions]
        dones = [t.done for t in transitions]

        advantages = [0.0] * len(transitions)
        gae = 0.0

        for t in reversed(range(len(transitions))):
            next_value = 0.0 if t == len(transitions) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1.0 - float(dones[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages[t] = gae

        returns = [advantages[i] + values[i] for i in range(len(transitions))]
        return advantages, returns

    def update(self, transitions: List[Transition]):
        advantages, returns = self.compute_gae(transitions)

        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor([t.action for t in transitions], dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32, device=self.device)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = len(transitions)
        if n == 0:
            return

        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n, device=self.device)

            for start in range(0, n, self.minibatch_size):
                idx = perm[start:start + self.minibatch_size]

                losses = []
                p_losses = []
                v_losses = []
                ents = []

                for j in idx.tolist():
                    tr = transitions[j]
                    tr_state = {
                        "node_features": tr.node_features,
                        "edge_index": tr.edge_index,
                        "precedence_edge_index": tr.precedence_edge_index,
                        "machine_edge_index": tr.machine_edge_index,
                        "mask": tr.mask,
                    }
                    out = self._forward_policy(tr_state)
                    logits = out["action_logits"]
                    value = out["value"].squeeze()
                    dist = torch.distributions.Categorical(logits=logits)

                    new_log_prob = dist.log_prob(actions_t[j])
                    entropy = dist.entropy()

                    ratio = torch.exp(new_log_prob - old_log_probs_t[j])
                    surr1 = ratio * advantages_t[j]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_t[j]
                    policy_loss = -torch.min(surr1, surr2)
                    value_loss = F.mse_loss(value, returns_t[j])

                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                    losses.append(loss)
                    p_losses.append(policy_loss.detach())
                    v_losses.append(value_loss.detach())
                    ents.append(entropy.detach())

                batch_loss = torch.stack(losses).mean()
                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                epoch_policy_losses.append(torch.stack(p_losses).mean().item())
                epoch_value_losses.append(torch.stack(v_losses).mean().item())
                epoch_entropies.append(torch.stack(ents).mean().item())

        self.policy_losses.append(float(np.mean(epoch_policy_losses)))
        self.value_losses.append(float(np.mean(epoch_value_losses)))
        self.entropies.append(float(np.mean(epoch_entropies)))

    def train_episode(self, instance: JobShopInstance):
        env = JobShopEnv(instance)
        state = env.reset()
        done = False
        episode_reward = 0.0
        transitions: List[Transition] = []

        while not done:
            action, log_prob, value = self.select_action(state, deterministic=False)
            next_state, reward, done, _ = env.step(action)

            transitions.append(
                Transition(
                    node_features=state["node_features"].detach().cpu(),
                    edge_index=state["edge_index"].detach().cpu(),
                    precedence_edge_index=state["precedence_edge_index"].detach().cpu(),
                    machine_edge_index=state["machine_edge_index"].detach().cpu(),
                    mask=state["mask"].detach().cpu(),
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=log_prob,
                    done=done,
                )
            )

            state = next_state
            episode_reward += reward

        makespan = env.get_makespan()
        self.update(transitions)
        self.episode_rewards.append(episode_reward)
        self.episode_makespans.append(makespan)
        return episode_reward, makespan

    def train(self, instances: List[JobShopInstance], epochs: int = 30, verbose: bool = True):
        best_avg_makespan = float("inf")

        for epoch in range(epochs):
            ep_rewards = []
            ep_makespans = []

            for instance in instances:
                r, m = self.train_episode(instance)
                ep_rewards.append(r)
                ep_makespans.append(m)

            avg_r = float(np.mean(ep_rewards))
            avg_m = float(np.mean(ep_makespans))
            best_avg_makespan = min(best_avg_makespan, avg_m)

            if verbose:
                print(
                    f"Epoch {epoch + 1:03d}/{epochs} | "
                    f"avg_reward={avg_r:8.3f} | "
                    f"avg_makespan={avg_m:8.3f} | "
                    f"best={best_avg_makespan:8.3f} | "
                    f"policy_loss={self.policy_losses[-1]:8.4f} | "
                    f"value_loss={self.value_losses[-1]:8.4f}"
                )

        return {
            "best_avg_makespan": best_avg_makespan,
            "episode_rewards": self.episode_rewards,
            "episode_makespans": self.episode_makespans,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "entropies": self.entropies,
        }

    def evaluate(self, instance: JobShopInstance):
        env = JobShopEnv(instance)
        state = env.reset()
        done = False

        while not done:
            action, _, _ = self.select_action(state, deterministic=True)
            state, _, done, _ = env.step(action)

        return env.schedule, env.get_makespan()

    def save_checkpoint(self, checkpoint_path: str, metadata: Optional[Dict] = None) -> None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        payload = {
            "model_state_dict": self.policy.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(payload, checkpoint_path)


def build_policy(model_type: str, node_feature_dim: int, hidden_dim: int, num_heads: int, num_layers: int):
    if model_type == "muxgnn":
        return MuxJobShopGNNPolicy(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
    return JobShopGNNPolicy(
        node_feature_dim=node_feature_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )


def evaluate_with_model(
    checkpoint_path: str,
    instances: List[JobShopInstance],
    device: str = "cpu",
) -> Dict[str, float]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device)
    metadata = payload.get("metadata", {})

    model_type = metadata.get("model_type", "gnn")
    node_feature_dim = int(metadata.get("node_feature_dim", 8))
    hidden_dim = int(metadata.get("hidden_dim", 64))
    num_heads = int(metadata.get("num_heads", 4))
    num_layers = int(metadata.get("num_layers", 3))

    policy = build_policy(
        model_type=model_type,
        node_feature_dim=node_feature_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    policy.load_state_dict(payload["model_state_dict"])

    evaluator = PPOScheduler(policy=policy, device=device)
    evaluator.policy.eval()

    makespans: List[float] = []
    for instance in instances:
        _, makespan = evaluator.evaluate(instance)
        makespans.append(float(makespan))

    return {
        "avg_makespan": float(np.mean(makespans)) if makespans else float("inf"),
        "best_makespan": float(np.min(makespans)) if makespans else float("inf"),
        "worst_makespan": float(np.max(makespans)) if makespans else float("inf"),
        "num_instances": float(len(makespans)),
    }

def get_instances(instances_file: str, max_instances: int, seed: int) -> List[JobShopInstance]:
    instances: List[JobShopInstance]
    if instances_file and os.path.exists(instances_file):
        instances = load_instances_from_json(instances_file)
    else:
        instances = generate_general_instances(
            num_instances=max_instances,
            num_jobs_range=(3, 7),
            num_machines_range=(3, 6),
            num_op_range=(3, 6),
            seed=seed,
        )

    if max_instances > 0:
        instances = instances[:max_instances]

    return instances


def main():
    parser = argparse.ArgumentParser(description="Corrected PPO + GNN trainer for Job Shop Scheduling")
    parser.add_argument("--instances-file", type=str, default="instances.json")
    parser.add_argument("--max-instances", type=int, default=50)
    parser.add_argument("--eval-instances", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model-type", type=str, choices=["gnn", "muxgnn"], default="muxgnn")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/muxgnn_ppo.pt")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    instances = get_instances(args.instances_file, args.max_instances, args.seed)
    if len(instances) == 0:
        raise ValueError("No instances found for training.")

    if args.eval_only:
        eval_instances = instances[: max(1, args.eval_instances)]
        print(f"Evaluating {len(eval_instances)} instances using: {args.checkpoint_path}")
        print("Use evaluate_model.py for a better evaluation interface:")
        print(f"  python evaluate_model.py {args.checkpoint_path} --max-instances {len(eval_instances)}")
        return

    policy = build_policy(
        model_type=args.model_type,
        node_feature_dim=8,  # Match JobShopEnv in new_ppo.py (8 features)
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )

    trainer = PPOScheduler(
        policy=policy,
        lr=args.lr,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        device=device,
    )

    print("=" * 70)
    print("Training PPO + GNN for Job Shop Scheduling")
    print(f"Device: {device}")
    print(f"Instances: {len(instances)}")
    print(f"Model type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print("=" * 70)

    stats = trainer.train(instances, epochs=args.epochs, verbose=True)

    trainer.save_checkpoint(
        checkpoint_path=args.checkpoint_path,
        metadata={
            "model_type": args.model_type,
            "node_feature_dim": 8,  # Match JobShopEnv in new_ppo.py (8 features)
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "seed": args.seed,
        },
    )

    print("=" * 70)
    print(f"Best avg makespan: {stats['best_avg_makespan']:.3f}")
    print(f"Saved model: {args.checkpoint_path}")
    print("\nTo evaluate the model on new instances, run:")
    print(f"  python evaluate_model.py {args.checkpoint_path} --max-instances 10")
    print("=" * 70)


def small_main():
    parser = argparse.ArgumentParser(description="Corrected PPO + GNN trainer for Job Shop Scheduling")
    parser.add_argument("--instances-file", type=str, default="instances.json")
    parser.add_argument("--max-instances", type=int, default=50)
    parser.add_argument("--eval-instances", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model-type", type=str, choices=["gnn", "muxgnn"], default="muxgnn")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/small_muxgnn_ppo.pt")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    instances = get_instances(args.instances_file, args.max_instances, args.seed)
    if len(instances) == 0:
        raise ValueError("No instances found for training.")

    if args.eval_only:
        eval_instances = instances[: max(1, args.eval_instances)]
        print(f"Evaluating {len(eval_instances)} instances using: {args.checkpoint_path}")
        print("Use evaluate_model.py for a better evaluation interface:")
        print(f"  python evaluate_model.py {args.checkpoint_path} --max-instances {len(eval_instances)}")
        return

    policy = build_policy(
        model_type=args.model_type,
        node_feature_dim=8,  # Match JobShopEnv in new_ppo.py (8 features)
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )

    trainer = PPOScheduler(
        policy=policy,
        lr=args.lr,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        device=device,
    )

    print("=" * 70)
    print("Training PPO + GNN for Job Shop Scheduling")
    print(f"Device: {device}")
    print(f"Instances: {len(instances)}")
    print(f"Model type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print("=" * 70)

    stats = trainer.train(instances, epochs=args.epochs, verbose=True)

    trainer.save_checkpoint(
        checkpoint_path=args.checkpoint_path,
        metadata={
            "model_type": args.model_type,
            "node_feature_dim": 8,  # Match JobShopEnv in new_ppo.py (8 features)
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "seed": args.seed,
        },
    )

    print("=" * 70)
    print(f"Best avg makespan: {stats['best_avg_makespan']:.3f}")
    print(f"Saved model: {args.checkpoint_path}")
    print("\nTo evaluate the model on new instances, run:")
    print(f"  python evaluate_model.py {args.checkpoint_path} --max-instances 10")
    print("=" * 70)


def train_gnn():
    parser = argparse.ArgumentParser(description="Corrected PPO + GNN trainer for Job Shop Scheduling")
    parser.add_argument("--instances-file", type=str, default="instances.json")
    parser.add_argument("--max-instances", type=int, default=30)
    parser.add_argument("--eval-instances", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model-type", type=str, choices=["gnn", "muxgnn"], default="gnn")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/small_gnn_ppo.pt")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    instances = get_instances(args.instances_file, args.max_instances, args.seed)
    if len(instances) == 0:
        raise ValueError("No instances found for training.")

    if args.eval_only:
        eval_instances = instances[: max(1, args.eval_instances)]
        print(f"Evaluating {len(eval_instances)} instances using: {args.checkpoint_path}")
        print("Use evaluate_model.py for a better evaluation interface:")
        print(f"  python evaluate_model.py {args.checkpoint_path} --max-instances {len(eval_instances)}")
        return

    policy = build_policy(
        model_type=args.model_type,
        node_feature_dim=8,  # Match JobShopEnv in new_ppo.py (8 features)
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )

    trainer = PPOScheduler(
        policy=policy,
        lr=args.lr,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        device=device,
    )

    print("=" * 70)
    print("Training PPO + GNN for Job Shop Scheduling")
    print(f"Device: {device}")
    print(f"Instances: {len(instances)}")
    print(f"Model type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print("=" * 70)

    stats = trainer.train(instances, epochs=args.epochs, verbose=True)

    trainer.save_checkpoint(
        checkpoint_path=args.checkpoint_path,
        metadata={
            "model_type": args.model_type,
            "node_feature_dim": 8,  # Match JobShopEnv in new_ppo.py (8 features)
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "lr": args.lr,
            "seed": args.seed,
        },
    )

    print("=" * 70)
    print(f"Best avg makespan: {stats['best_avg_makespan']:.3f}")
    print(f"Saved model: {args.checkpoint_path}")
    print("\nTo evaluate the model on new instances, run:")
    print(f"  python evaluate_model.py {args.checkpoint_path} --max-instances 10")
    print("=" * 70)


if __name__ == "__main__":
    train_gnn()
