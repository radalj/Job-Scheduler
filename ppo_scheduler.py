"""
ppo_scheduler.py
================
Trains a PPO agent on the Job Shop Scheduling Problem using:
  - Stable-Baselines3 PPO (no PPO implemented from scratch)
  - A custom GNN features extractor (BaseFeaturesExtractor) that wraps
    JobShopGNNPolicy from GNN.py — all its weights (SimpleGNN backbone,
    policy_head, value_head) are updated by PPO every gradient step.

Key design (from SB3 docs: custom_policy.html):
  ┌──────────────────────────────────────────────────────┐
  │  Dict obs (node_features, edge_index, mask)          │
  │                   ↓                                  │
  │   GNNFeaturesExtractor                               │
  │     └─ JobShopGNNPolicy  (GNN.py)  ← trained by PPO │
  │           └─ SimpleGNN (GATLayer stack)              │
  │                   ↓  [batch, hidden_dim]             │
  │   SB3 Actor MLP  →  action logits                    │
  │   SB3 Critic MLP →  value estimate                   │
  └──────────────────────────────────────────────────────┘

Usage:
    python ppo_scheduler.py [--timesteps 100000] [--jobs 5] [--machines 5]
"""

from __future__ import annotations

import argparse
import os
import json
import time
from typing import Dict

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import obs_as_tensor

from generator import generate_general_instances
from GNN import JobShopGNNPolicy
from jobshop_gym_wrapper import JobShopGymEnv
from generator import load_instances_from_json


# ============================================================
# Custom PPO with Makespan Loss
# ============================================================

class MakespanCallback:
    """
    Callback that tracks makespans and can apply additional loss.
    This is a simpler approach than modifying PPO internals.
    """
    def __init__(self, makespan_coef=0.1):
        self.makespan_coef = makespan_coef
        self.recent_makespans = []
        self.episode_count = 0
    
    def __call__(self, locals_dict, globals_dict):
        # Track makespans from completed episodes
        if 'infos' in locals_dict:
            for info in locals_dict['infos']:
                if 'episode' in info and 'makespan' in info:
                    makespan = info['makespan']
                    self.recent_makespans.append(makespan)
                    self.episode_count += 1
                    
                    if self.episode_count % 10 == 0:
                        avg_makespan = np.mean(self.recent_makespans[-10:])
                        print(f"  Last 10 episodes avg makespan: {avg_makespan:.1f}")
        
        return True

# For now, we'll use standard PPO with makespan tracking via callback
# The direct loss modification needs deeper SB3 integration
class MakespanPPO(PPO):
    """Placeholder - using standard PPO with makespan tracking for now"""
    def __init__(self, *args, makespan_coef=0.1, **kwargs):
        self.makespan_coef = makespan_coef
        super().__init__(*args, **kwargs)



# ============================================================
# 1.  GNN Feature Extractor
#     Registered as the features_extractor for MultiInputPolicy.
#     JobShopGNNPolicy from GNN.py lives here → all its weights
#     (GATLayers, graph_pooling, policy_head, value_head) are
#     updated by PPO's optimizer every gradient step.
# ============================================================

class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor that runs JobShopGNNPolicy (from GNN.py)
    over the padded Dict-observation produced by JobShopGymEnv.

    Internally, JobShopGNNPolicy contains:
      - SimpleGNN (GATLayer stack + graph pooling)
      - policy_head  (actor linear layer)
      - value_head   (critic linear layer)
    All of these weights live inside this extractor and are trained
    end-to-end by SB3's PPO optimizer.

    The extractor exposes the graph_embedding (output of SimpleGNN's
    mean-pool + graph_pooling layer) to SB3's actor/critic MLP heads.

    Input (per sample inside a batch):
        node_features   : [num_ops * node_feat_dim]  (flattened)
        edge_index_flat : [2 * max_edges]            (padded src then dst)
        num_edges       : [1]                        (actual edge count)
        valid_mask      : [num_ops]                  (ignored here, used by actor)

    Output:
        [batch_size, gnn_hidden_dim]
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        max_ops: int,
        node_feat_dim: int = 7,
        gnn_hidden_dim: int = 64,
        gnn_num_heads: int = 2,
        gnn_num_layers: int = 1,
    ):
        # features_dim is the size of the vector returned by forward()
        super().__init__(observation_space, features_dim=gnn_hidden_dim)

        self.max_ops = max_ops
        self.node_feat_dim = node_feat_dim
        self.max_edges = observation_space["edge_index_flat"].shape[0] // 2

        # ---- Full JobShopGNNPolicy from GNN.py (weights updated by PPO) ----
        self.gnn_policy = JobShopGNNPolicy(
            node_feature_dim=node_feat_dim,
            hidden_dim=gnn_hidden_dim,
            num_heads=gnn_num_heads,
            num_layers=gnn_num_layers,
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """
        observations: dict of batched tensors from SB3's rollout buffer.
        Returns: [batch_size, gnn_hidden_dim]
        """
        batch_nf   = observations["node_features"]    # [B, max_ops*feat_dim]
        batch_ei   = observations["edge_index_flat"]  # [B, 2*max_edges]
        batch_ne   = observations["num_edges"]        # [B, 1]
        batch_an   = observations["actual_num_ops"]   # [B, 1]

        batch_size = batch_nf.shape[0]
        device = batch_nf.device

        graph_embeddings = []
        for i in range(batch_size):
            actual_ops = int(batch_an[i].item())

            # Slice only real nodes (discard padding zeros)
            nf_full = batch_nf[i].view(self.max_ops, self.node_feat_dim)
            nf = nf_full[:actual_ops]  # [actual_ops, feat_dim]

            # Reconstruct edge_index: [2, num_edges]
            ne = int(batch_ne[i].item())
            if ne > 0:
                src = batch_ei[i, :ne].long()          # [ne]
                dst = batch_ei[i, self.max_edges: self.max_edges + ne].long()
                edge_index = th.stack([src, dst], dim=0)  # [2, ne]
            else:
                edge_index = th.zeros((2, 0), dtype=th.long, device=device)

            # Run through JobShopGNNPolicy's internal SimpleGNN.
            # gnn_policy.gnn is the SimpleGNN (GATLayer stack + pooling)
            # that forms the shared backbone of actor and critic.
            gnn_out = self.gnn_policy.gnn(nf, edge_index)
            graph_embeddings.append(gnn_out["graph_embedding"])  # [hidden_dim]

        return th.stack(graph_embeddings, dim=0)  # [B, hidden_dim]


# ============================================================
# 2.  Environment factory
# ============================================================

def make_env_fn(instance, reward_mode="dense", seed=0):
    """Return a callable that creates a single monitored gymnasium env."""
    def _init():
        env = JobShopGymEnv(instance, reward_mode=reward_mode)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


# ============================================================
# 3.  Training script
# ============================================================

def train(args):
    global max_ops  # Ensure max_ops is accessible globally
    print("=" * 65)
    print("Job Shop PPO trainer  (SB3 + GNN feature extractor) - FAST MODE")
    print("=" * 65)
    start_time = time.time()

    # ---- generate instances ----
    # Generate smaller instances to fit max_ops=800 and train faster
    instances = generate_general_instances(
        num_instances=args.num_instances,
        num_jobs_range=(3, 30),      # Reduced job range
        num_machines_range=(3, 15),   # Reduced machine range  
        num_op_range=(3, 20),        # Reduced operations per job
        seed=args.seed
    )

    # # Generate instances with specific num_ops 
    # specific_instances = generate_general_instances(
    #     num_instances=2, num_jobs_range=(60, 65), num_machines_range=(10, 25),
    #     num_op_range=(25, 30), seed=42
    # )

    # # Combine specific instances with general instances
    # instances.extend(specific_instances)

    # # Log the inclusion of specific instances
    # print("Added training instances with 1000 and 1500 operations.")

    # Shared max_ops: pad every env to the same obs/action space size
    max_ops = 1500  # Reduced from 1500 for faster training and less memory
    print(f"\nInstances      : {len(instances)}")
    print(f"max_ops (pad)  : {max_ops}")
    print(f"Op range       : {min(i.num_operations for i in instances)} "
          f"– {max(i.num_operations for i in instances)}")

    # Save max_ops to a file
    with open("max_ops.json", "w") as f:
        json.dump({"max_ops": max_ops}, f)

    # ---- build environments ----
    def make_multi_env(seed=0):
        """Factory: env that samples from ALL instances on each reset."""
        def _init():
            env = JobShopGymEnv(
                instances=instances,
                max_ops=max_ops,
                reward_mode=args.reward_mode,
            )
            env = Monitor(env)
            env.reset(seed=seed)
            return env
        return _init

    train_env = make_vec_env(make_multi_env(seed=args.seed), n_envs=args.n_envs)  # Use parallel envs

    # Memory usage diagnostics
    sample_env = train_env.envs[0]
    obs_space = sample_env.observation_space
    print(f"\nMemory diagnostics:")
    print(f"  n_envs × n_steps = {args.n_envs} × {args.n_steps} = {args.n_envs * args.n_steps} buffer slots")
    
    total_obs_size = 0
    for key, space in obs_space.spaces.items():
        size = np.prod(space.shape)
        total_obs_size += size
        print(f"  {key:15s}: {space.shape} = {size:,} elements")
    
    buffer_size_gb = (total_obs_size * args.n_envs * args.n_steps * 4) / (1024**3)  # 4 bytes per float32
    print(f"  Total obs size  : {total_obs_size:,} elements")  
    print(f"  Buffer memory   : {buffer_size_gb:.2f} GB")
    
    if buffer_size_gb > 8:
        print(f"  WARNING: Large buffer size detected!")

    # Eval env also spans all instances (use same reward mode as training)
    eval_env = Monitor(
        JobShopGymEnv(instances=instances, max_ops=max_ops, reward_mode=args.reward_mode)
    )

    from jobshop_env import JobShopEnv as _JE
    node_feat_dim = _JE.NODE_FEATURE_DIM

    # ---- policy kwargs: plug in our GNN extractor ----
    policy_kwargs = dict(
        features_extractor_class=GNNFeaturesExtractor,
        features_extractor_kwargs=dict(
            max_ops=max_ops,
            node_feat_dim=node_feat_dim,
            gnn_hidden_dim=32,  # Increased capacity for better learning
            gnn_num_heads=2,    # More attention heads
            gnn_num_layers=3,   # Deeper network
        ),
        # Larger actor/critic MLP on top of GNN output
        net_arch=dict(
            pi=[32, 32],  # Deeper policy network
            vf=[8],       # Minimal value network since it's nearly disabled
        ),
        # Don't share features - let policy focus on action selection
        share_features_extractor=False,
    )

    # ---- Custom PPO with Makespan Loss ----
    print(f"Value function coef: {args.vf_coef} (low values disable value learning)")
    print(f"Makespan loss coef: {args.makespan_coef} (direct makespan² optimization)")
    model = MakespanPPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.95,  # Lower gamma for sparse rewards
        gae_lambda=0.9,  # Lower GAE lambda  
        clip_range=0.2,
        ent_coef=0.02,  # Increased exploration
        vf_coef=args.vf_coef,  # User-configurable - nearly disable value function for sparse rewards!
        max_grad_norm=0.5,
        makespan_coef=args.makespan_coef,  # New: direct makespan loss coefficient!
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=args.log_dir if args.tensorboard else None,
        seed=args.seed,
        device="auto",  # Automatically use GPU if available
    )

    print(f"\nPolicy:\n{model.policy}\n")
    total_params      = sum(p.numel() for p in model.policy.parameters())
    gnn_policy_params = sum(p.numel()
                            for p in model.policy.features_extractor.gnn_policy.parameters())
    backbone_params   = sum(p.numel()
                            for p in model.policy.features_extractor.gnn_policy.gnn.parameters())
    print(f"max_ops (action/obs pad)    : {max_ops}")
    print(f"Total policy params         : {total_params:,}")
    print(f"  JobShopGNNPolicy (GNN.py) : {gnn_policy_params:,}  (updated by PPO)")
    print(f"    of which SimpleGNN      : {backbone_params:,}")

    # ---- callbacks ----
    os.makedirs(args.log_dir, exist_ok=True)
    callbacks = []

    if not args.fast_mode:
        # Early stopping if performance is good enough (optional)
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=-2.0,  # Adjust based on your problem scale
            verbose=1
        )
        
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(args.log_dir, "best"),
            log_path=args.log_dir,
            eval_freq=max(args.eval_freq, 1),
            n_eval_episodes=3,  # Reduced from 5 for faster evaluation
            deterministic=True,
            verbose=1,
            callback_after_eval=stop_callback,  # Properly wrap stop callback
        )
        callbacks.append(eval_cb)
    else:
        print("Fast mode: Skipping evaluation during training")

    ckpt_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix="jobshop_ppo",
        verbose=0,
    )
    callbacks.append(ckpt_cb)

    # Add makespan tracking callback
    makespan_cb = MakespanCallback(makespan_coef=args.makespan_coef)
    callbacks.append(makespan_cb)

    # ---- train ----
    print(f"\nTraining for {args.timesteps:,} timesteps with {args.n_envs} parallel environments...")
    print(f"Batch size: {args.batch_size}, Steps per rollout: {args.n_steps}, Learning rate: {args.lr}")
    print(f"Model capacity: {gnn_policy_params:,} GNN params, Reward mode: {args.reward_mode}")
    
    # Add a custom callback to monitor training progress
    class ProgressLogger:
        def __init__(self):
            self.step_count = 0
            self.episode_rewards = []
        
        def __call__(self, locals_dict, globals_dict):
            self.step_count += 1
            if 'infos' in locals_dict:
                for info in locals_dict['infos']:
                    if 'episode' in info:
                        reward = info['episode']['r']
                        self.episode_rewards.append(reward)
                        if len(self.episode_rewards) % 10 == 0:
                            recent_avg = np.mean(self.episode_rewards[-10:])
                            print(f"  Step {self.step_count}: Last 10 episodes avg reward: {recent_avg:.2f}")
            return True
    
    progress_logger = ProgressLogger()
    
    train_start = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    train_time = time.time() - train_start
    print(f"\nTraining completed in {train_time:.1f} seconds ({train_time/60:.1f} minutes)")

    # ---- save ----
    save_path = os.path.join(args.log_dir, "final_model")
    model.save(save_path)
    print(f"\nModel saved to: {save_path}.zip")

    # ---- evaluate ----
    print("\nFinal evaluation (10 episodes)...")
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"  Mean reward : {mean_r:.2f} ± {std_r:.2f}")

    print(f"\nTotal runtime: {time.time() - start_time:.1f} seconds")
    return model


# ============================================================
# 4.  Entry point
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="PPO + GNN for Job Shop Scheduling")

    # Instance
    p.add_argument("--num_instances",type=int, default=500)  # Reduced from 4000 for faster training
    p.add_argument("--seed",         type=int, default=435)

    # GNN
    p.add_argument("--hidden_dim",   type=int, default=16)
    p.add_argument("--num_heads",    type=int, default=2)
    p.add_argument("--num_layers",   type=int, default=2)

    # PPO
    p.add_argument("--timesteps",    type=int, default=50_000)  # Much more training needed
    p.add_argument("--lr",           type=float, default=3e-4)  # Lower, more stable learning rate
    p.add_argument("--n_steps",      type=int, default=128)     # Keep for memory efficiency
    p.add_argument("--batch_size",   type=int, default=32)      # Keep for memory efficiency
    p.add_argument("--n_epochs",     type=int, default=10)      # More epochs for better learning
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--vf_coef",      type=float, default=0.001, help="Value function coefficient - set very low for sparse rewards")
    p.add_argument("--makespan_coef", type=float, default=0.1, help="Makespan² loss coefficient - direct optimization of scheduling objective")
    p.add_argument("--reward_mode",  type=str, default="makespan_squared", 
                   choices=["sparse", "dense", "shaped", "makespan_squared"],
                   help="Reward mode: sparse (final only), dense (step penalties), shaped (comprehensive), makespan_squared (direct makespan² loss!)")

    # Logging  
    p.add_argument("--log_dir",         type=str, default="./logs/ppo_jobshop")
    p.add_argument("--tensorboard",     action="store_true")
    p.add_argument("--eval_freq",       type=int, default=2000)   # More frequent eval for faster feedback
    p.add_argument("--checkpoint_freq", type=int, default=5000)    # More frequent checkpoints
    p.add_argument("--n_envs",          type=int, default=2)       # Reduced parallel environments for memory efficiency
    p.add_argument("--fast_mode",       action="store_true", help="Skip evaluation during training for faster iteration")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Check if training is required
    final_model_path = os.path.join(args.log_dir, "final_model.zip")
    max_ops_path = "max_ops.json"

    if os.path.exists(final_model_path) and os.path.exists(max_ops_path):
        print("Final model found. Skipping training phase.")
        model = PPO.load(final_model_path)

        # Load max_ops from the file
        with open(max_ops_path, "r") as f:
            max_ops = json.load(f)["max_ops"]
    else:
        model = train(args)

    # after training test the model on saved instance.json file
    # save the results to results.txt file (end time of the schedule)
    # load the instance from json file
    test_instances = load_instances_from_json("instances.json")

    results_lines = []
    i = 0
    for test_instance in test_instances:
        i += 1
        if i % 10 != 0:  # Test every 10th instance instead of every 100th
            continue
        # Wrap each test instance with the same max_ops used during training
        test_env = JobShopGymEnv(
            instances=test_instance,
            max_ops=max_ops,
            reward_mode="sparse",
        )
        obs, _ = test_env.reset()          # gymnasium: (obs, info)
        done = False
        reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated

        makespan = info.get("makespan", -reward)
        line = f"Instance: {test_instance} | Makespan: {makespan}"
        print(line)
        results_lines.append(line)

    with open("gnn_results.txt", "w") as f:
        f.write("\n".join(results_lines) + "\n")
    print("\nResults written to gnn_results.txt")