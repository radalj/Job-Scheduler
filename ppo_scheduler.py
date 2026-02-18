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
from typing import Dict

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from generator import generate_general_instances
from GNN import JobShopGNNPolicy
from jobshop_gym_wrapper import JobShopGymEnv


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
        gnn_num_layers: int = 3,
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
    print("Job Shop PPO trainer  (SB3 + GNN feature extractor)")
    print("=" * 65)

    # ---- generate instances ----
    instances = generate_general_instances(
        num_instances=args.num_instances,
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
    max_ops = 1500  # Reduced from 1500 to 1000
    print(f"\nInstances      : {len(instances)}")
    print(f"max_ops (pad)  : {max_ops}")
    print(f"Op range       : {min(i.num_operations for i in instances)} "
          f"– {max_ops}")

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
                reward_mode="dense",
            )
            env = Monitor(env)
            env.reset(seed=seed)
            return env
        return _init

    train_env = make_vec_env(make_multi_env(seed=args.seed), n_envs=1)

    # Eval env also spans all instances
    eval_env = Monitor(
        JobShopGymEnv(instances=instances, max_ops=max_ops, reward_mode="sparse")
    )

    from jobshop_env import JobShopEnv as _JE
    node_feat_dim = _JE.NODE_FEATURE_DIM

    # ---- policy kwargs: plug in our GNN extractor ----
    policy_kwargs = dict(
        features_extractor_class=GNNFeaturesExtractor,
        features_extractor_kwargs=dict(
            max_ops=max_ops,
            node_feat_dim=node_feat_dim,
            gnn_hidden_dim=32,  # Reduced from 64 to 32
            gnn_num_heads=2,
            gnn_num_layers=2,
        ),
        # Small actor/critic MLP on top of GNN output
        net_arch=dict(
            pi=[32],  # Reduced from [64] to [32]
            vf=[32],  # Reduced from [64] to [32]
        ),
        # share the GNN extractor between actor and critic (default for on-policy)
        share_features_extractor=True,
    )

    # ---- PPO model (SB3, not from scratch) ----
    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=args.log_dir if args.tensorboard else None,
        seed=args.seed,
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

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.log_dir, "best"),
        log_path=args.log_dir,
        eval_freq=max(args.eval_freq, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    callbacks.append(eval_cb)

    ckpt_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix="jobshop_ppo",
        verbose=0,
    )
    callbacks.append(ckpt_cb)

    # ---- train ----
    print(f"\nTraining for {args.timesteps:,} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

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

    return model


# ============================================================
# 4.  Entry point
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="PPO + GNN for Job Shop Scheduling")

    # Instance
    p.add_argument("--num_instances",type=int, default=10)
    p.add_argument("--seed",         type=int, default=400)

    # GNN
    p.add_argument("--hidden_dim",   type=int, default=32)
    p.add_argument("--num_heads",    type=int, default=2)
    p.add_argument("--num_layers",   type=int, default=3)

    # PPO
    p.add_argument("--timesteps",    type=int, default=1_000)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--n_steps",      type=int, default=64)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--n_epochs",     type=int, default=10)
    p.add_argument("--gamma",        type=float, default=0.99)

    # Logging
    p.add_argument("--log_dir",         type=str, default="./logs/ppo_jobshop")
    p.add_argument("--tensorboard",     action="store_true")
    p.add_argument("--eval_freq",       type=int, default=5000)
    p.add_argument("--checkpoint_freq", type=int, default=20_000)

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
    from generator import load_instances_from_json
    test_instances = load_instances_from_json("instances.json")

    results_lines = []
    i = 0
    for test_instance in test_instances:
        i += 1
        if (test_instance.num_operations >= 100):
            break
        if (i % 15) != 0:
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