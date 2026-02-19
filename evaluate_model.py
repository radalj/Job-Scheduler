#!/usr/bin/env python3
"""
evaluate_model.py
=================
Standalone script to evaluate a saved PPO+GNN model on job shop instances.

Usage:
  python evaluate_model.py checkpoints/muxgnn_ppo.pt --max-instances 10 --seed 42
"""

import argparse
import os
import random
from typing import Dict, List

import numpy as np
import torch

from GNN import JobShopGNNPolicy
from muxGNN import MuxJobShopGNNPolicy
from generator import load_instances_from_json, generate_general_instances
from jobshop import JobShopInstance
from jobshop_env import JobShopEnv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path: str, device: str) -> Dict:
    """Load model state and metadata from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device)
    return payload


def build_policy(model_type: str, node_feature_dim: int, hidden_dim: int, num_heads: int, num_layers: int, device: str):
    """Instantiate the appropriate policy class."""
    if model_type == "muxgnn":
        policy = MuxJobShopGNNPolicy(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
    else:
        policy = JobShopGNNPolicy(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
    return policy.to(device)


def evaluate_instance(policy, instance: JobShopInstance, device: str = "cpu") -> float:
    """Run policy on a single instance in deterministic mode (greedy) and return makespan."""
    policy.eval()

    env = JobShopEnv(instance)
    state = env.reset()
    done = False
    makespan = 0.0

    with torch.no_grad():
        while not done:
            node_features = state["node_features"].to(device)
            edge_index = state["edge_index"].to(device)
            # Use valid_mask from JobShopEnv (not mask)
            mask = state.get("valid_mask", state.get("mask")).to(device)

            # Forward pass through policy
            # Check if policy is MuxJobShopGNNPolicy
            if isinstance(policy, MuxJobShopGNNPolicy):
                # Use explicit edge_indices dict
                edge_indices = {
                    "precedence": state.get("precedence_edge_index", edge_index).to(device),
                    "machine": state.get("machine_edge_index", edge_index).to(device),
                }
                out = policy(node_features, edge_index=edge_index, edge_indices=edge_indices)
            else:
                out = policy(node_features, edge_index, mask)

            logits = out["action_logits"]

            # Greedy action selection
            action = torch.argmax(logits).item()

            # Check if valid, otherwise pick random valid action
            if not mask[action].item():
                valid_actions = mask.nonzero(as_tuple=True)[0].tolist()
                if valid_actions:
                    action = random.choice(valid_actions)
                else:
                    break  # No valid actions available

            state, _, done, info = env.step(action)
            makespan = info.get("makespan", 0.0)

    return float(makespan)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO+GNN model")
    parser.add_argument("checkpoint_path", type=str, help="Path to saved model checkpoint")
    parser.add_argument("--instances-file", type=str, default="instances.json", help="Path to instances JSON")
    parser.add_argument("--max-instances", type=int, default=10, help="Max instances to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint_path}")
    payload = load_model(args.checkpoint_path, device=args.device)
    metadata = payload.get("metadata", {})

    # Extract model configuration
    model_type = metadata.get("model_type", "gnn")
    node_feature_dim = int(metadata.get("node_feature_dim", 8))
    hidden_dim = int(metadata.get("hidden_dim", 64))
    num_heads = int(metadata.get("num_heads", 4))
    num_layers = int(metadata.get("num_layers", 3))

    print(f"  Model type: {model_type}")
    print(f"  Node feature dim: {node_feature_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num heads: {num_heads}")
    print(f"  Num layers: {num_layers}")

    # Build and load policy
    policy = build_policy(model_type, node_feature_dim, hidden_dim, num_heads, num_layers, device=args.device)
    policy.load_state_dict(payload["model_state_dict"])
    policy.eval()

    # Load instances
    print(f"\nLoading instances from {args.instances_file}")
    if os.path.exists(args.instances_file):
        instances = load_instances_from_json(args.instances_file)
    else:
        print(f"  File not found, generating {args.max_instances} instances...")
        instances = generate_general_instances(
            num_instances=args.max_instances,
            num_jobs_range=(3, 7),
            num_machines_range=(3, 6),
            num_op_range=(3, 6),
            seed=args.seed,
        )

    if len(instances) == 0:
        print("ERROR: No instances to evaluate")
        return

    # Take up to max_instances
    instances = instances[:max(1, args.max_instances)]
    print(f"Evaluating on {len(instances)} instances...")

    # Evaluate
    makespans: List[float] = []
    try:
        for i, instance in enumerate(instances):
            makespan = evaluate_instance(policy, instance, device=args.device)
            makespans.append(makespan)
            print(f"  Instance {i+1}/{len(instances)}: makespan={makespan:.1f}")
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")

    # Report metrics
    if makespans:
        print("\n" + "=" * 70)
        print("Evaluation Results")
        print("=" * 70)
        print(f"Instances evaluated: {len(makespans)}")
        print(f"Average makespan:    {np.mean(makespans):.3f}")
        print(f"Best makespan:       {np.min(makespans):.3f}")
        print(f"Worst makespan:      {np.max(makespans):.3f}")
        print(f"Std dev makespan:    {np.std(makespans):.3f}")
        print("=" * 70)
    else:
        print("No instances were successfully evaluated.")


if __name__ == "__main__":
    main()
