#!/usr/bin/env python3
"""
Resumable evaluation for muxGNN.

- Writes one line per evaluated instance to muxGNN_result.txt
- On startup, counts already written lines and resumes from there
"""

import argparse
import os
import random

import torch

from evaluate_model import build_policy, load_model
from generator import load_instances_from_json
from muxGNN import MuxJobShopGNNPolicy
from new_ppo import JobShopEnv


def evaluate_instance_8dim(policy, instance, device="cpu"):
    policy.eval()
    env = JobShopEnv(instance)
    state = env.reset()
    done = False

    with torch.no_grad():
        while not done:
            node_features = state["node_features"].to(device)
            edge_index = state["edge_index"].to(device)
            mask = state["mask"].to(device)

            if isinstance(policy, MuxJobShopGNNPolicy):
                edge_indices = {
                    "precedence": state["precedence_edge_index"].to(device),
                    "machine": state["machine_edge_index"].to(device),
                }
                out = policy(node_features, edge_index=edge_index, mask=mask, edge_indices=edge_indices)
            else:
                out = policy(node_features, edge_index, mask)

            logits = out["action_logits"]
            action = torch.argmax(logits).item()

            if not mask[action].item():
                valid_actions = mask.nonzero(as_tuple=True)[0].tolist()
                if valid_actions:
                    action = random.choice(valid_actions)
                else:
                    break

            state, _, done, _ = env.step(action)

    return float(env.get_makespan())


def count_completed_results(output_file: str) -> int:
    if not os.path.exists(output_file):
        return 0

    completed = 0
    with open(output_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip().startswith("Instance:"):
                completed += 1
    return completed


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    parser = argparse.ArgumentParser(description="Resumable muxGNN evaluation to text file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/small_gnn_ppo.pt")
    parser.add_argument("--instances-file", type=str, default="instances.json")
    parser.add_argument("--output", type=str, default="gnn_results_end.txt")
    parser.add_argument("--device", type=str, default=device, choices=["cpu", "cuda"])
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    payload = load_model(args.checkpoint, device=args.device)
    metadata = payload.get("metadata", {})

    model_type = metadata.get("model_type", "muxgnn")
    node_feature_dim = int(metadata.get("node_feature_dim", 8))
    hidden_dim = int(metadata.get("hidden_dim", 64))
    num_heads = int(metadata.get("num_heads", 4))
    num_layers = int(metadata.get("num_layers", 3))

    policy = build_policy(model_type, node_feature_dim, hidden_dim, num_heads, num_layers, device=args.device)
    policy.load_state_dict(payload["model_state_dict"])
    policy.eval()

    instances = load_instances_from_json(args.instances_file)
    total_instances = len(instances)
    completed = count_completed_results(args.output)

    print(f"Total instances: {total_instances}")
    print(f"Already evaluated in {args.output}: {completed}")

    if completed >= total_instances:
        print("All instances are already evaluated. Nothing to do.")
        return

    print(f"Resuming from instance index: {completed} (1-based: {completed + 1})")

    with open(args.output, "a", encoding="utf-8") as file:
        for index in range(completed, total_instances):
            instance = instances[index]
            try:
                makespan = evaluate_instance_8dim(policy, instance, device=args.device)
                instance_name = getattr(instance, "name", "JobShopInstance")
                line = f"Instance: {instance} | Makespan: {int(makespan)}"
                print(f"{index + 1}/{total_instances} -> {line}")
            except Exception as exc:
                line = f"Instance: JobShopInstance | Error: {exc}"
                print(f"{index + 1}/{total_instances} -> FAILED: {exc}")

            file.write(line + "\n")
            file.flush()

    print(f"Done. Results are in {args.output}")


if __name__ == "__main__":
    main()
