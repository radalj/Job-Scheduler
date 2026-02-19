#!/usr/bin/env python3
import argparse
import random
from typing import Dict, List

import numpy as np
import torch

from evaluate_model import load_model, build_policy
from generator import load_instances_from_json
from new_ppo import JobShopEnv
from random_scheduler import random_schedule
from muxGNN import MuxJobShopGNNPolicy


def schedule_makespan(instance, schedule) -> float:
    op_lookup = {}
    for job in instance.jobs:
        for op in job:
            op_lookup[op.operation_id] = op

    makespan = 0.0
    for op_id, start_time in schedule.schedule.items():
        op = op_lookup[op_id]
        makespan = max(makespan, float(start_time + op.duration))
    return makespan


def evaluate_mux_instance(policy, instance, device: str = "cpu") -> float:
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


def main():
    parser = argparse.ArgumentParser(description="Compare MuxGNN vs Random scheduler on instances")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/muxgnn_full.pt")
    parser.add_argument("--instances-file", type=str, default="instances.json")
    parser.add_argument("--max-instances", type=int, default=0, help="0 means all instances")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    payload = load_model(args.checkpoint, device="cpu")
    metadata = payload.get("metadata", {})

    model_type = metadata.get("model_type", "muxgnn")
    node_feature_dim = int(metadata.get("node_feature_dim", 8))
    hidden_dim = int(metadata.get("hidden_dim", 64))
    num_heads = int(metadata.get("num_heads", 4))
    num_layers = int(metadata.get("num_layers", 3))

    policy = build_policy(model_type, node_feature_dim, hidden_dim, num_heads, num_layers, device="cpu")
    policy.load_state_dict(payload["model_state_dict"])
    policy.eval()

    instances = load_instances_from_json(args.instances_file)
    if args.max_instances > 0:
        instances = instances[:args.max_instances]

    mux_makespans: List[float] = []
    random_makespans: List[float] = []

    total = len(instances)
    print(f"Comparing on {total} instances")
    print("-" * 90)

    for i, instance in enumerate(instances, start=1):
        try:
            mux_m = evaluate_mux_instance(policy, instance, device="cpu")
            rand_schedule = random_schedule(instance)
            rand_m = schedule_makespan(instance, rand_schedule)

            mux_makespans.append(mux_m)
            random_makespans.append(rand_m)

            mux_avg = float(np.mean(mux_makespans))
            rand_avg = float(np.mean(random_makespans))

            print(
                f"Instance {i}/{total} | "
                f"muxgnn={mux_m:.1f} | random={rand_m:.1f} | "
                f"avg_muxgnn={mux_avg:.2f} | avg_random={rand_avg:.2f}"
            )
        except Exception as exc:
            print(f"Instance {i}/{total} | FAILED: {exc}")

    print("=" * 90)
    print(f"Final average muxgnn makespan: {float(np.mean(mux_makespans)) if mux_makespans else float('nan'):.3f}")
    print(f"Final average random makespan: {float(np.mean(random_makespans)) if random_makespans else float('nan'):.3f}")
    print(f"Instances compared: {min(len(mux_makespans), len(random_makespans))}")
    print("=" * 90)


if __name__ == "__main__":
    main()
