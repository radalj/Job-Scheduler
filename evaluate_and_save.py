#!/usr/bin/env python3
"""
Evaluate trained model on all instances and save makespan results to file.
"""
import json
import os
import random
import torch
from datetime import datetime

from evaluate_model import load_model, build_policy
from generator import load_instances_from_json
from new_ppo import JobShopEnv  # Use the 8-feature version from training
from muxGNN import MuxJobShopGNNPolicy

def evaluate_instance_8dim(policy, instance, device="cpu"):
    """Evaluate using JobShopEnv from new_ppo.py (8 features)."""
    policy.eval()
    
    env = JobShopEnv(instance)
    state = env.reset()
    done = False
    
    with torch.no_grad():
        while not done:
            node_features = state["node_features"].to(device)
            edge_index = state["edge_index"].to(device)
            mask = state["mask"].to(device)
            
            # Forward pass
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
            
            # Validate action
            if not mask[action].item():
                valid_actions = mask.nonzero(as_tuple=True)[0].tolist()
                if valid_actions:
                    action = random.choice(valid_actions)
                else:
                    break
            
            state, _, done, _ = env.step(action)
    
    return float(env.get_makespan())


# Load checkpoint
checkpoint_path = "checkpoints/muxgnn_full.pt"
print(f"Loading checkpoint: {checkpoint_path}")
payload = load_model(checkpoint_path, device="cpu")
metadata = payload.get("metadata", {})

# Build and load policy
model_type = metadata.get("model_type", "gnn")
node_feature_dim = int(metadata.get("node_feature_dim", 7))
hidden_dim = int(metadata.get("hidden_dim", 64))
num_heads = int(metadata.get("num_heads", 4))
num_layers = int(metadata.get("num_layers", 3))

print(f"Model: {model_type} (hidden={hidden_dim}, heads={num_heads}, layers={num_layers})")

policy = build_policy(model_type, node_feature_dim, hidden_dim, num_heads, num_layers, device="cpu")
policy.load_state_dict(payload["model_state_dict"])
policy.eval()

# Load all instances
instances_file = "instances.json"
print(f"Loading instances from: {instances_file}")
instances = load_instances_from_json(instances_file)
print(f"Total instances: {len(instances)}")

# Evaluate all instances
results = []
print("\nEvaluating instances...")
for i, instance in enumerate(instances):
    try:
        makespan = evaluate_instance_8dim(policy, instance, device="cpu")
        result = {
            "instance_id": i,
            "num_jobs": instance.num_jobs,
            "num_machines": instance.num_machines,
            "num_operations": instance.num_operations,
            "makespan": float(makespan)
        }
        results.append(result)
        print(f"  Instance {i+1}/{len(instances)}: makespan={makespan:.1f} " +
              f"(jobs={instance.num_jobs}, machines={instance.num_machines}, ops={instance.num_operations})")
    except Exception as e:
        print(f"  Instance {i+1}/{len(instances)}: FAILED - {e}")
        results.append({
            "instance_id": i,
            "num_jobs": instance.num_jobs,
            "num_machines": instance.num_machines,
            "num_operations": instance.num_operations,
            "makespan": None,
            "error": str(e)
        })

# Calculate statistics
valid_makespans = [r["makespan"] for r in results if r["makespan"] is not None]
stats = {
    "total_instances": len(instances),
    "successful_evaluations": len(valid_makespans),
    "failed_evaluations": len(instances) - len(valid_makespans),
    "mean_makespan": sum(valid_makespans) / len(valid_makespans) if valid_makespans else None,
    "min_makespan": min(valid_makespans) if valid_makespans else None,
    "max_makespan": max(valid_makespans) if valid_makespans else None,
}

# Save results to JSON
output_file = f"muxgnn_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
output_data = {
    "timestamp": datetime.now().isoformat(),
    "checkpoint_path": checkpoint_path,
    "model_metadata": metadata,
    "statistics": stats,
    "instance_results": results
}

with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
print(f"Total instances evaluated: {stats['total_instances']}")
print(f"Successful: {stats['successful_evaluations']}")
print(f"Failed: {stats['failed_evaluations']}")
if valid_makespans:
    print(f"Mean makespan: {stats['mean_makespan']:.3f}")
    print(f"Best makespan: {stats['min_makespan']:.3f}")
    print(f"Worst makespan: {stats['max_makespan']:.3f}")
print(f"\nResults saved to: {output_file}")
print("=" * 70)
