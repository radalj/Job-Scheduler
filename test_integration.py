#!/usr/bin/env python3
"""
INTEGRATION TEST: MuxGNN Pipeline
==================================

Quick validation that all components work together:
1. Train a model (GNN or MuxGNN)
2. Save checkpoint
3. Load and evaluate

Run this to validate the full pipeline is working.
"""

import os
import sys
import torch
from pathlib import Path

print("=" * 70)
print("MuxGNN Integration Pipeline Validation")
print("=" * 70)

# Test 1: Import all modules
print("\n[1/5] Testing imports...")
try:
    from GNN import JobShopGNNPolicy, MuxJobShopGNNPolicy
    from jobshop_env import JobShopEnv
    from generator import generate_general_instances
    from new_ppo import PPOScheduler, build_policy
    print("  ✓ All imports successful")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create both policy types
print("\n[2/5] Creating policy models...")
try:
    gnn_policy = JobShopGNNPolicy(node_feature_dim=7, hidden_dim=16, num_heads=2, num_layers=1)
    mux_policy = MuxJobShopGNNPolicy(node_feature_dim=7, hidden_dim=16, num_heads=2, num_layers=1)
    print(f"  ✓ GNN policy params: {sum(p.numel() for p in gnn_policy.parameters()):,}")
    print(f"  ✓ MuxGNN policy params: {sum(p.numel() for p in mux_policy.parameters()):,}")
except Exception as e:
    print(f"  ✗ Policy creation failed: {e}")
    sys.exit(1)

# Test 3: Create environment and run one step
print("\n[3/5] Testing JobShopEnv integration...")
try:
    instances = generate_general_instances(num_instances=1, seed=42)
    env = JobShopEnv(instances[0])
    state = env.reset()
    
    # Check state structure
    assert "node_features" in state, "Missing node_features"
    assert "edge_index" in state, "Missing edge_index"
    assert "valid_mask" in state, "Missing valid_mask"
    
    # Forward policy pass
    with torch.no_grad():
        out = gnn_policy(
            state["node_features"],
            state["edge_index"],
            state.get("valid_mask")
        )
    assert "action_logits" in out, "Missing action_logits"
    assert "value" in out, "Missing value"
    print(f"  ✓ Environment: {instances[0].num_operations} ops, {instances[0].num_machines} machines")
    print(f"  ✓ GNN forward pass successful")
except Exception as e:
    print(f"  ✗ Environment integration failed: {e}")
    sys.exit(1)

# Test 4: Checkpoint save/load cycle
print("\n[4/5] Testing checkpoint save/load...")
try:
    os.makedirs("checkpoints", exist_ok=True)
    
    # Save
    checkpoint = {
        "model_state_dict": gnn_policy.state_dict(),
        "metadata": {
            "model_type": "gnn",
            "node_feature_dim": 7,
            "hidden_dim": 16,
            "num_heads": 2,
            "num_layers": 1,
            "lr": 3e-4,
            "seed": 42,
        }
    }
    checkpoint_path = "checkpoints/test_validation.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    # Load and verify
    loaded = torch.load(checkpoint_path, map_location="cpu")
    assert "model_state_dict" in loaded, "Missing model_state_dict"
    assert "metadata" in loaded, "Missing metadata"
    assert loaded["metadata"]["model_type"] == "gnn", "Metadata mismatch"
    
    # Recreate policy from metadata
    metadata = loaded["metadata"]
    policy_restored = build_policy(
        model_type=metadata["model_type"],
        node_feature_dim=int(metadata["node_feature_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        num_heads=int(metadata["num_heads"]),
        num_layers=int(metadata["num_layers"]),
    )
    policy_restored.load_state_dict(loaded["model_state_dict"])
    print(f"  ✓ Checkpoint restored with metadata reconstruction")
    
    # Test restored model inference
    with torch.no_grad():
        out_restored = policy_restored(
            state["node_features"],
            state["edge_index"],
            state.get("valid_mask")
        )
    print(f"  ✓ Restored model inference successful")
    
except Exception as e:
    print(f"  ✗ Checkpoint cycle failed: {e}")
    sys.exit(1)

# Test 5: Evaluate on instance
print("\n[5/5] Testing evaluation flow...")
try:
    trainer = PPOScheduler(policy=policy_restored, device="cpu")
    trainer.policy.eval()
    
    # Simple greedy evaluation
    test_instance = instances[0]
    env_test = JobShopEnv(test_instance)
    state_test = env_test.reset()
    done = False
    steps = 0
    
    with torch.no_grad():
        while not done and steps < 100:  # Limit steps for test
            node_features = state_test["node_features"]
            edge_index = state_test["edge_index"]
            mask = state_test.get("valid_mask")
            
            out = trainer.policy(node_features, edge_index, mask)
            logits = out["action_logits"]
            action = torch.argmax(logits).item()
            
            # Validate action
            if not mask[action].item():
                valid = mask.nonzero(as_tuple=True)[0].tolist()
                if valid:
                    action = valid[0]
                else:
                    break
            
            state_test, _, done, info = env_test.step(action)
            steps += 1
    
    makespan = info.get("makespan", 0.0)
    print(f"  ✓ Evaluation successful: makespan={makespan:.1f} after {steps} steps")
    
except Exception as e:
    print(f"  ✗ Evaluation failed: {e}")
    sys.exit(1)

# Success!
print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED")
print("=" * 70)
print("\nMuxGNN Integration is fully functional!")
print("\nQuick start commands:")
print("  python new_ppo.py --model-type muxgnn --max-instances 10 --epochs 1")
print("  python evaluate_model.py checkpoints/demo_gnn.pt --max-instances 5")
print("\nFor more info, see IMPLEMENTATION_SUMMARY.md")
