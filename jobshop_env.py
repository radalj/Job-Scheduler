"""
JobShopEnv
==========
A gym-style environment for the Job Shop Scheduling Problem that produces
GNN-ready observations at every step.

Observation (dict, all `torch.Tensor`):
    node_features : [num_ops, 7]
        [duration, machine_id, job_id, op_position_in_job,
         is_scheduled, earliest_start, remaining_ops_in_job]
    edge_index    : [2, num_edges]  – directed edges (conjunctive + disjunctive)
    valid_mask    : [num_ops]       – 1 for schedulable operations, 0 otherwise

Actions:
    Integer index into [0, num_ops) that selects which operation to schedule next.
    Only operations whose `valid_mask[action] == 1` are legal.

Reward:
    Sparse: −normalized_makespan at the terminal step; 0 elsewhere.
    Dense  : −1 per step + progress rewards based on machine utilization.
    Shaped : Comprehensive reward shaping with progress, balance, and completion bonuses.
    Makespan²: Direct −makespan² loss for optimal RL learning!

Usage:
    env = JobShopEnv(instance, reward_mode="sparse")
    obs = env.reset()
    done = False
    while not done:
        mask   = obs["valid_mask"]
        action = policy(obs)            # pick a valid op index
        obs, reward, done, info = env.step(action)
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from jobshop import JobShopInstance
from graph_generator import generate_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tensors(
    instance: JobShopInstance,
    machine_earliest: List[int],
    job_earliest: List[int],
    scheduled: List[bool],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build node-feature matrix and edge_index from current scheduling state.

    Node features (per operation):
        0  duration
        1  machine_id
        2  job_id
        3  position_in_job  (op index within its job)
        4  is_scheduled     (0/1)
        5  earliest_start   (max of machine-free time and job-free time)
        6  remaining_ops    (# unscheduled ops left in the same job after this one)
    """
    ops: List = []  # flat list of Operation objects in canonical order
    for job in instance.jobs:
        for op in job:
            ops.append(op)

    num_ops = len(ops)

    # Precompute remaining-ops-in-job counts
    job_remaining: Dict[int, int] = {}
    for job_id, job in enumerate(instance.jobs):
        unscheduled = sum(1 for op in job if not scheduled[op.operation_id])
        job_remaining[job_id] = unscheduled

    rows = []
    for op in ops:
        jid = op.job_id
        mid = op.machine_id
        oid = op.operation_id
        pos = op.position_in_job

        es = max(machine_earliest[mid], job_earliest[jid])
        is_done = 1.0 if scheduled[oid] else 0.0
        # remaining after this op in job (not counting itself)
        rem = job_remaining[jid] - (1 if not scheduled[oid] else 0)
        rem = max(rem, 0)

        rows.append([
            float(op.duration),
            float(mid),
            float(jid),
            float(pos),
            is_done,
            float(es),
            float(rem),
        ])

    node_features = torch.tensor(rows, dtype=torch.float32)  # [num_ops, 7]
    return node_features


def _build_edge_index(
    instance: JobShopInstance,
    scheduled: List[bool],
    schedule_order: List[int],   # op ids in the order they were scheduled
) -> torch.Tensor:
    """
    Build directed edge_index on the disjunctive graph.

    Edges included:
      Conjunctive (precedence): op_i -> op_{i+1} within each job (always present).
      Disjunctive (machine):    among unscheduled ops on the same machine,
                                all-to-all edges represent contention.
                                For already-scheduled ops, only the decided
                                ordering chain is added.
    """
    src_list: List[int] = []
    dst_list: List[int] = []

    # --- conjunctive edges (job precedence) ---
    for job in instance.jobs:
        for k in range(len(job) - 1):
            s = job[k].operation_id
            d = job[k + 1].operation_id
            src_list.append(s)
            dst_list.append(d)

    # --- disjunctive edges (machine conflicts) ---
    # Group ops by machine
    machine_ops: Dict[int, List[int]] = {}
    for job in instance.jobs:
        for op in job:
            machine_ops.setdefault(op.machine_id, []).append(op.operation_id)

    for mid, op_ids in machine_ops.items():
        if len(op_ids) < 2:
            continue

        # Ops scheduled on this machine in order → form a chain
        chain = [oid for oid in schedule_order if oid in op_ids]
        for k in range(len(chain) - 1):
            s, d = chain[k], chain[k + 1]
            src_list.append(s)
            dst_list.append(d)

        # Unscheduled ops on this machine → all-to-all (undirected = both dirs)
        unscheduled_on_m = [oid for oid in op_ids if not scheduled[oid]]
        for i, oid_i in enumerate(unscheduled_on_m):
            for oid_j in unscheduled_on_m[i + 1:]:
                src_list.append(oid_i)
                dst_list.append(oid_j)
                src_list.append(oid_j)
                dst_list.append(oid_i)

    if len(src_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor([src_list, dst_list], dtype=torch.long)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class JobShopEnv:
    """
    Gym-style Job Shop environment whose observations are directly consumable
    by `JobShopGNNPolicy` from GNN.py.

    Parameters
    ----------
    instance : JobShopInstance
    reward_mode : "sparse" | "dense" | "shaped" | "makespan_squared"
        sparse → −normalized_makespan only at the last step.
        dense  → progress rewards and step penalties during episode + final reward.
        shaped → comprehensive reward shaping with balance, progress, and completion bonuses.
        makespan_squared → direct minimization of makespan² (best for RL learning!)
    normalize_features : bool
        If True, scale duration / earliest_start by the max duration in the
        instance so all features live roughly in [0, 1].
    """

    # Feature dimension exposed to the policy
    NODE_FEATURE_DIM: int = 7

    def __init__(
        self,
        instance: JobShopInstance,
        reward_mode: str = "sparse",
        normalize_features: bool = True,
    ):
        assert reward_mode in ("sparse", "dense", "shaped", "makespan_squared"), \
            "reward_mode must be 'sparse', 'dense', 'shaped', or 'makespan_squared'"

        self.instance = instance
        self.reward_mode = reward_mode
        self.normalize_features = normalize_features

        self.num_ops: int = instance.num_operations
        self.num_jobs: int = instance.num_jobs
        self.num_machines: int = instance.num_machines

        # Flat list of ops in canonical order (job0-op0, job0-op1, …)
        self._ops = [op for job in instance.jobs for op in job]

        # Normalisation constant (max duration)
        self._max_duration: float = max(op.duration for op in self._ops) or 1.0

        # State will be initialised in reset()
        self._scheduled: List[bool] = []
        self._machine_earliest: List[int] = []
        self._job_earliest: List[int] = []
        self._schedule_order: List[int] = []   # op ids in scheduling order
        self._start_times: Dict[int, int] = {}
        self._steps: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset to initial (empty) schedule and return first observation."""
        self._scheduled = [False] * self.num_ops
        self._machine_earliest = [0] * self.num_machines
        self._job_earliest = [0] * self.num_jobs
        self._schedule_order = []
        self._start_times = {}
        self._steps = 0
        # Initialize tracking for dense reward mode
        self._prev_max_machine_time = 0
        return self._get_obs()

    def step(
        self, action: int
    ) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict]:
        """
        Schedule the operation identified by `action` (integer op_id index).

        Returns
        -------
        obs    : dict with node_features, edge_index, valid_mask
        reward : float
        done   : bool
        info   : dict  (contains 'makespan' when done)
        """
        # --- validate ---
        valid = self._valid_mask()
        if not valid[action]:
            raise ValueError(
                f"Action {action} is not schedulable. "
                f"Valid ops: {valid.nonzero(as_tuple=True)[0].tolist()}"
            )

        # --- schedule op ---
        op = self._ops[action]
        jid = op.job_id
        mid = op.machine_id

        start = max(self._machine_earliest[mid], self._job_earliest[jid])
        finish = start + op.duration

        self._scheduled[action] = True
        self._machine_earliest[mid] = finish
        self._job_earliest[jid] = finish
        self._schedule_order.append(action)
        self._start_times[action] = start
        self._steps += 1

        # --- termination ---
        done = all(self._scheduled)

        # --- reward ---
        if done:
            makespan = max(self._machine_earliest)
            
            if self.reward_mode == "sparse":
                # Sparse: simple negative makespan (agent learns to minimize)
                reward = -makespan / 100.0  # Scale down but keep meaningful magnitude
            elif self.reward_mode == "dense":
                # Dense: final reward + step penalty
                reward = -makespan / 100.0 - self._steps * 0.01  # Penalize long episodes
            elif self.reward_mode == "makespan_squared":
                # Direct makespan² loss - exactly what you requested!
                reward = -(makespan ** 2) / 10000.0  # Scale down to prevent huge numbers
            else:  # shaped mode
                # Shaped: final reward with improvement bonus
                total_processing_time = sum(op.duration for job in self.instance.jobs for op in job)
                efficiency = total_processing_time / makespan  # Closer to 1 is better
                reward = efficiency * 10 - makespan / 100.0  # Scale efficiency reward
            
            info = {"makespan": makespan, "steps": self._steps}
        else:
            if self.reward_mode == "dense":
                # Small step penalty to encourage efficiency
                reward = -0.01
            else:
                reward = 0.0
            info = {}

        obs = self._get_obs()
        return obs, reward, done, info

    def valid_actions(self) -> List[int]:
        """Return list of currently schedulable operation indices."""
        return self._valid_mask().nonzero(as_tuple=True)[0].tolist()

    def action_space_size(self) -> int:
        return self.num_ops

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict[str, torch.Tensor]:
        node_features = _build_tensors(
            self.instance,
            self._machine_earliest,
            self._job_earliest,
            self._scheduled,
        )
        edge_index = _build_edge_index(
            self.instance,
            self._scheduled,
            self._schedule_order,
        )
        valid_mask = self._valid_mask()

        if self.normalize_features:
            node_features = self._normalize(node_features)

        return {
            "node_features": node_features,   # [num_ops, 7]
            "edge_index": edge_index,          # [2, num_edges]
            "valid_mask": valid_mask,          # [num_ops]  bool
        }

    def _valid_mask(self) -> torch.Tensor:
        """
        An operation is schedulable iff:
          - it is not yet scheduled, AND
          - all preceding operations in its job are scheduled.
        """
        mask = torch.zeros(self.num_ops, dtype=torch.bool)
        for job in self.instance.jobs:
            for pos, op in enumerate(job):
                oid = op.operation_id
                if self._scheduled[oid]:
                    continue
                # All predecessors in job must be done
                if pos == 0 or self._scheduled[job[pos - 1].operation_id]:
                    mask[oid] = True
                break  # only first unscheduled op in each job can be scheduled
        return mask

    def _normalize(self, nf: torch.Tensor) -> torch.Tensor:
        """Scale duration (col 0) and earliest_start (col 5) by max_duration."""
        nf = nf.clone()
        # duration
        nf[:, 0] /= self._max_duration
        # machine_id  → normalise by num_machines
        nf[:, 1] /= max(self.num_machines - 1, 1)
        # job_id      → normalise by num_jobs
        nf[:, 2] /= max(self.num_jobs - 1, 1)
        # position    → normalise by max ops per job
        max_ops_per_job = max(len(job) for job in self.instance.jobs)
        nf[:, 3] /= max(max_ops_per_job - 1, 1)
        # earliest_start
        nf[:, 5] /= self._max_duration
        # remaining ops
        nf[:, 6] /= max(max_ops_per_job - 1, 1)
        return nf

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"JobShopEnv("
            f"jobs={self.num_jobs}, "
            f"machines={self.num_machines}, "
            f"ops={self.num_ops}, "
            f"reward={self.reward_mode})"
        )


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from generator import generate_general_instances
    from GNN import JobShopGNNPolicy
    import torch.nn.functional as F

    print("=" * 60)
    print("JobShopEnv  ×  GNN  integration demo")
    print("=" * 60)

    # ---- create a small instance ----
    instances = generate_general_instances(num_instances=1,
                                           num_jobs_range=(3, 5),
                                           num_machines_range=(3, 5),
                                           num_op_range=(2, 4),
                                           seed=42)
    instance = instances[0]
    print(f"\nInstance : {instance}")

    # ---- build env ----
    env = JobShopEnv(instance, reward_mode="sparse", normalize_features=True)
    print(f"Env      : {env}")
    print(f"NODE_FEATURE_DIM : {env.NODE_FEATURE_DIM}")

    # ---- build policy ----
    policy = JobShopGNNPolicy(
        node_feature_dim=env.NODE_FEATURE_DIM,
        hidden_dim=32,
        num_heads=2,
        num_layers=2,
    )
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # ---- rollout ----
    obs = env.reset()
    done = False
    total_reward = 0.0
    step = 0

    while not done:
        nf    = obs["node_features"]   # [num_ops, 7]
        ei    = obs["edge_index"]      # [2, E]
        mask  = obs["valid_mask"]      # [num_ops] bool

        with torch.no_grad():
            action, log_prob, entropy, value = policy.get_action_and_value(
                nf, ei, mask=mask
            )

        # force valid action if policy picked an invalid one
        if not mask[action]:
            valid_ids = mask.nonzero(as_tuple=True)[0]
            action = valid_ids[torch.randint(len(valid_ids), (1,)).item()]

        obs, reward, done, info = env.step(action.item())
        total_reward += reward
        step += 1

        print(f"  step {step:3d} | action={action.item():3d} | "
              f"reward={reward:8.1f} | value={value.item():7.4f}")

    print(f"\nDone in {step} steps | total_reward={total_reward:.1f} | "
          f"makespan={info['makespan']}")
    print("=" * 60)
