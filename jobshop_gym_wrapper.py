"""
jobshop_gym_wrapper.py
======================
Wraps JobShopEnv in a proper gymnasium.Env with **fixed-size** numpy
observation and action spaces, making it compatible with Stable-Baselines3.

Supports variable-size instances via global `max_ops` padding so that a
single trained model can handle any instance up to that size.

Observation space (gymnasium.spaces.Dict):
    "node_features"  : Box(shape=(max_ops * NODE_FEAT_DIM,), float32)
                       Flattened, zero-padded node feature matrix.
                       Real nodes occupy the first actual_num_ops * feat_dim
                       entries; the rest are zeros.
    "edge_index_flat": Box(shape=(2 * max_edges,), float32)
                       Padded edge index (src rows then dst rows).
    "num_edges"      : Box(shape=(1,), float32)
                       Actual number of edges used.
    "actual_num_ops" : Box(shape=(1,), float32)
                       Real operation count of the current instance.
                       The extractor uses this to slice padding off before
                       feeding into the GNN.
    "valid_mask"     : Box(shape=(max_ops,), float32)
                       1.0 for schedulable ops, 0.0 for invalid / padding.

Action space:
    Discrete(max_ops) — actions >= actual_num_ops are treated as invalid
    and auto-corrected to a random valid op inside step().

Multi-instance training:
    Pass a list of JobShopInstance objects as `instances`.  Each reset()
    samples a random instance from the list.
"""

from __future__ import annotations

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional, Tuple, Dict, Union

from jobshop import JobShopInstance
from jobshop_env import JobShopEnv


class JobShopGymEnv(gym.Env):
    """
    Gymnasium wrapper around JobShopEnv.

    Parameters
    ----------
    instances : JobShopInstance | list[JobShopInstance]
        One or more instances.  If a list is given, one is sampled at
        random on every reset() so the model trains across all of them.
    max_ops : int | None
        Pad all observations to this operation count.  Must be >= the
        largest instance's num_operations.  When None, uses the maximum
        across the provided instances.
    reward_mode : "sparse" | "dense"
    normalize_features : bool
    max_edges_multiplier : float
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        instances: Union[JobShopInstance, List[JobShopInstance]],
        max_ops: Optional[int] = 1500,
        reward_mode: str = "sparse",
        normalize_features: bool = True,
        max_edges_multiplier: float = 2.5,
    ):
        super().__init__()

        if isinstance(instances, JobShopInstance):
            instances = [instances]
        self._instances: List[JobShopInstance] = instances
        self._reward_mode = reward_mode
        self._normalize_features = normalize_features

        # Ensure max_ops is explicitly set to the value used during training
        self.max_ops: int = max_ops or 1500  # Default to 1500 operations
        # Ensure max_ops during testing does not exceed training max_ops
        if self.max_ops > max_ops:
            print(f"Warning: max_ops during testing ({self.max_ops}) exceeds training max_ops ({max_ops}). Padding observations.")
            self.max_ops = max_ops

        self.node_feat_dim: int = JobShopEnv.NODE_FEATURE_DIM  # 7
        # Worst-case edge upper bound based on max_ops
        self.max_edges: int = int(self.max_ops * self.max_ops * max_edges_multiplier)

        # ---- observation space (fixed size, padded) ----
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.max_ops * self.node_feat_dim,),
                dtype=np.float32,
            ),
            "edge_index_flat": spaces.Box(
                low=0, high=self.max_ops,
                shape=(2 * self.max_edges,),
                dtype=np.float32,
            ),
            "num_edges": spaces.Box(
                low=0, high=self.max_edges,
                shape=(1,), dtype=np.float32,
            ),
            "actual_num_ops": spaces.Box(
                low=1, high=self.max_ops,
                shape=(1,), dtype=np.float32,
            ),
            "valid_mask": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_ops,), dtype=np.float32,
            ),
        })

        # ---- action space (fixed to max_ops) ----
        self.action_space = spaces.Discrete(self.max_ops)

        # Runtime state (set on reset)
        self._env: Optional[JobShopEnv] = None
        self._current_instance: Optional[JobShopInstance] = None
        self._last_valid_mask: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)

        # Sample a new instance
        self._current_instance = random.choice(self._instances)
        self._env = JobShopEnv(
            self._current_instance,
            reward_mode=self._reward_mode,
            normalize_features=self._normalize_features,
        )
        obs_dict = self._env.reset()
        sb3_obs = self._convert_obs(obs_dict)
        self._last_valid_mask = sb3_obs["valid_mask"].astype(bool)
        return sb3_obs, {}

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        # Correct out-of-range or invalid action → random valid op
        actual_ops = self._env.num_ops
        if (action >= actual_ops
                or (self._last_valid_mask is not None
                    and not self._last_valid_mask[action])):
            valid_ids = np.where(
                self._last_valid_mask[:actual_ops]
                if self._last_valid_mask is not None
                else np.ones(actual_ops, dtype=bool)
            )[0]
            action = int(np.random.choice(valid_ids))

        obs_dict, reward, done, info = self._env.step(int(action))
        sb3_obs = self._convert_obs(obs_dict)
        self._last_valid_mask = sb3_obs["valid_mask"].astype(bool)
        return sb3_obs, float(reward), done, False, info

    def render(self):
        pass

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _convert_obs(self, obs_dict: Dict) -> Dict[str, np.ndarray]:
        """Convert torch-tensor obs from JobShopEnv to padded numpy arrays."""
        actual_ops = self._env.num_ops

        # Ensure actual_ops does not exceed max_ops
        if actual_ops > self.max_ops:
            raise ValueError(
                f"actual_ops ({actual_ops}) exceeds max_ops ({self.max_ops}). "
                "Ensure max_ops is consistent between training and testing."
            )

        # node features: [actual_ops, feat_dim] → pad to [max_ops * feat_dim]
        nf = obs_dict["node_features"].numpy()  # [actual_ops, feat_dim]
        nf_pad = np.zeros(self.max_ops * self.node_feat_dim, dtype=np.float32)
        nf_flat = nf.flatten()
        if nf_flat.size > nf_pad.size:
            raise ValueError(
                f"Flattened node features size ({nf_flat.size}) exceeds padded size ({nf_pad.size}). "
                "Check max_ops and node_feat_dim consistency."
            )
        nf_pad[:nf_flat.size] = nf_flat

        # edge_index → pad to 2 * max_edges
        ei = obs_dict["edge_index"]  # [2, num_edges]
        num_edges = ei.shape[1]
        src = ei[0].numpy().astype(np.float32)
        dst = ei[1].numpy().astype(np.float32)
        if num_edges > self.max_edges:
            num_edges = self.max_edges
            src = src[:num_edges]
            dst = dst[:num_edges]
        src_pad = np.zeros(self.max_edges, dtype=np.float32)
        dst_pad = np.zeros(self.max_edges, dtype=np.float32)
        src_pad[:num_edges] = src
        dst_pad[:num_edges] = dst
        ei_flat = np.concatenate([src_pad, dst_pad])

        # valid mask: pad beyond actual_ops with 0 (invalid)
        vm_full = np.zeros(self.max_ops, dtype=np.float32)
        vm = obs_dict["valid_mask"].numpy().astype(np.float32)  # [actual_ops]
        vm_full[:actual_ops] = vm

        return {
            "node_features":   nf_pad,
            "edge_index_flat": ei_flat,
            "num_edges":       np.array([num_edges], dtype=np.float32),
            "actual_num_ops":  np.array([actual_ops], dtype=np.float32),
            "valid_mask":      vm_full,
        }

    def action_masks(self) -> np.ndarray:
        """For sb3-contrib MaskablePPO: return boolean mask of valid actions."""
        if self._last_valid_mask is not None:
            return self._last_valid_mask.copy()
        return np.ones(self.max_ops, dtype=bool)
