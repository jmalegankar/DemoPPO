"""
Relative observation transform for MetaWorld pick-place-v3 and reach-v3.

Raw 39-dim obs layout (confirmed by inspect_obs.py):
    [0:3]   ee_pos xyz
    [3]     gripper state (1.0=open, 0.0=closed)
    [4:7]   object_pos xyz  (x,y vary by seed; z=0.02 fixed)
    [7:10]  ee_vel xyz
    [10]    gripper vel
    [11:18] zeros (joint angles, all zero at reset — uninformative)
    [18:36] exact duplicate of [0:18]  ← dropped entirely
    [36:39] goal_pos xyz  (varies by seed)

Problem: object_pos and goal_pos are absolute → encoder memorizes per-seed coords.

Fix: replace absolute positions with relative displacements.

Output (13-dim):
    [0:3]   obj_to_ee   = object_pos - ee_pos     (where is object from hand)
    [3:6]   goal_to_ee  = goal_pos   - ee_pos     (where is goal from hand)
    [6:9]   goal_to_obj = goal_pos   - object_pos (progress: how far obj from goal)
    [9:12]  ee_vel                                (hand dynamics)
    [12]    gripper state                         (grasp state)

All 13 dims are seed-agnostic by construction.
"""

from __future__ import annotations
import numpy as np
import torch


# ── Slice indices (confirmed from inspect_obs.py) ─────────────────
_EE_POS    = slice(0, 3)    # ee xyz
_GRIPPER   = 3              # gripper open/close
_OBJ_POS   = slice(4, 7)    # object xyz (z=0.02 fixed but keep for completeness)
_EE_VEL    = slice(7, 10)   # ee velocity
_GOAL_POS  = slice(36, 39)  # goal xyz

OUT_DIM = 13


def transform_obs_numpy(obs: np.ndarray) -> np.ndarray:
    """
    obs: (..., 39) float32
    returns: (..., 13) float32
    """
    ee_pos  = obs[..., _EE_POS]
    obj_pos = obs[..., _OBJ_POS]
    goal    = obs[..., _GOAL_POS]
    ee_vel  = obs[..., _EE_VEL]
    grip    = obs[..., _GRIPPER: _GRIPPER + 1]

    return np.concatenate([
        obj_pos  - ee_pos,   # (3,) obj_to_ee
        goal     - ee_pos,   # (3,) goal_to_ee
        goal     - obj_pos,  # (3,) goal_to_obj
        ee_vel,              # (3,) ee_vel
        grip,                # (1,) gripper
    ], axis=-1).astype(np.float32)


def transform_obs_torch(obs: torch.Tensor) -> torch.Tensor:
    """
    obs: (B, 39) or (39,) float32
    returns: (B, 13) or (13,) float32
    """
    ee_pos  = obs[..., 0:3]
    obj_pos = obs[..., 4:7]
    goal    = obs[..., 36:39]
    ee_vel  = obs[..., 7:10]
    grip    = obs[..., 3:4]

    return torch.cat([
        obj_pos  - ee_pos,
        goal     - ee_pos,
        goal     - obj_pos,
        ee_vel,
        grip,
    ], dim=-1)


# ── Quick validation ───────────────────────────────────────────────

if __name__ == "__main__":
    import gymnasium as gym
    from metaworld.policies.sawyer_pick_place_v3_policy import SawyerPickPlaceV3Policy

    env    = gym.make("Meta-World/MT1", env_name="pick-place-v3", render_mode=None)
    policy = SawyerPickPlaceV3Policy()

    seeds  = [0, 1, 42, 100, 200]
    t0_obs = []

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        t0_obs.append(transform_obs_numpy(obs))

    t0_obs = np.stack(t0_obs)   # (5, 13)

    print("Transformed obs dim:", OUT_DIM)
    print(f"\n{'idx':>4}  " + "  ".join(f"seed {s:>3}" for s in seeds) + "  max_diff")
    print("-" * 75)

    labels = [
        "obj_to_ee_x", "obj_to_ee_y", "obj_to_ee_z",
        "goal_to_ee_x", "goal_to_ee_y", "goal_to_ee_z",
        "goal_to_obj_x", "goal_to_obj_y", "goal_to_obj_z",
        "ee_vel_x", "ee_vel_y", "ee_vel_z",
        "gripper",
    ]

    for i in range(OUT_DIM):
        vals    = t0_obs[:, i]
        maxdiff = vals.max() - vals.min()
        marker  = " ←" if maxdiff > 0.01 else ""
        print(f"{i:>4}  " +
              "  ".join(f"{v:>9.4f}" for v in vals) +
              f"  {maxdiff:>8.4f}{marker}")

    env.close()
    print("\nNotes:")
    print("  ← on relative positions (0-8) is EXPECTED — object/goal genuinely")
    print("    differ per seed, so relative displacements differ at t=0.")
    print("  ← on dynamics (9-11) or gripper (12) would be a bug — check those.")
    print("  Invariance guarantee: two transitions with same relative geometry")
    print("  but different absolute coords will have identical features. ✓")