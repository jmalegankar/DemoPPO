"""
Inspect MetaWorld observation structure for pick-place-v3 and reach-v3.

Usage:
    python inspect_obs.py

Prints:
  - Raw obs values at t=0 for two different seeds
  - Element-wise diff between seeds (large diff = absolute position)
  - MetaWorld's own obs_dict breakdown if available
"""

import numpy as np
import gymnasium as gym
import metaworld
from metaworld.policies.sawyer_pick_place_v3_policy import SawyerPickPlaceV3Policy
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy


def inspect_task(task_name: str, policy, seeds: list[int] = [0, 1, 42]):
    print(f"\n{'='*60}")
    print(f"TASK: {task_name}")
    print(f"{'='*60}")

    env = gym.make("Meta-World/MT1", env_name=task_name, render_mode=None)

    obs_per_seed = {}
    info_per_seed = {}

    for seed in seeds:
        obs, info = env.reset(seed=seed)
        obs_per_seed[seed] = obs.copy()
        info_per_seed[seed] = info.copy()

    # ── Raw obs at t=0 ────────────────────────────────────────────
    print(f"\nObs dim: {obs.shape[0]}")
    print(f"\n{'idx':>4}  {'seed '+str(seeds[0]):>10}  {'seed '+str(seeds[1]):>10}  {'seed '+str(seeds[2]):>10}  {'max_diff':>10}")
    print("-" * 55)

    obs0 = obs_per_seed[seeds[0]]
    obs1 = obs_per_seed[seeds[1]]
    obs2 = obs_per_seed[seeds[2]]

    for i in range(len(obs0)):
        diff = max(abs(obs0[i] - obs1[i]), abs(obs0[i] - obs2[i]),
                   abs(obs1[i] - obs2[i]))
        marker = " ←" if diff > 0.01 else ""
        print(f"{i:>4}  {obs0[i]:>10.4f}  {obs1[i]:>10.4f}  {obs2[i]:>10.4f}  {diff:>10.4f}{marker}")

    # ── Info dict ─────────────────────────────────────────────────
    print(f"\nInfo dict keys: {list(info_per_seed[seeds[0]].keys())}")

    # ── Try to get obs_dict from unwrapped env ─────────────────────
    print("\nAttempting obs_dict from unwrapped env ...")
    try:
        unwrapped = env.unwrapped
        env.reset(seed=seeds[0])
        # step once to populate internal state
        a = policy.get_action(obs_per_seed[seeds[0]])
        env.step(a)
        if hasattr(unwrapped, '_get_obs_dict'):
            obs_dict = unwrapped._get_obs_dict()
            print("obs_dict keys and shapes:")
            for k, v in obs_dict.items():
                v = np.atleast_1d(v)
                print(f"  {k:30s}: {v.shape}  {v}")
        else:
            print("  No _get_obs_dict method found")
            # try common attribute names
            for attr in ['hand_pos', 'obj_pos', 'goal_pos', '_target_pos',
                         'tcp_center', 'gripper_distance_apart']:
                if hasattr(unwrapped, attr):
                    val = getattr(unwrapped, attr)
                    print(f"  env.{attr} = {val}")
    except Exception as e:
        print(f"  Could not access obs_dict: {e}")

    # ── Heuristic grouping by diff pattern ────────────────────────
    print("\nHeuristic grouping (indices with max_diff > 0.01 = seed-dependent):")
    stable_idx  = [i for i in range(len(obs0))
                   if max(abs(obs0[i]-obs1[i]), abs(obs0[i]-obs2[i])) <= 0.01]
    varying_idx = [i for i in range(len(obs0))
                   if max(abs(obs0[i]-obs1[i]), abs(obs0[i]-obs2[i])) > 0.01]
    print(f"  Stable  (shared across seeds): {stable_idx}")
    print(f"  Varying (seed-dependent):      {varying_idx}")

    env.close()


def main():
    inspect_task("pick-place-v3", SawyerPickPlaceV3Policy(), seeds=[0, 1, 42])
    inspect_task("reach-v3",      SawyerReachV3Policy(),     seeds=[0, 1, 42])


if __name__ == "__main__":
    main()