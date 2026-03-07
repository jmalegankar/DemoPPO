"""
Collect expert demonstrations from MetaWorld and save as .npz files.

Usage:
    # standard collection
    python collect_demos.py --task reach-v3      --n-seeds 50 --eps-per-seed 4  --out data/demos
    python collect_demos.py --task pick-place-v3 --n-seeds 20 --eps-per-seed 10 --out data/demos

    # held-out visualization set (offset seeds so they never overlap training)
    python collect_demos.py --task reach-v3      --n-seeds 10 --eps-per-seed 2  --seed-offset 1000 --out data/demos/holdout
    python collect_demos.py --task pick-place-v3 --n-seeds 10 --eps-per-seed 2  --seed-offset 1000 --out data/demos/holdout

Output structure:
    data/demos/
        reach-v3_demos.npz
        pick-place-v3_demos.npz
    data/demos/holdout/
        reach-v3_demos.npz
        pick-place-v3_demos.npz

Each .npz contains:
    obs_t    : (N, obs_dim)   float32
    action   : (N, act_dim)   float32
    obs_next : (N, obs_dim)   float32
    ep_ids   : (N,)           int32     global episode index
    seed_ids : (N,)           int32     which seed each transition came from
"""

import argparse
import numpy as np
import gymnasium as gym
from pathlib import Path
from helpers.obs_transform import transform_obs_numpy
import metaworld

EXPERT_POLICIES = {
    "reach-v3": "metaworld.policies.sawyer_reach_v3_policy.SawyerReachV3Policy",
    "pick-place-v3": "metaworld.policies.sawyer_pick_place_v3_policy.SawyerPickPlaceV3Policy",
}

MAX_STEPS = 500   # MetaWorld default episode length
MAX_ATTEMPTS_PER_SEED = 20   # give up on a seed if it keeps failing


def load_policy(task_name: str):
    import importlib
    if task_name not in EXPERT_POLICIES:
        raise ValueError(
            f"No expert policy registered for '{task_name}'. "
            f"Available: {list(EXPERT_POLICIES.keys())}"
        )
    module_path, class_name = EXPERT_POLICIES[task_name].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def collect_seed(
    env,
    policy,
    seed: int,
    n_episodes: int,
) -> dict | None:
    """
    Collect n_episodes successful episodes for a single seed.
    Returns dict of transitions or None if seed consistently fails.

    A single seed fixes the task configuration (object + goal placement).
    Multiple episodes per seed = same layout, varied arm dynamics/noise.
    """
    obs_t_list    = []
    action_list   = []
    obs_next_list = []

    collected = 0
    attempts  = 0

    while collected < n_episodes:
        attempts += 1
        if attempts > MAX_ATTEMPTS_PER_SEED:
            print(f"    seed {seed}: gave up after {attempts} attempts ({collected} successes)")
            return None

        obs, _ = env.reset(seed=seed)
        ep_obs_t, ep_actions, ep_obs_next = [], [], []
        success = False

        for _ in range(MAX_STEPS):
            action = policy.get_action(obs)
            obs_next, _, terminated, truncated, info = env.step(action)

            ep_obs_t.append(np.array(obs).astype(np.float32))
            ep_actions.append(action.copy().astype(np.float32))
            ep_obs_next.append(np.array(obs_next).astype(np.float32))

            obs = obs_next

            if int(info.get("success", 0)) == 1:
                success = True
                break
            if terminated or truncated:
                break

        if success:
            obs_t_list.append(np.stack(ep_obs_t))
            action_list.append(np.stack(ep_actions))
            obs_next_list.append(np.stack(ep_obs_next))
            collected += 1

    return {
        "obs_t":    np.concatenate(obs_t_list,    axis=0),
        "action":   np.concatenate(action_list,   axis=0),
        "obs_next": np.concatenate(obs_next_list, axis=0),
    }


def collect(
    task_name:    str,
    n_seeds:      int,
    eps_per_seed: int,
    seed_offset:  int,
) -> dict:
    env    = gym.make("Meta-World/MT1", env_name=task_name, render_mode=None)
    policy = load_policy(task_name)

    all_obs_t    = []
    all_actions  = []
    all_obs_next = []
    all_ep_ids   = []
    all_seed_ids = []

    global_ep = 0
    seeds_done = 0

    print(
        f"Collecting {n_seeds} seeds × {eps_per_seed} episodes "
        f"for '{task_name}' (seed offset={seed_offset}) ..."
    )

    for s in range(n_seeds):
        seed = seed_offset + s
        result = collect_seed(env, policy, seed, eps_per_seed)

        if result is None:
            print(f"  [{s+1}/{n_seeds}] seed {seed} — skipped (too many failures)")
            continue

        n_trans = len(result["obs_t"])

        # ep_ids: each episode within this seed gets a unique global id
        # seed produced eps_per_seed episodes; approximate per-ep boundaries
        # (exact boundaries not needed for SC-VAE training, only for analysis)
        ep_ids_local = np.repeat(
            np.arange(global_ep, global_ep + eps_per_seed, dtype=np.int32),
            n_trans // eps_per_seed + 1,
        )[:n_trans]

        all_obs_t.append(result["obs_t"])
        all_actions.append(result["action"])
        all_obs_next.append(result["obs_next"])
        all_ep_ids.append(ep_ids_local)
        all_seed_ids.append(np.full(n_trans, seed, dtype=np.int32))

        global_ep  += eps_per_seed
        seeds_done += 1
        print(f"  [{seeds_done}/{n_seeds}] seed {seed} — {n_trans} transitions")

    env.close()

    return {
        "obs_t":    np.concatenate(all_obs_t,    axis=0),
        "action":   np.concatenate(all_actions,  axis=0),
        "obs_next": np.concatenate(all_obs_next, axis=0),
        "ep_ids":   np.concatenate(all_ep_ids,   axis=0),
        "seed_ids": np.concatenate(all_seed_ids, axis=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",          default="reach-v3")
    ap.add_argument("--n-seeds",       type=int, default=50)
    ap.add_argument("--eps-per-seed",  type=int, default=4)
    ap.add_argument("--seed-offset",   type=int, default=0,
                    help="Add this to all seeds — use 1000 for held-out sets")
    ap.add_argument("--out",           default="data/demos")
    args = ap.parse_args()

    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task}_demos.npz"

    data = collect(
        args.task,
        args.n_seeds,
        args.eps_per_seed,
        args.seed_offset,
    )

    np.savez_compressed(out_path, **data)

    n = len(data["obs_t"])
    unique_seeds = len(np.unique(data["seed_ids"]))
    unique_eps   = len(np.unique(data["ep_ids"]))
    print(f"\nSaved → {out_path}")
    print(f"  transitions : {n}")
    print(f"  seeds       : {unique_seeds}")
    print(f"  episodes    : {unique_eps}")
    print(f"  obs_dim     : {data['obs_t'].shape[1]}")
    print(f"  act_dim     : {data['action'].shape[1]}")


if __name__ == "__main__":
    main()