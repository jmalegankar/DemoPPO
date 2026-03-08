"""
Evaluate a saved DemoPPO model on MetaWorld with rendering.

Usage:
    # render with GUI (human mode)
    python eval_meta_world.py --model runs/demo_ppo/pick-place-v3/final_model.zip --task pick-place-v3

    # save video frames instead of live render
    python eval_meta_world.py --model runs/demo_ppo/pick-place-v3/final_model.zip --task pick-place-v3 --record out_video/

    # multiple episodes, specific seeds
    python eval_meta_world.py --model runs/demo_ppo/reach-v3/final_model.zip --task reach-v3 --n-episodes 20 --seed 42
"""

import argparse
import time
from pathlib import Path

import numpy as np
import gymnasium as gym

from demo_ppo.demo_ppo import DemoPPO
from models.config import DemoPPOConfig
from models.sc_vae import TransitionSCVAE
from models.embeddings import MLPObsEmbedding

import metaworld


class MetaWorldEvalEnv(gym.Env):
    """MetaWorld wrapper with optional rendering."""

    OBS_DIM = 39
    ACT_DIM = 4

    def __init__(self, task_name: str, render_mode: str = "human"):
        self._env = gym.make(
            "Meta-World/MT1",
            env_name=task_name,
            render_mode=render_mode,
        )
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        reward = float(info.get("success", 0))
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


def evaluate(
    model_path: str,
    task_name: str,
    n_episodes: int = 10,
    seed: int = 0,
    render_mode: str = "human",
    record_dir: str | None = None,
    max_steps: int = 500,
    fps: float = 30.0,
    device: str = "cpu",
):
    # ── Load model ────────────────────────────────────────────────
    cfg = DemoPPOConfig()
    embedding = MLPObsEmbedding(
        obs_dim=cfg.obs_dim,
        hidden_dim=cfg.mlp_hidden_dim,
        out_dim=cfg.embed_out_dim,
        n_layers=cfg.mlp_n_layers,
    )
    policy_kwargs = {
        "features_extractor_kwargs": {"mu_dim": cfg.latent_dim},
        "net_arch": [256, 256],
        "vae_features_extractor_class": TransitionSCVAE,
        "vae_features_extractor_kwargs": {"embedding": embedding, "cfg": cfg},
    }
    model = DemoPPO.load(
        model_path,
        device=device,
        custom_objects={"policy_kwargs": policy_kwargs},
    )
    policy = model.policy
    policy.set_training_mode(False)

    null_action = np.zeros(MetaWorldEvalEnv.ACT_DIM, dtype=np.float32)

    # ── Optionally record frames ──────────────────────────────────
    if record_dir:
        render_mode = "rgb_array"
        record_path = Path(record_dir)
        record_path.mkdir(parents=True, exist_ok=True)

    # ── Environment ───────────────────────────────────────────────
    env = MetaWorldEvalEnv(task_name, render_mode=render_mode)

    # ── Rollout ───────────────────────────────────────────────────
    successes = []
    episode_lengths = []
    episode_rewards = []
    frame_idx = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        prev_obs = obs.copy()
        prev_action = null_action.copy()

        done = False
        ep_reward = 0.0
        ep_success = False
        step = 0

        while not done and step < max_steps:
            # ── Policy inference ──────────────────────────────────
            action = policy.predict(prev_obs, prev_action, obs, deterministic=True)
            action = action.flatten()

            # ── Step ──────────────────────────────────────────────
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            if int(info.get("success", 0)) == 1:
                ep_success = True

            # ── Render / record ───────────────────────────────────
            if record_dir:
                frame = env.render()
                if frame is not None:
                    import imageio
                    imageio.imwrite(
                        str(record_path / f"frame_{frame_idx:06d}.png"),
                        frame,
                    )
                    frame_idx += 1
            elif render_mode == "human":
                env.render()
                time.sleep(1.0 / fps)

            # ── Shift history ─────────────────────────────────────
            prev_obs = obs.copy()
            prev_action = action.copy()
            obs = next_obs

            step += 1

            if ep_success:
                # optionally keep rendering a few more frames after success
                # or break immediately:
                break

        successes.append(ep_success)
        episode_lengths.append(step)
        episode_rewards.append(ep_reward)

        status = "SUCCESS" if ep_success else "FAIL"
        print(f"  Episode {ep+1:3d}/{n_episodes}  {status}  steps={step:4d}  reward={ep_reward:.2f}")

    env.close()

    # ── Summary ───────────────────────────────────────────────────
    success_rate = np.mean(successes)
    avg_length = np.mean(episode_lengths)
    avg_reward = np.mean(episode_rewards)

    print(f"\n{'='*50}")
    print(f"Task         : {task_name}")
    print(f"Model        : {model_path}")
    print(f"Episodes     : {n_episodes}")
    print(f"Success rate : {success_rate:.1%} ({sum(successes)}/{n_episodes})")
    print(f"Avg length   : {avg_length:.1f}")
    print(f"Avg reward   : {avg_reward:.3f}")
    print(f"{'='*50}")

    if record_dir:
        print(f"\nFrames saved to {record_dir}/ ({frame_idx} frames)")
        print(f"To make a video:  ffmpeg -framerate {int(fps)} -i {record_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p out.mp4")

    return {
        "success_rate": success_rate,
        "avg_length": avg_length,
        "avg_reward": avg_reward,
        "successes": successes,
        "episode_lengths": episode_lengths,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate a saved DemoPPO model on MetaWorld")
    ap.add_argument("--model", required=True, help="Path to saved .zip model")
    ap.add_argument("--task", default="pick-place-v3", help="MetaWorld task name")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", default=None, help="Directory to save frames (uses rgb_array mode)")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--fps", type=float, default=30.0, help="Render speed (frames per second)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-render", action="store_true", help="Skip rendering, just compute stats")
    args = ap.parse_args()

    render_mode = "human"
    if args.no_render:
        render_mode = "rgb_array"  # still need a mode, just won't display
    if args.record:
        render_mode = "rgb_array"

    evaluate(
        model_path=args.model,
        task_name=args.task,
        n_episodes=args.n_episodes,
        seed=args.seed,
        render_mode=render_mode,
        record_dir=args.record,
        max_steps=args.max_steps,
        fps=args.fps,
        device=args.device,
    )


if __name__ == "__main__":
    main()