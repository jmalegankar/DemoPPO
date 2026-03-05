"""
DemoPPO training on MetaWorld.

Usage:
    # warm-start VAE from pretrained checkpoint
    python train_meta_world.py --task pick-place-v3 --checkpoint checkpoints/pick-place/best.pt

    # train VAE fully online from scratch
    python train_meta_world.py --task reach-v3 --out runs/reach
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from helpers.obs_transform import transform_obs_numpy
from models.config import DemoPPOConfig
from models.sc_vae import TransitionSCVAE
from models.embeddings import MLPObsEmbedding
from demo_ppo.demo_ppo import DemoPPO
from demo_ppo.policies import DemoActorCriticPolicy


# ── MetaWorld gymnasium wrapper ───────────────────────────────────────────────

class MetaWorldEnv(gym.Env):
    """
    Wraps Meta-World/MT1 and applies the relative observation transform
    (raw 39-dim → seed-agnostic 13-dim).  Action space is unchanged (4-dim
    continuous).
    """

    OBS_DIM = 13   # output of transform_obs_numpy
    ACT_DIM = 4    # MetaWorld default

    def __init__(self, task_name: str):
        self._env = gym.make("Meta-World/MT1", env_name=task_name, render_mode=None)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = self._env.action_space

    # gymnasium API ────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return transform_obs_numpy(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        # use sparse success signal as reward so the agent learns to succeed
        reward = float(info.get("success", 0))
        return transform_obs_numpy(obs), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


def make_env(task_name: str, seed: int):
    """Factory for SubprocVecEnv — must return a callable."""
    def _init():
        env = MetaWorldEnv(task_name)
        env.reset(seed=seed)
        return env
    return _init


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",       default="pick-place-v3",
                    help="MetaWorld task name (e.g. reach-v3, pick-place-v3)")
    ap.add_argument("--checkpoint", default=None,
                    help="Pretrained SC-VAE .pt checkpoint to warm-start the VAE")
    ap.add_argument("--out",        default="runs/demo_ppo",
                    help="Directory to save tensorboard logs and final model")
    ap.add_argument("--timesteps",  type=int, default=1_000_000)
    ap.add_argument("--n-envs",     type=int, default=4,
                    help="Number of parallel envs (1 → DummyVecEnv, >1 → SubprocVecEnv)")
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed",       type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = DemoPPOConfig(
        obs_type="state",
        obs_dim=MetaWorldEnv.OBS_DIM,   # 13
        act_dim=MetaWorldEnv.ACT_DIM,   # 4
    )

    # ── Environments ──────────────────────────────────────────────────────────
    env_fns = [make_env(args.task, args.seed + i) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)

    # ── Build embedding to pass into the policy's VAE ─────────────────────────
    # DemoActorCriticPolicy constructs TransitionSCVAE(embedding, cfg) internally.
    embedding = MLPObsEmbedding(
        obs_dim=cfg.obs_dim,
        hidden_dim=cfg.mlp_hidden_dim,
        out_dim=cfg.embed_out_dim,
        n_layers=cfg.mlp_n_layers,
    )

    # ── DemoPPO ───────────────────────────────────────────────────────────────
    model = DemoPPO(
        policy=DemoActorCriticPolicy,
        env=env,
        null_action=np.zeros(cfg.act_dim, dtype=np.float32),
        learning_rate=cfg.lr,
        n_steps=cfg.n_steps,
        batch_size=cfg.ppo_batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.discount,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        clip_range_vf=cfg.clip_range_vf,
        normalize_advantage=cfg.normalize_advantage,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        vae_recon_coef=cfg.vae_recon_coef,
        vae_kl_coef=cfg.vae_kl_coef,
        intrinsic_scale=cfg.intrinsic_scale,
        max_grad_norm=cfg.max_grad_norm,
        target_kl=cfg.target_kl,
        tensorboard_log=str(out_dir / "tb"),
        policy_kwargs={
            "features_extractor_kwargs": {"mu_dim": cfg.latent_dim},
            "net_arch": [256, 256],
        },
        verbose=1,
        seed=args.seed,
        device=args.device,
        vae_features_extractor_class=TransitionSCVAE,
        vae_features_extractor_kwargs={"embedding": embedding, "cfg": cfg},
    )

    # ── Optionally warm-start the VAE from a pretrained checkpoint ────────────
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        model.policy.vae_feature_extractor.load_state_dict(ckpt["model"])
        print(f"Loaded pretrained SC-VAE from {args.checkpoint}  (epoch {ckpt['epoch']})")

    print(f"Task      : {args.task}")
    print(f"Obs dim   : {cfg.obs_dim}  →  latent dim: {cfg.latent_dim}")
    print(f"Timesteps : {args.timesteps:,}")
    print(f"N envs    : {args.n_envs}")
    print(f"Device    : {args.device}")
    print(f"Output    : {out_dir}")
    print()

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=args.task,
        progress_bar=True,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = out_dir / "final_model"
    model.save(str(save_path))
    print(f"\nModel saved → {save_path}")


if __name__ == "__main__":
    main()