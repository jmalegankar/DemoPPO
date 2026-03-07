from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class DemoPPOConfig:
    # ── Observation ────────────────────────────────────────────────
    obs_type: Literal["state", "pixels"] = "state"
    obs_dim:  int = 39          # flat state dim (state mode only)
    act_dim:  int = 4           # continuous action dim

    # ── MLP Embedding (state mode) ─────────────────────────────────
    mlp_hidden_dim: int = 256
    mlp_n_layers:   int = 2
    embed_out_dim:  int = 256   # output dim fed into encoder trunk

    # ── Conv Embedding (pixel mode) ────────────────────────────────
    conv_channels: list[int] = field(default_factory=lambda: [32, 64, 64])

    # ── Action Embedding ───────────────────────────────────────────
    action_embed_dim: int = 64

    # ── Encoder / Decoder trunk ────────────────────────────────────
    hidden_dim: int = 256
    latent_dim: int = 256

    # ── Spherical Cauchy ───────────────────────────────────────────
    kl_quad_points: int   = 64

    # ── VAE loss weights ──────────────────────────────────────────
    beta:         float = 0.5    # KL weight (standalone pre-training)
    gamma:        float = 0.001  # uniformity regularisation weight
    uniformity_t: float = 2.0    # RBF bandwidth for uniformity kernel

    # ── KL annealing (standalone pre-training only) ───────────────
    kl_warmup_epochs: int   = 0
    kl_ramp_epochs:   int   = 0
    beta_target:      float = 0.05

    # ── Online VAE coefficients (inside PPO update) ───────────────
    vae_recon_coef: float = 0.1
    vae_kl_coef:    float = 0.01

    # ── Intrinsic reward ─────────────────────────────────────────
    intrinsic_scale: float = 0.1

    # ── PPO / RL hyperparameters ──────────────────────────────────
    # `discount` used for PPO γ to avoid clash with VAE `gamma` above.
    lr:                  float          = 1e-4
    weight_decay:        float          = 1e-9
    n_steps:             int            = 160   # rollout steps per env
    ppo_batch_size:      int            = 1024     # PPO mini-batch size
    n_epochs:            int            = 1     # PPO update passes
    discount:            float          = 0.99
    gae_lambda:          float          = 0.95
    clip_range:          float          = 0.2
    clip_range_vf:       Optional[float] = None
    normalize_advantage: bool           = True
    ent_coef:            float          = 0.0
    vf_coef:             float          = 0.5
    max_grad_norm:       float          = 0.5
    target_kl:           Optional[float] = None