from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SCVAEConfig:
    # ── Observation ────────────────────────────────────────────────
    obs_type:  Literal["state", "pixels"] = "state"
    obs_dim:   int  = 39          # flat state dim  (state mode only)
    act_dim:   int  = 4           # continuous action dim

    # ── MLP Embedding (state mode) ─────────────────────────────────
    mlp_hidden_dim:  int       = 256
    mlp_n_layers:   int        = 2    # depth of obs embedding MLP
    embed_out_dim:   int       = 256  # output dim of embedding → feeds encoder trunk

    # ── Conv Embedding (pixel mode) ────────────────────────────────
    conv_channels:   list[int] = field(default_factory=lambda: [32, 64, 64])

    # ── Action Embedding ───────────────────────────────────────────
    action_embed_dim: int = 64

    # ── Encoder / Decoder trunk ────────────────────────────────────
    hidden_dim:  int = 256
    latent_dim:  int = 32

    # ── Spherical Cauchy ───────────────────────────────────────────
    rho_min:      float = 0.1
    rho_max:      float = 0.95
    kl_quad_points: int = 64     # Gauss-Legendre points for quadrature KL

    # ── Loss weights ───────────────────────────────────────────────
    beta:         float = 0.5    # KL weight
    gamma:        float = 0.001    # uniformity weight
    uniformity_t: float = 2.0   # bandwidth for uniformity kernel

        # ── Training ───────────────────────────────────────────────────
    lr:               float = 3e-4
    weight_decay:     float = 1e-4
    batch_size:       int   = 256
    epochs:           int   = 100

    # ── KL annealing (matches paper SMILES setup) ──────────────────
    kl_warmup_epochs: int   = 0    # beta=0 for this many epochs
    kl_ramp_epochs:   int   = 0    # linear ramp from 0 → beta_target
    beta_target:      float = 0.05  # final KL weight after ramp