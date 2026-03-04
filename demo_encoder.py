"""
DemoEncoder — inference-only wrapper around a trained SC-VAE checkpoint.

Intended for use downstream in PPO rollouts and analysis.

Usage:
    enc = DemoEncoder.from_checkpoint(
        ckpt_path   = "checkpoints/pick-place/best.pt",
        train_demo  = "data/demos/pick-place-v3_demos.npz",
        device      = "cuda",
    )

    # single transition (PPO rollout)
    mu, rho = enc.encode(obs_t, action, obs_next)   # (1, D), (1,)

    # batch
    mu, rho = enc.encode(obs_t, action, obs_next)   # (B, D), (B,)

    # mu only (most common case)
    mu = enc.encode_mu(obs_t, action, obs_next)     # (B, D)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from config import SCVAEConfig
from sc_vae import build_scvae
from demo_dataset import DemoDataset


class DemoEncoder(nn.Module):
    """
    Thin inference wrapper around a trained SC-VAE.

    Handles:
      - checkpoint + config loading
      - obs normalization (using training-set statistics)
      - batch or single-sample encode
      - gradient control (frozen by default for PPO use)
    """

    def __init__(
        self,
        model:     nn.Module,
        cfg:       SCVAEConfig,
        obs_mean:  torch.Tensor,
        obs_std:   torch.Tensor,
        frozen:    bool = True,
    ):
        super().__init__()
        self.model    = model
        self.cfg      = cfg
        self.latent_dim = cfg.latent_dim

        # buffers → move to device automatically with .to(device)
        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_std",  obs_std)

        if frozen:
            self.freeze()

    # ── Construction ──────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path:  Union[str, Path],
        train_demo: Union[str, Path],
        device:     Union[str, torch.device] = "cpu",
        frozen:     bool = True,
    ) -> "DemoEncoder":
        """
        Load a trained checkpoint and pair it with training-set norm stats.

        Args:
            ckpt_path:  path to best.pt / latest.pt
            train_demo: path to the .npz used for training (for norm stats)
            device:     target device
            frozen:     if True, parameters are frozen (no grad)
        """
        ckpt_path = Path(ckpt_path)
        device    = torch.device(device)

        # ── Config ────────────────────────────────────────────────
        cfg_path = ckpt_path.parent / "config.json"
        cfg_dict = json.loads(cfg_path.read_text())
        cfg      = SCVAEConfig(**cfg_dict)

        # ── Model weights ─────────────────────────────────────────
        model = build_scvae(cfg).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(
            f"[DemoEncoder] loaded epoch={ckpt['epoch']} "
            f"val_loss={ckpt['val_loss']:.4f}  ({ckpt_path.name})"
        )

        # ── Norm stats from training data ─────────────────────────
        train_ds = DemoDataset(train_demo, normalize=True)
        obs_mean = torch.from_numpy(train_ds.obs_mean).to(device)
        obs_std  = torch.from_numpy(train_ds.obs_std ).to(device)

        return cls(model, cfg, obs_mean, obs_std, frozen=frozen)

    # ── Core API ──────────────────────────────────────────────────

    @torch.no_grad()
    def encode(
        self,
        obs_t:    torch.Tensor,
        action:   torch.Tensor,
        obs_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of transitions.

        Args:
            obs_t:    (B, obs_dim) or (obs_dim,)  — raw (unnormalized) obs
            action:   (B, act_dim) or (act_dim,)
            obs_next: (B, obs_dim) or (obs_dim,)  — raw (unnormalized) obs

        Returns:
            mu:  (B, latent_dim)  unit vectors on S^{D-1}
            rho: (B,)             concentration scalars in [0, 1)
        """
        obs_t, action, obs_next = self._prepare(obs_t, action, obs_next)
        mu, rho = self.model.encode(obs_t, action, obs_next)
        return mu, rho.squeeze(-1)

    @torch.no_grad()
    def encode_mu(
        self,
        obs_t:    torch.Tensor,
        action:   torch.Tensor,
        obs_next: torch.Tensor,
    ) -> torch.Tensor:
        """Returns only mu — the most common case for PPO feature augmentation."""
        mu, _ = self.encode(obs_t, action, obs_next)
        return mu

    # ── Gradient-enabled variant (for joint fine-tuning experiments) ──

    def encode_with_grad(
        self,
        obs_t:    torch.Tensor,
        action:   torch.Tensor,
        obs_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Same as encode() but allows gradients to flow through mu.
        Only useful when cfg.freeze_encoder=False in the PPO experiment.
        """
        obs_t, action, obs_next = self._prepare(obs_t, action, obs_next)
        mu, rho = self.model.encode(obs_t, action, obs_next)
        return mu, rho.squeeze(-1)

    # ── Utility ───────────────────────────────────────────────────

    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def unfreeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(True)

    @property
    def device(self) -> torch.device:
        return self.obs_mean.device

    def _prepare(
        self,
        obs_t:    torch.Tensor,
        action:   torch.Tensor,
        obs_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ensure 2-D, on correct device, obs normalized."""
        obs_t    = obs_t.to(self.device)
        action   = action.to(self.device)
        obs_next = obs_next.to(self.device)

        if obs_t.dim() == 1:
            obs_t, action, obs_next = (
                obs_t.unsqueeze(0), action.unsqueeze(0), obs_next.unsqueeze(0)
            )

        obs_t    = (obs_t    - self.obs_mean) / self.obs_std
        obs_next = (obs_next - self.obs_mean) / self.obs_std
        return obs_t, action, obs_next

    def __repr__(self) -> str:
        status = "frozen" if not next(self.model.parameters()).requires_grad else "trainable"
        return (
            f"DemoEncoder("
            f"latent_dim={self.latent_dim}, "
            f"obs_dim={self.cfg.obs_dim}, "
            f"act_dim={self.cfg.act_dim}, "
            f"{status})"
        )