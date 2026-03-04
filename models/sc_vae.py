from __future__ import annotations

from typing import NamedTuple, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.obs_embeddings import MLPObsEmbedding, CNNObsEmbedding
from models.config import SCVAEConfig


# ===================================================================
# Spherical Cauchy helpers
# ===================================================================

def _mobius_add(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    a_sq = (a * a).sum(-1, keepdim=True)
    x_sq = (x * x).sum(-1, keepdim=True)
    ax   = (a * x).sum(-1, keepdim=True)
    num  = (1 + 2 * ax + x_sq) * a + (1 - a_sq) * x
    den  = 1 + 2 * ax + a_sq * x_sq
    return F.normalize(num / (den + 1e-8), p=2, dim=-1)


def _sc_sample(mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    xi = F.normalize(torch.randn_like(mu), p=2, dim=-1)
    return _mobius_add(rho * mu, xi)



def _sc_kl_asymptotic(rho: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Proposition 2 (paper §3.2.2): KL ≈ (d-1)*log((1+ρ)/(1-ρ)) + ψ((d-1)/2) - ψ(d-1)
    Verified against paper eq. directly. Valid when ρ > 0.9 (z → 1).
    """
    d_minus_1  = float(dim - 1)
    # (d-1) * log((1+ρ)/(1-ρ)) — positive ✓
    log_term   = d_minus_1 * (torch.log1p(rho) - torch.log1p(-rho))
    # ψ((d-1)/2) - ψ(d-1) — negative but small relative to log_term ✓
    correction = (
        torch.digamma(torch.tensor(d_minus_1 / 2.0, device=rho.device, dtype=rho.dtype))
        - torch.digamma(torch.tensor(d_minus_1,     device=rho.device, dtype=rho.dtype))
    )
    return (log_term + correction).clamp_min(0.0)


_GL_CACHE: dict = {}


def _gauss_legendre_01(n: int, device: torch.device, dtype: torch.dtype) -> tuple:
    """
    Gauss-Legendre nodes and weights mapped to [0, 1].
    Cached per (n, device, dtype).
    """
    key = (n, str(device), str(dtype))
    if key in _GL_CACHE:
        return _GL_CACHE[key]

    import numpy as np
    nodes_np, weights_np = np.polynomial.legendre.leggauss(n)
    # map [-1,1] -> [0,1]: t = (s+1)/2, w -> w/2
    t = torch.tensor((nodes_np + 1.0) / 2.0, device=device, dtype=dtype)
    w = torch.tensor(weights_np / 2.0,        device=device, dtype=dtype)
    _GL_CACHE[key] = (t, w)
    return t, w


def _sc_kl_quadrature(
    rho: torch.Tensor,
    dim: int,
    *,
    n_points: int = 64,
) -> torch.Tensor:
    """
    Proposition 1 (paper §3.2.2): quadrature-based KL for ρ ≤ 0.9.

    KL = (d-1)*log((1-ρ)/(1+ρ))
       + (d-1) * ∫_0^1 [t^{d-2}/(1-t)] * [1 - ((1-ρ)²/((1+ρ)²-4ρt))^{(d-1)/2}] dt

    The integrand has a removable singularity at t=1 (both numerator 1/(1-t)
    and the bracket → 0, limit is finite: 2(d-1)ρ/(1-ρ)²). Standard G-L
    handles this stably with enough points.

    This avoids the pref=((1-ρ)/(1+ρ))^{d-1} underflow in the power series
    branch, which collapses to 0 for dim≥16 at moderate ρ.
    """
    device, dtype = rho.device, rho.dtype
    d_minus_1     = float(dim - 1)
    alpha         = d_minus_1 / 2.0

    t_gl, w_gl = _gauss_legendre_01(n_points, device, dtype)  # (Q,)

    # broadcast: rho (B,1), t (1,Q) -> (B,Q)
    t_gl = t_gl.unsqueeze(0)    # (1, Q)
    w_gl = w_gl.unsqueeze(0)    # (1, Q)

    # ((1-ρ)² / ((1+ρ)² - 4ρt))^α
    denom = (1.0 + rho).pow(2) - 4.0 * rho * t_gl   # (B, Q)
    denom = denom.clamp_min(1e-10)
    ratio = (1.0 - rho).pow(2) / denom               # (B, Q), in (0,1]
    inner = 1.0 - ratio.pow(alpha)                   # (B, Q), in [0,1)

    # t^{d-2} / (1-t): clamp near t=1 (singularity is removable, inner→0 there)
    t_pow        = t_gl.pow(max(d_minus_1 - 1.0, 0.0))   # (1, Q)
    one_minus_t  = (1.0 - t_gl).clamp_min(1e-10)          # (1, Q)
    integrand    = (t_pow / one_minus_t) * inner           # (B, Q)

    integral = (w_gl * integrand).sum(dim=-1, keepdim=True)  # (B, 1)

    # term1 is negative (log of value < 1); term2 is positive and larger
    term1 = d_minus_1 * (torch.log1p(-rho) - torch.log1p(rho))
    term2 = d_minus_1 * integral
    return (term1 + term2).clamp_min(0.0)


_RHO_ASYMP = 0.9

def _sc_kl_uniform(
    rho: torch.Tensor,
    dim: int,
    *,
    n_quad_points: int = 64,
) -> torch.Tensor:
    """
    KL( spCauchy_d(·|μ,ρ) ‖ Uniform(S^{d-1}) )

    Branches (matching paper §3.2.2):
      ρ ≤ 0.9  →  Proposition 1 (Gauss-Legendre quadrature) — stable for all d
      ρ >  0.9 →  Proposition 2 (asymptotic)

    Args:
        rho:           (B,) or (B,1) in [0, 1)
        dim:           latent dimension d ≥ 2
        n_quad_points: G-L quadrature points for the low-ρ branch
    Returns:
        kl: (B,1) non-negative tensor
    """
    if dim < 2:
        raise ValueError("dim must be >= 2")

    rho = rho.to(dtype=torch.float32)
    if rho.dim() == 1:
        rho = rho.unsqueeze(-1)
    rho = rho.clamp(0.0, 1.0 - 1e-7)

    high_rho = rho > _RHO_ASYMP   # (B,1)
    kl       = torch.zeros_like(rho)

    # ── Branch A: quadrature (ρ ≤ 0.9) ──────────────────────────
    if high_rho.logical_not().any():
        rho_lo         = rho.clone(); rho_lo[high_rho] = 0.5
        kl_lo          = _sc_kl_quadrature(rho_lo, dim, n_points=n_quad_points)
        kl             = torch.where(high_rho, kl, kl_lo)

    # ── Branch B: asymptotic (ρ > 0.9) ───────────────────────────
    if high_rho.any():
        rho_hi = rho.clone(); rho_hi[~high_rho] = 0.5
        kl     = torch.where(high_rho, _sc_kl_asymptotic(rho_hi, dim), kl)

    return kl


def _uniformity_loss(mu: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    sq_dists = 2.0 - 2.0 * (mu @ mu.T)
    mask     = ~torch.eye(mu.size(0), dtype=torch.bool, device=mu.device)
    return torch.log(torch.exp(-t * sq_dists[mask]).mean())


# ===================================================================
# Output type
# ===================================================================

class SCVAEForwardOutput(NamedTuple):
    recon:        torch.Tensor   # (B, recon_target_dim)
    mu:           torch.Tensor   # (B, latent_dim)  unit vectors
    rho:          torch.Tensor   # (B, 1)
    z:            torch.Tensor   # (B, latent_dim)
    recon_target: torch.Tensor   # (B, recon_target_dim)


# ===================================================================
# TransitionSCVAE  —  embedding-agnostic
# ===================================================================

ObsEmbedding = Union[MLPObsEmbedding, CNNObsEmbedding]


class TransitionSCVAE(nn.Module):
    """
    Spherical Cauchy VAE over (s_t, a_t, s_{t+1}) transitions.

    The observation embedding is injected — pass MLPObsEmbedding for state
    obs or CNNObsEmbedding for pixels. The rest of the model is identical.

    The encoder trunk sees:
        [embed(s_t)  |  action_embed(a_t)  |  embed(s_next)]
        dim: embed_out_dim + action_embed_dim + embed_out_dim

    The decoder reconstructs the same concatenation as the regression target.
    """

    def __init__(self, embedding: ObsEmbedding, cfg: SCVAEConfig):
        super().__init__()

        self.cfg        = cfg
        self.embedding  = embedding
        self.latent_dim = cfg.latent_dim

        E = embedding.out_dim          # embedding output dim
        A = cfg.action_embed_dim
        H = cfg.hidden_dim

        # ── Action embedding (continuous → dense) ─────────────────
        self.action_embed = nn.Sequential(
            nn.Linear(cfg.act_dim, A),
            nn.ReLU(),
        )

        # ── Encoder trunk ─────────────────────────────────────────
        # input: [h_s | a_emb | h_sn]
        self.encoder_trunk = nn.Sequential(
            nn.Linear(2 * E + A, H),
            nn.LayerNorm(H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.LayerNorm(H),
            nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(H, cfg.latent_dim)
        self.fc_rho = nn.Linear(H, 1)

        # ── Decoder ───────────────────────────────────────────────
        # reconstructs [embed(s_t) | action_embed(a_t) | embed(s_next)]
        self._recon_dim = 2 * E + A
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, self._recon_dim),
        )

    # ── Internal helpers ──────────────────────────────────────────

    def _embed(self, obs: torch.Tensor) -> torch.Tensor:
        return self.embedding(obs)    # (B, E)

    def _encode_parts(
        self,
        s_t:    torch.Tensor,
        a_t:    torch.Tensor,
        s_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_s   = self._embed(s_t)           # (B, E)
        h_sn  = self._embed(s_next)        # (B, E)
        a_emb = self.action_embed(a_t)     # (B, A)
        return h_s, a_emb, h_sn

    def _build_target(
        self,
        s_t:    torch.Tensor,
        a_t:    torch.Tensor,
        s_next: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            h_s, a_emb, h_sn = self._encode_parts(s_t, a_t, s_next)
        return torch.cat([h_s, a_emb, h_sn], dim=-1)   # (B, 2E+A)

    # ── Public API ────────────────────────────────────────────────

    def encode(
        self,
        s_t:    torch.Tensor,
        a_t:    torch.Tensor,
        s_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_s, a_emb, h_sn = self._encode_parts(s_t, a_t, s_next)
        h   = self.encoder_trunk(torch.cat([h_s, a_emb, h_sn], dim=-1))
        mu  = F.normalize(self.fc_mu(h), p=2, dim=-1)
        rho = torch.sigmoid(self.fc_rho(h))   # free in (0, 1)
        return mu, rho     # (B, latent_dim), (B, 1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)   # (B, 2E+A)

    def encode_state(self, s_t: torch.Tensor) -> torch.Tensor:
        """
        Returns the embedding of s_t only.
        This is what the PPO policy consumes — no future obs required.
        Gradient flow is controlled by the caller (detach for frozen encoder).
        """
        return self._embed(s_t)   # (B, E)

    def forward(
        self,
        s_t:    torch.Tensor,
        a_t:    torch.Tensor,
        s_next: torch.Tensor,
    ) -> SCVAEForwardOutput:
        mu, rho      = self.encode(s_t, a_t, s_next)
        z            = _sc_sample(mu, rho) if self.training else mu
        recon        = self.decode(z)
        recon_target = self._build_target(s_t, a_t, s_next)
        return SCVAEForwardOutput(recon, mu, rho, z, recon_target)

    def loss(self, out: SCVAEForwardOutput) -> Tuple[torch.Tensor, torch.Tensor]:
        l_recon = F.mse_loss(out.recon, out.recon_target)
        l_kl    = _sc_kl_uniform(
            out.rho, self.latent_dim, n_quad_points=self.cfg.kl_quad_points
        ).mean()
        l_uniform = _uniformity_loss(out.mu, t=self.cfg.uniformity_t)
        return l_recon, l_kl, self.cfg.gamma * l_uniform


# ===================================================================
# Factory
# ===================================================================

def build_scvae(cfg: SCVAEConfig) -> TransitionSCVAE:
    """Construct the right embedding + VAE from config alone."""
    if cfg.obs_type == "state":
        embedding = MLPObsEmbedding(
            obs_dim=cfg.obs_dim,
            hidden_dim=cfg.mlp_hidden_dim,
            out_dim=cfg.embed_out_dim,
            n_layers=cfg.mlp_n_layers,
        )
    elif cfg.obs_type == "pixels":
        embedding = CNNObsEmbedding(
            in_channels=3,
            conv_channels=cfg.conv_channels,
            out_dim=cfg.embed_out_dim,
        )
    else:
        raise ValueError(f"Unknown obs_type: {cfg.obs_type}")

    return TransitionSCVAE(embedding, cfg)