import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple

from .utils import sc_kl_uniform, uniformity_loss, sc_sample
from .embeddings import EmbeddingInterface, MLPObsEmbedding, CNNObsEmbedding
from .config import DemoPPOConfig

@th.jit.script
class VAEOutput:
    def __init__(
        self,
        recon: th.Tensor,
        mu: th.Tensor,
        rho: th.Tensor,
        recon_target: th.Tensor,
        skips: Optional[List[th.Tensor]] = None,
    ):
        self.recon        = recon
        self.mu           = mu
        self.rho          = rho
        self.recon_target = recon_target
        self.skips        = skips


@th.jit.script
class VAELoss:
    def __init__(
        self,
        recon_loss: th.Tensor,
        kl_loss: th.Tensor,
        aux_loss: Optional[th.Tensor] = None,
    ):
        self.recon_loss = recon_loss
        self.kl_loss    = kl_loss
        self.aux_loss   = aux_loss


@th.jit.interface
class VAEInterface:
    def encode(
        self, s_t: th.Tensor, a_t: th.Tensor, s_tp1: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[List[th.Tensor]]]:
        pass

    def decode(
        self, z: th.Tensor
    ) -> th.Tensor:
        pass

    def forward(
        self, s_t: th.Tensor, a_t: th.Tensor, s_tp1: th.Tensor
    ) -> VAEOutput:
        pass

    def loss(self, output: VAEOutput) -> VAELoss:
        pass


# ===================================================================
# TransitionSCVAE
# ===================================================================


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

    def __init__(self, embedding: EmbeddingInterface, cfg: DemoPPOConfig):
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

    def _embed(self, obs: th.Tensor) -> th.Tensor:
        return self.embedding(obs)    # (B, E)

    def _encode_parts(
        self,
        s_t:    th.Tensor,
        a_t:    th.Tensor,
        s_next: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        h_s   = self._embed(s_t)           # (B, E)
        h_sn  = self._embed(s_next)        # (B, E)
        a_emb = self.action_embed(a_t)     # (B, A)
        return h_s, a_emb, h_sn

    def _build_target(
        self,
        s_t:    th.Tensor,
        a_t:    th.Tensor,
        s_next: th.Tensor,
    ) -> th.Tensor:
        with th.no_grad():
            h_s, a_emb, h_sn = self._encode_parts(s_t, a_t, s_next)
        return th.cat([h_s, a_emb, h_sn], dim=-1)   # (B, 2E+A)

    # ── Public API ────────────────────────────────────────────────

    def encode(
        self,
        s_t:    th.Tensor,
        a_t:    th.Tensor,
        s_next: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, Optional[List[th.Tensor]]]:
        h_s, a_emb, h_sn = self._encode_parts(s_t, a_t, s_next)
        h   = self.encoder_trunk(th.cat([h_s, a_emb, h_sn], dim=-1))
        mu  = F.normalize(self.fc_mu(h), p=2, dim=-1)
        rho = th.sigmoid(self.fc_rho(h))   # free in (0, 1)
        skips: Optional[List[th.Tensor]] = None 

        return mu, rho, skips     # (B, latent_dim), (B, 1), Optional[List of skip tensors]

    def decode(self, z: th.Tensor) -> th.Tensor:
        return self.decoder(z)   # (B, 2E+A)

    def encode_state(self, s_t: th.Tensor) -> th.Tensor:
        return self._embed(s_t)   # (B, E)

    def forward(
        self,
        s_t:    th.Tensor,
        a_t:    th.Tensor,
        s_next: th.Tensor,
    ) -> VAEOutput:
        mu, rho, skips      = self.encode(s_t, a_t, s_next)
        z            = sc_sample(mu, rho) if self.training else mu
        recon        = self.decode(z)
        recon_target = self._build_target(s_t, a_t, s_next)
        return VAEOutput(recon, mu, rho, recon_target, skips)

    def loss(self, out: VAEOutput) -> VAELoss:
        l_recon = F.mse_loss(out.recon, out.recon_target)
        l_kl    = sc_kl_uniform(out.rho, self.latent_dim).mean()
        l_uniform = uniformity_loss(out.mu, t=self.cfg.uniformity_t)
        return VAELoss(l_recon, l_kl, l_uniform)


# ===================================================================
# Factory
# ===================================================================

def build_scvae(cfg: DemoPPOConfig) -> TransitionSCVAE:
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