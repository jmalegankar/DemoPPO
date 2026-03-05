
from __future__ import annotations

import torch
import torch.nn as nn


@torch.jit.interface
class EmbeddingInterface:
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        pass

    def out_dim(self) -> int:
        pass


class MLPObsEmbedding(nn.Module):
    """
    Embeds a flat state observation vector → (B, out_dim).

    Uses LayerNorm instead of BatchNorm so it's safe at batch size 1
    during PPO rollout.

    Args:
        obs_dim:    input dimension (e.g. 39 for MetaWorld state obs)
        hidden_dim: width of hidden layers
        out_dim:    output embedding dimension
        n_layers:   number of hidden layers (≥ 1)
    """

    def __init__(
        self,
        obs_dim:    int,
        hidden_dim: int = 256,
        out_dim:    int = 256,
        n_layers:   int = 2,
    ):
        super().__init__()

        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        layers: list[nn.Module] = [
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ]
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net          = nn.Sequential(*layers)
        self._out_dim_val = out_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim_val

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_dim)
        Returns:
            (B, out_dim)
        """
        if obs.dim() != 2:
            raise ValueError(f"Expected (B, obs_dim), got {tuple(obs.shape)}")
        return self.net(obs)


class CNNObsEmbedding(nn.Module):
    """
    Pixel observation embedding.
    Returns (B, out_dim) via conv stack + flatten + linear projection.

    Args:
        in_channels:   number of image channels (e.g. 3 for RGB)
        conv_channels: channel sizes for each conv block
        out_dim:       final embedding dimension
        input_hw:      (H, W) of input images — used only to infer flat dim
    """

    def __init__(
        self,
        in_channels:   int,
        conv_channels: list[int],
        out_dim:       int = 256,
        input_hw:      tuple[int, int] = (64, 64),
    ):
        super().__init__()

        layers: list[nn.Module] = []
        ch = in_channels
        for out_ch in conv_channels:
            layers += [
                nn.Conv2d(ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            ch = out_ch
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            dummy    = torch.zeros(1, in_channels, *input_hw)
            flat_dim = self.conv(dummy).flatten(1).shape[1]

        self.proj         = nn.Linear(flat_dim, out_dim)
        self._out_dim_val = out_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim_val

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, C, H, W)
        Returns:
            (B, out_dim)
        """
        return self.proj(self.conv(obs).flatten(1))