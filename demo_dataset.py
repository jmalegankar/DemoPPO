"""
DemoDataset — wraps collected .npz demo files for SC-VAE pretraining.

Returns (obs_t, action, obs_next) tuples as float32 tensors.
Handles normalization, train/val split, and multi-task loading.

Usage:
    ds = DemoDataset("data/demos/reach-v3_demos.npz")
    train_ds, val_ds = ds.split(val_fraction=0.1)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    for obs_t, action, obs_next in loader:
        out = model(obs_t, action, obs_next)
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset


class DemoDataset(Dataset):
    """
    Single-task demo dataset.

    Args:
        path:       path to .npz file produced by collect_demos.py
        normalize:  if True, obs are z-scored using dataset statistics
                    (stats computed on load, stored as .obs_mean / .obs_std)
        stats:      optional pre-computed (mean, std) tuple — pass this when
                    building a val set so it uses the train split's statistics
    """

    def __init__(
        self,
        path: Union[str, Path],
        normalize: bool = True,
        stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        data     = np.load(path)
        obs_t    = data["obs_t"].astype(np.float32)
        action   = data["action"].astype(np.float32)
        obs_next = data["obs_next"].astype(np.float32)

        # ----------------------------------------------------------------
        # Normalization
        # ----------------------------------------------------------------
        if normalize:
            if stats is None:
                self.obs_mean = obs_t.mean(axis=0)
                self.obs_std  = obs_t.std(axis=0) + 1e-8
            else:
                self.obs_mean, self.obs_std = stats

            obs_t    = (obs_t    - self.obs_mean) / self.obs_std
            obs_next = (obs_next - self.obs_mean) / self.obs_std
        else:
            self.obs_mean = np.zeros(obs_t.shape[-1], dtype=np.float32)
            self.obs_std  = np.ones( obs_t.shape[-1], dtype=np.float32)

        # ----------------------------------------------------------------
        # Store as tensors
        # ----------------------------------------------------------------
        self.obs_t    = torch.from_numpy(obs_t)
        self.action   = torch.from_numpy(action)
        self.obs_next = torch.from_numpy(obs_next)
        self.ep_ids   = torch.from_numpy(data["ep_ids"])

        self.obs_dim = self.obs_t.shape[-1]
        self.act_dim = self.action.shape[-1]

    def __len__(self) -> int:
        return len(self.obs_t)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.obs_t[idx], self.action[idx], self.obs_next[idx]

    # ------------------------------------------------------------------
    # Convenience: train / val split preserving normalization stats
    # ------------------------------------------------------------------
    def split(
        self,
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> Tuple["DemoDataset", "_SubsetDataset"]:
        """
        Returns (train_dataset, val_dataset).
        val_dataset shares the same normalization stats as train.
        """
        n       = len(self)
        n_val   = max(1, int(n * val_fraction))
        n_train = n - n_val

        gen = torch.Generator().manual_seed(seed)
        train_idx, val_idx = random_split(
            range(n), [n_train, n_val], generator=gen
        )

        return (
            _SubsetDataset(self, list(train_idx)),
            _SubsetDataset(self, list(val_idx)),
        )

    @property
    def stats(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.obs_mean, self.obs_std

    def __repr__(self) -> str:
        return (
            f"DemoDataset("
            f"n={len(self)}, obs_dim={self.obs_dim}, act_dim={self.act_dim})"
        )


class _SubsetDataset(Dataset):
    """Index into a DemoDataset without copying tensors."""

    def __init__(self, parent: DemoDataset, indices: list):
        self.parent  = parent
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.parent[self.indices[idx]]


# ------------------------------------------------------------------
# Multi-task: concatenate reach + pick-place into one dataset
# ------------------------------------------------------------------

def load_multitask(
    paths: list[Union[str, Path]],
    normalize: bool = True,
) -> ConcatDataset:
    """
    Load multiple task demos and concatenate.
    Normalization is computed independently per task —
    this matters because reach and pick-place have different obs ranges.

    Returns a ConcatDataset suitable for DataLoader.
    """
    datasets = [DemoDataset(p, normalize=normalize) for p in paths]
    return ConcatDataset(datasets)


# # ------------------------------------------------------------------
# # Quick sanity check
# # ------------------------------------------------------------------
# if __name__ == "__main__":
#     import sys

#     path = sys.argv[1] if len(sys.argv) > 1 else "data/demos/reach-v3_demos.npz"
#     ds   = DemoDataset(path)
#     print(ds)

#     train_ds, val_ds = ds.split(val_fraction=0.1)
#     print(f"  train={len(train_ds)}  val={len(val_ds)}")
#     print(f"  obs_dim={ds.obs_dim}  act_dim={ds.act_dim}")

#     loader = DataLoader(train_ds, batch_size=256, shuffle=True)
#     s_t, a, s_n = next(iter(loader))
#     print(f"\nSample batch:")
#     print(f"  obs_t    {tuple(s_t.shape)}  mean={s_t.mean():.3f}  std={s_t.std():.3f}")
#     print(f"  action   {tuple(a.shape)}   mean={a.mean():.3f}  std={a.std():.3f}")
#     print(f"  obs_next {tuple(s_n.shape)}  mean={s_n.mean():.3f}  std={s_n.std():.3f}")