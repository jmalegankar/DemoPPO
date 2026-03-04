"""
Latent space visualizations for trained SC-VAE.

Usage:
    python visualize_latent.py \
        --checkpoint checkpoints/pick-place-rel-no-anneal-uniformity/latest.pt \
        --demo       data/demos/holdout/pick-place-v3_demos.npz \
        --train-demo data/demos/pick-place-v3_demos.npz \
        --out        figures/pick-place-rel-no-anneal-uniformity  

Produces:
    umap_phase.png      — UMAP of mu, colored by task phase (timestep quantile)
    umap_rho.png        — same layout, colored by rho value
    umap_seed.png       — same layout, colored by seed (generalization check)
    rho_hist.png        — histogram of rho values across all transitions
    rho_by_phase.png    — rho distribution split by task phase
    slerp.png           — spherical interpolation between two demo transitions
"""

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from torch.utils.data import DataLoader

from demo_dataset import DemoDataset
from config import SCVAEConfig
from sc_vae import build_scvae


# ── Helpers ───────────────────────────────────────────────────────

def load_model(ckpt_path: str, cfg: SCVAEConfig, device: torch.device):
    model = build_scvae(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    return model


@torch.no_grad()
def encode_dataset(model, demo_path: str, train_stats, device: torch.device):
    """
    Encode all transitions in a demo file.
    Returns dict with mu, rho, timestep_frac, seed_id arrays.
    """
    ds     = DemoDataset(demo_path, normalize=True, stats=train_stats)
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    all_mu  = []
    all_rho = []

    for s_t, a, s_next in loader:
        s_t, a, s_next = s_t.to(device), a.to(device), s_next.to(device)
        mu, rho = model.encode(s_t, a, s_next)
        all_mu.append(mu.cpu().numpy())
        all_rho.append(rho.squeeze(-1).cpu().numpy())

    data     = np.load(demo_path)
    ep_ids   = data["ep_ids"]
    seed_ids = data["seed_ids"]

    # timestep fraction within each episode (0=start, 1=end)
    t_frac = np.zeros(len(ep_ids), dtype=np.float32)
    for ep in np.unique(ep_ids):
        mask       = ep_ids == ep
        n          = mask.sum()
        t_frac[mask] = np.linspace(0, 1, n)

    return {
        "mu":      np.concatenate(all_mu,  axis=0),
        "rho":     np.concatenate(all_rho, axis=0),
        "t_frac":  t_frac,
        "seed_ids": seed_ids,
        "ep_ids":  ep_ids,
    }


def run_umap(mu: np.ndarray) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        return reducer.fit_transform(mu)
    except ImportError:
        print("umap-learn not installed, falling back to PCA")
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(mu)


def slerp(mu_a: np.ndarray, mu_b: np.ndarray, n: int = 11) -> np.ndarray:
    """Spherical linear interpolation between two unit vectors."""
    dot   = np.clip(np.dot(mu_a, mu_b), -1.0, 1.0)
    omega = np.arccos(dot)
    if abs(omega) < 1e-6:
        return np.stack([mu_a] * n)
    ts    = np.linspace(0, 1, n)
    interps = [
        (np.sin((1 - t) * omega) * mu_a + np.sin(t * omega) * mu_b) / np.sin(omega)
        for t in ts
    ]
    return np.stack(interps)   # (n, latent_dim)


# ── Plots ─────────────────────────────────────────────────────────

def plot_umap_phase(emb, t_frac, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=t_frac, cmap="plasma",
                    s=4, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Timestep fraction (0=start, 1=end)")
    ax.set_title("UMAP of μ — colored by task phase")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "umap_phase.png", dpi=150)
    plt.close(fig)
    print("  saved umap_phase.png")


def plot_umap_rho(emb, rho, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=rho, cmap="viridis",
                    s=4, alpha=0.6, rasterized=True,
                    vmin=rho.min(), vmax=rho.max())
    plt.colorbar(sc, ax=ax, label="ρ (concentration)")
    ax.set_title("UMAP of μ — colored by ρ")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "umap_rho.png", dpi=150)
    plt.close(fig)
    print("  saved umap_rho.png")


def plot_umap_seed(emb, seed_ids, out_dir):
    unique_seeds = np.unique(seed_ids)
    cmap         = cm.get_cmap("tab20", len(unique_seeds))
    seed_to_idx  = {s: i for i, s in enumerate(unique_seeds)}
    colors       = [cmap(seed_to_idx[s]) for s in seed_ids]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=4, alpha=0.6, rasterized=True)
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=cmap(i), markersize=6, label=f"seed {s}")
               for i, s in enumerate(unique_seeds)]
    ax.legend(handles=handles, loc="upper right", fontsize=6,
              ncol=2, framealpha=0.7)
    ax.set_title("UMAP of μ — colored by seed (generalization)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "umap_seed.png", dpi=150)
    plt.close(fig)
    print("  saved umap_seed.png")


def plot_rho_hist(rho, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rho, bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(rho.mean(), color="red", linestyle="--",
               label=f"mean={rho.mean():.3f}")
    ax.set_xlabel("ρ (concentration)"); ax.set_ylabel("Count")
    ax.set_title("Distribution of ρ across holdout transitions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rho_hist.png", dpi=150)
    plt.close(fig)
    print("  saved rho_hist.png")


def plot_rho_by_phase(rho, t_frac, out_dir, n_bins=5):
    """
    Split timestep fraction into n_bins quantile buckets,
    plot rho distribution per bucket as overlapping KDEs.
    """
    from scipy.stats import gaussian_kde

    fig, ax   = plt.subplots(figsize=(8, 4))
    edges     = np.linspace(0, 1, n_bins + 1)
    cmap      = cm.get_cmap("plasma", n_bins)
    x_grid    = np.linspace(0, 1, 200)
    labels    = ["start", "early-mid", "mid", "late-mid", "end"]

    for i in range(n_bins):
        mask = (t_frac >= edges[i]) & (t_frac < edges[i + 1])
        if mask.sum() < 10:
            continue
        kde = gaussian_kde(rho[mask], bw_method=0.15)
        ax.plot(x_grid, kde(x_grid), color=cmap(i),
                label=labels[i] if i < len(labels) else f"bin{i}", linewidth=2)
        ax.fill_between(x_grid, kde(x_grid), alpha=0.15, color=cmap(i))

    ax.set_xlabel("ρ (concentration)"); ax.set_ylabel("Density")
    ax.set_title("ρ distribution by task phase")
    ax.legend(title="Phase", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "rho_by_phase.png", dpi=150)
    plt.close(fig)
    print("  saved rho_by_phase.png")


def plot_slerp(mu, t_frac, model, device, out_dir):
    """
    Pick a start-phase transition and an end-phase transition,
    SLERP between their mu's, decode each, show recon dim norms as a heatmap.
    """
    start_idx = np.where(t_frac < 0.1)[0][0]
    end_idx   = np.where(t_frac > 0.9)[0][0]

    mu_a = mu[start_idx]
    mu_b = mu[end_idx]

    path    = slerp(mu_a, mu_b, n=11)               # (11, latent_dim)
    z_path  = torch.tensor(path, dtype=torch.float32).to(device)

    with torch.no_grad():
        recons = model.decode(z_path).cpu().numpy()  # (11, recon_dim)

    # visualise as a heatmap of recon values along the path
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(recons.T, aspect="auto", cmap="RdBu_r",
                   extent=[0, 1, 0, recons.shape[1]])
    plt.colorbar(im, ax=ax, label="Decoded value")
    ax.set_xlabel("Interpolation (0=start transition, 1=end transition)")
    ax.set_ylabel("Reconstruction dim")
    ax.set_title("Spherical interpolation (SLERP) between task phases")
    fig.tight_layout()
    fig.savefig(out_dir / "slerp.png", dpi=150)
    plt.close(fig)
    print("  saved slerp.png")


# ── Main ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",  required=True)
    ap.add_argument("--demo",        required=True, help="Holdout demo .npz")
    ap.add_argument("--train-demo",  required=True, help="Training demo .npz (for norm stats)")
    ap.add_argument("--out",         default="figures/pick-place")
    ap.add_argument("--device",      default="cpu")
    args = ap.parse_args()

    device  = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config from checkpoint dir ───────────────────────────
    cfg_path = Path(args.checkpoint).parent / "config.json"
    cfg_dict = json.loads(cfg_path.read_text())
    cfg      = SCVAEConfig(**cfg_dict)

    # ── Load model ────────────────────────────────────────────────
    model = load_model(args.checkpoint, cfg, device)

    # ── Normalization stats from training set ─────────────────────
    train_ds    = DemoDataset(args.train_demo, normalize=True)
    train_stats = train_ds.stats   # (mean, std) computed on training data

    # ── Encode holdout ────────────────────────────────────────────
    print("Encoding holdout transitions ...")
    enc = encode_dataset(model, args.demo, train_stats, device)
    print(f"  {len(enc['mu'])} transitions | "
          f"rho mean={enc['rho'].mean():.3f} std={enc['rho'].std():.3f}")

    # ── UMAP ──────────────────────────────────────────────────────
    print("Running UMAP ...")
    emb = run_umap(enc["mu"])

    # ── Generate all plots ────────────────────────────────────────
    print("Generating plots ...")
    plot_umap_phase(emb, enc["t_frac"],   out_dir)
    plot_umap_rho  (emb, enc["rho"],      out_dir)
    plot_umap_seed (emb, enc["seed_ids"], out_dir)
    plot_rho_hist  (enc["rho"],           out_dir)
    plot_rho_by_phase(enc["rho"], enc["t_frac"], out_dir)
    plot_slerp     (enc["mu"], enc["t_frac"], model, device, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()