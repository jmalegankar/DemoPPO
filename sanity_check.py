"""
Quick wiring check — runs one forward + loss pass before committing to training.

Usage:
    python sanity_check.py --demo data/demos/pick-place-v3_demos.npz
"""

import argparse
import torch
from torch.utils.data import DataLoader

from demo_dataset import DemoDataset
from config import SCVAEConfig
from sc_vae import build_scvae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="data/demos/pick-place-v3_demos.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────
    ds     = DemoDataset(args.demo)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    s_t, a, s_next = next(iter(loader))
    s_t    = s_t.to(device)
    a      = a.to(device)
    s_next = s_next.to(device)

    print(f"\nDataset : {ds}")
    print(f"Batch   : s_t={tuple(s_t.shape)}  a={tuple(a.shape)}  s_next={tuple(s_next.shape)}")

    # ── Model ─────────────────────────────────────────────────────
    cfg = SCVAEConfig(
        obs_type = "state",
        obs_dim  = ds.obs_dim,
        act_dim  = ds.act_dim,
    )
    model = build_scvae(cfg).to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel   : {total_params:,} parameters")

    # ── Forward ───────────────────────────────────────────────────
    out = model(s_t, a, s_next)
    print(f"\nForward pass:")
    print(f"  mu    {tuple(out.mu.shape)}   norm={out.mu.norm(dim=-1).mean():.4f} (should be ~1.0)")
    print(f"  rho   {tuple(out.rho.shape)}  mean={out.rho.mean():.4f}  std={out.rho.std():.4f}")
    print(f"  z     {tuple(out.z.shape)}")
    print(f"  recon {tuple(out.recon.shape)}")

    # ── Loss ──────────────────────────────────────────────────────
    l_recon, l_kl = model.loss(out)
    loss = l_recon + cfg.beta_target * l_kl

    print(f"\nLoss:")
    print(f"  recon   = {l_recon.item():.4f}")
    print(f"  kl      = {l_kl.item():.4f}")
    print(f"  total   = {loss.item():.4f}")

    # ── Backward ──────────────────────────────────────────────────
    loss.backward()
    print(f"\nBackward pass: OK")

    # ── encode_state (what PPO will call) ─────────────────────────
    model.eval()
    with torch.no_grad():
        z_state = model.encode_state(s_t)
    print(f"\nencode_state: {tuple(z_state.shape)}  (PPO policy input)")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()