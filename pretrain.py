"""
SC-VAE pretraining on MetaWorld expert demos.

Usage:
    python pretrain.py --demo data/demos/pick-place-v3_demos.npz --out checkpoints/pick-place
    python pretrain.py --demo data/demos/reach-v3_demos.npz      --out checkpoints/reach
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from demo_dataset import DemoDataset
from config import SCVAEConfig
from sc_vae import build_scvae


# ── KL annealing schedule ─────────────────────────────────────────

def kl_beta(epoch: int, cfg: SCVAEConfig) -> float:
    """
    Epoch 0 .. kl_warmup_epochs-1        → beta = 0
    Epoch kl_warmup_epochs .. +ramp-1    → beta linear 0 → beta_target
    Epoch kl_warmup_epochs + ramp_epochs → beta = beta_target
    """
    if epoch < cfg.kl_warmup_epochs:
        return 0.0
    ramp_progress = (epoch - cfg.kl_warmup_epochs) / max(cfg.kl_ramp_epochs, 1)
    return float(min(ramp_progress, 1.0) * cfg.beta_target)


# ── One epoch ─────────────────────────────────────────────────────

def run_epoch(
    model,
    loader: DataLoader,
    optimizer,
    cfg: SCVAEConfig,
    beta: float,
    device: torch.device,
    train: bool,
) -> dict:
    model.train(train)
    context = torch.enable_grad() if train else torch.no_grad()

    total_recon = total_kl = total_uniform = total_loss = 0.0
    rho_mean_acc = rho_std_acc = 0.0
    n_batches = 0

    with context:
        for s_t, a, s_next in loader:
            s_t, a, s_next = s_t.to(device), a.to(device), s_next.to(device)

            out = model(s_t, a, s_next)
            l_recon, l_kl, l_uniform = model.loss(out)
            loss = l_recon + beta * l_kl + l_uniform

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_recon += l_recon.item()
            total_kl    += l_kl.item()
            total_uniform += l_uniform.item()
            total_loss  += loss.item()
            rho_mean_acc += out.rho.mean().item()
            rho_std_acc  += out.rho.std().item()
            n_batches    += 1

    return {
        "loss":     total_loss  / n_batches,
        "recon":    total_recon / n_batches,
        "kl":       total_kl    / n_batches,
        "uniform":  total_uniform / n_batches,
        "rho_mean": rho_mean_acc / n_batches,
        "rho_std":  rho_std_acc  / n_batches,
    }


# ── Main ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo",    required=True,  help="Path to .npz demo file")
    ap.add_argument("--out",     required=True,  help="Output directory for checkpoints")
    ap.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs",  type=int, default=None, help="Override config epochs")
    args = ap.parse_args()

    device  = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────
    ds             = DemoDataset(args.demo)
    train_ds, val_ds = ds.split(val_fraction=0.1)

    cfg = SCVAEConfig(
        obs_dim = ds.obs_dim,
        act_dim = ds.act_dim,
    )
    if args.epochs:
        cfg.epochs = args.epochs

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=True
    )
    val_loader = DataLoader(
        val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=False
    )

    # ── Model ─────────────────────────────────────────────────────
    model     = build_scvae(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    # save config alongside checkpoints
    (out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))

    print(f"Dataset  : {len(train_ds)} train / {len(val_ds)} val transitions")
    print(f"Model    : {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device   : {device}")
    print(f"Epochs   : {cfg.epochs}  "
          f"(warmup={cfg.kl_warmup_epochs}, ramp={cfg.kl_ramp_epochs}, "
          f"beta_target={cfg.beta_target})")
    print()

    best_val_loss = float("inf")
    log_rows = []

    for epoch in range(cfg.epochs):
        beta = kl_beta(epoch, cfg)
        t0   = time.time()

        train_m = run_epoch(model, train_loader, optimizer, cfg, beta, device, train=True)
        val_m   = run_epoch(model, val_loader,   optimizer, cfg, beta, device, train=False)

        scheduler.step(val_m["loss"])
        elapsed = time.time() - t0

        # ── Logging ───────────────────────────────────────────────
        print(
            f"epoch {epoch+1:03d}/{cfg.epochs} | "
            f"beta={beta:.3f} | "
            f"train loss={train_m['loss']:.4f} "
            f"(recon={train_m['recon']:.4f} kl={train_m['kl']:.4f} uniform={train_m['uniform']:.4f}) | "
            f"val loss={val_m['loss']:.4f} | "
            f"rho {val_m['rho_mean']:.3f}±{val_m['rho_std']:.3f} | "
            f"{elapsed:.1f}s"
        )

        row = {"epoch": epoch + 1, "beta": beta, **{f"train_{k}": v for k, v in train_m.items()},
               **{f"val_{k}": v for k, v in val_m.items()}}
        log_rows.append(row)

        # ── Checkpointing ─────────────────────────────────────────
        # Only track best after warmup — warmup has no KL so val loss
        # is artificially low and not representative of the latent structure
        past_warmup = epoch >= cfg.kl_warmup_epochs
        is_best = past_warmup and val_m["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_m["loss"]
            torch.save(
                {"epoch": epoch + 1, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "val_loss": best_val_loss},
                out_dir / "best.pt",
            )

        # always save latest so training can be resumed
        torch.save(
            {"epoch": epoch + 1, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "val_loss": val_m["loss"]},
            out_dir / "latest.pt",
        )

    # ── Save training log ──────────────────────────────────────────
    import csv
    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints → {out_dir}")
    print(f"Training log → {log_path}")


if __name__ == "__main__":
    main()