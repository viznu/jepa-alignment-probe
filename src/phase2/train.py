"""Train LayerJEPA on saved activation trajectories.

Loss: MSE between predictor output and EMA-target encoder output, measured
only at masked layer positions. Target encoder is a copy whose weights EMA
the online encoder (no gradient flow through target).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.phase2.model import LayerJEPA, random_block_mask


def ema_update(target: torch.nn.Module, online: torch.nn.Module, tau: float) -> None:
    for p_t, p in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(tau).add_(p.data, alpha=1.0 - tau)


def compute_loss(online: LayerJEPA, target: LayerJEPA,
                 x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = online(x, mask)  # (B, L, d_model)
    with torch.no_grad():
        tgt = target.encode(x)  # (B, L, d_model), EMA target — no mask
    diff = (pred - tgt)[mask]  # (n_masked, d_model)
    return diff.pow(2).mean()


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    traj = blob["trajectories"]  # (N, L, d)
    N, L, d = traj.shape
    print(f"loaded {args.data}: N={N} L={L} d={d}")

    # Z-normalize per layer across the dataset — stabilizes training given
    # the 30x norm growth from layer 0 to layer 34.
    mean = traj.mean(dim=0, keepdim=True)
    std = traj.std(dim=0, keepdim=True).clamp_min(1e-6)
    traj_norm = (traj - mean) / std

    n_val = max(1, int(N * args.val_frac))
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr = traj_norm[tr_idx]
    vl = traj_norm[val_idx]
    print(f"train={len(tr)} val={len(vl)}")

    loader = DataLoader(TensorDataset(tr), batch_size=args.batch_size, shuffle=True, drop_last=False)

    online = LayerJEPA(d_in=d, d_model=args.d_model, num_layers=args.jepa_layers,
                       num_heads=args.heads, max_L=max(L, 64)).to(device)
    target = copy.deepcopy(online).to(device)
    for p in target.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in online.parameters())
    print(f"params={n_params:,}  d_model={args.d_model}  jepa_layers={args.jepa_layers}")

    opt = torch.optim.AdamW(online.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log = []
    for epoch in range(args.epochs):
        t0 = time.time()
        online.train()
        tr_losses = []
        for (x,) in loader:
            x = x.to(device)
            mask = random_block_mask(x.size(0), L, ratio=args.mask_ratio, device=device)
            loss = compute_loss(online, target, x, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
            opt.step()
            ema_update(target, online, tau=args.ema_tau)
            tr_losses.append(loss.item())
        sched.step()

        online.eval()
        with torch.no_grad():
            xv = vl.to(device)
            mv = random_block_mask(xv.size(0), L, ratio=args.mask_ratio, device=device)
            val_loss = compute_loss(online, target, xv, mv).item()

        tr_loss = sum(tr_losses) / len(tr_losses)
        lr_now = opt.param_groups[0]["lr"]
        dt = time.time() - t0
        log.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, "lr": lr_now, "time": dt})
        print(f"epoch {epoch:3d}  train={tr_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}  ({dt:.1f}s)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "online_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "config": vars(args),
        "log": log,
        "norm_mean": mean,
        "norm_std": std,
        "d_in": d,
        "L": L,
    }, out)
    with open(out.with_suffix(".log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/activations_iter1_smoke.pt")
    p.add_argument("--output", default="results/jepa_iter1.pt")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--jepa_layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mask_ratio", type=float, default=0.4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--ema_tau", type=float, default=0.996)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
