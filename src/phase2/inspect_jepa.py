"""Compute per-sample reconstruction error for a trained JEPA.

For each trajectory, draw K random contiguous masks and average the
predictor-vs-target MSE over the masked layers. This gives a scalar
anomaly score per trajectory — the JEPA-SCORE proxy that iter 3 will
compare against true JEPA-SCORE (Jacobian-based).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.phase2.model import LayerJEPA, random_block_mask


@torch.no_grad()
def reconstruction_error(online: LayerJEPA, target: LayerJEPA,
                         x: torch.Tensor, mask_ratio: float,
                         n_masks: int, seed: int = 0) -> torch.Tensor:
    """Per-sample mean MSE across n_masks random contiguous masks."""
    B, L, _ = x.shape
    device = x.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    errs = torch.zeros(B, device=device)
    for _ in range(n_masks):
        starts = torch.randint(0, L - max(1, int(L * mask_ratio)) + 1, (B,), generator=gen)
        block = max(1, int(L * mask_ratio))
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            mask[b, starts[b]:starts[b] + block] = True
        pred = online(x, mask)
        tgt = target.encode(x)
        per_sample = ((pred - tgt) ** 2 * mask.unsqueeze(-1)).sum(dim=(1, 2)) / (mask.sum(dim=1) * pred.size(-1))
        errs += per_sample
    return errs / n_masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jepa", default="results/_jepa_smoke50.pt")
    p.add_argument("--data", default="data/activations_iter1_smoke.pt")
    p.add_argument("--n_masks", type=int, default=16)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(args.jepa, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    online = LayerJEPA(
        d_in=ckpt["d_in"], d_model=cfg["d_model"],
        num_layers=cfg["jepa_layers"], num_heads=cfg["heads"],
        max_L=max(ckpt["L"], 64),
    ).to(device)
    target = LayerJEPA(
        d_in=ckpt["d_in"], d_model=cfg["d_model"],
        num_layers=cfg["jepa_layers"], num_heads=cfg["heads"],
        max_L=max(ckpt["L"], 64),
    ).to(device)
    online.load_state_dict(ckpt["online_state_dict"])
    target.load_state_dict(ckpt["target_state_dict"])
    online.eval()
    target.eval()

    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    traj = blob["trajectories"]
    mean = ckpt["norm_mean"].to("cpu")
    std = ckpt["norm_std"].to("cpu")
    traj_norm = ((traj - mean) / std).to(device)

    errs = reconstruction_error(online, target, traj_norm, cfg["mask_ratio"], args.n_masks)
    e = errs.cpu().numpy()

    print(f"n={len(e)}  mean={e.mean():.4f}  std={e.std():.4f}  min={e.min():.4f}  max={e.max():.4f}")
    print(f"quantiles: q10={float(torch.tensor(e).quantile(0.1)):.4f}  "
          f"q50={float(torch.tensor(e).quantile(0.5)):.4f}  "
          f"q90={float(torch.tensor(e).quantile(0.9)):.4f}")
    top_k = 5
    idx_sorted = e.argsort()
    print(f"\n{top_k} lowest-error prompts (most in-distribution):")
    for i in idx_sorted[:top_k]:
        print(f"  [{e[i]:.4f}] {blob['prompts'][i][:80]!r}")
    print(f"\n{top_k} highest-error prompts (most anomalous):")
    for i in idx_sorted[-top_k:][::-1]:
        print(f"  [{e[i]:.4f}] {blob['prompts'][i][:80]!r}")


if __name__ == "__main__":
    main()
