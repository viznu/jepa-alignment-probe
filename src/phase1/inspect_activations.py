"""Sanity-check a saved trajectory tensor: shape, dtype, finiteness, per-layer norms."""
import argparse
from pathlib import Path

import torch


def inspect(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    traj = blob["trajectories"]
    N, L, d = traj.shape

    print(f"file        : {path}")
    print(f"model       : {blob['model_name']}")
    print(f"token_pos   : {blob.get('token_position', '?')}")
    print(f"shape       : N={N} L={L} d={d}")
    print(f"dtype       : {traj.dtype}")
    print(f"size_MB     : {traj.element_size() * traj.nelement() / 1e6:.1f}")
    print(f"finite      : {torch.isfinite(traj).all().item()}")
    if not torch.isfinite(traj).all():
        print(f"  nan count : {torch.isnan(traj).sum().item()}")
        print(f"  inf count : {torch.isinf(traj).sum().item()}")

    norms = traj.norm(dim=-1)  # (N, L)
    mean_by_layer = norms.mean(dim=0)
    std_by_layer = norms.std(dim=0)
    print(f"\nper-layer ||h|| (mean over {N} prompts):")
    for l in range(L):
        print(f"  layer {l:2d}: mean={mean_by_layer[l]:8.2f}  std={std_by_layer[l]:8.2f}")

    print(f"\nprompt samples:")
    for i, p in enumerate(blob["prompts"][:3]):
        print(f"  [{i}] {p[:80]!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", type=str)
    args = p.parse_args()
    inspect(Path(args.path))


if __name__ == "__main__":
    main()
