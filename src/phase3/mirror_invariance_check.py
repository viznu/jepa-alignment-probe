"""Empirical verification of the mirror-invariance lemma for JEPA-SCORE.

Lemma (informal): For an approximately locally-linear encoder f,
    sigma_k(J_f(+x)) ≈ sigma_k(J_f(-x))   up to O(||x||^2)
and therefore
    JEPA-SCORE(+x) ≈ JEPA-SCORE(-x).

This script measures the singular-value agreement numerically on the
pair-residualized test set, where within-pair trajectories are exact
+/- mirrors of one another.

For each of the 62 test pairs:
  1. Compute Jacobian J(x_honest) = d f(x_honest) / d x
  2. Compute Jacobian J(x_deceptive) where x_deceptive = -x_honest
  3. Compare singular value vectors:
       - relative L2 norm of difference
       - max elementwise |delta sigma|
       - Pearson correlation
  4. Also compare the scalar JEPA-SCORE values directly.

Reports mean/median/max across all pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.autograd.functional import jacobian
from tqdm import tqdm

from src.phase2.model import LayerJEPA
from src.phase3.pair_transforms import apply_pair_transform


def load_jepa(ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = LayerJEPA(
        d_in=ckpt["d_in"], d_model=cfg["d_model"],
        num_layers=cfg["jepa_layers"], num_heads=cfg["heads"],
        max_L=max(ckpt["L"], 64),
    ).to(device)
    model.load_state_dict(ckpt["online_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt["norm_mean"].to(device), ckpt["norm_std"].to(device)


def singular_values(encoder: LayerJEPA, x: torch.Tensor, pool: str, layer_idx: int | None) -> torch.Tensor:
    def enc_pool(xi: torch.Tensor) -> torch.Tensor:
        z = encoder.encode(xi)
        if pool == "mean":
            return z.mean(dim=1)
        if pool == "layer":
            return z[:, layer_idx, :]
        raise ValueError(pool)

    J = jacobian(enc_pool, x)  # (1, d_model, 1, L, d_in)
    J_flat = J.view(J.size(1), -1)
    return torch.linalg.svdvals(J_flat)  # (d_model,)


def compare_pair(encoder: LayerJEPA, x_plus: torch.Tensor, x_minus: torch.Tensor,
                 pool: str, layer_idx: int | None) -> dict:
    sv_plus = singular_values(encoder, x_plus, pool, layer_idx)
    sv_minus = singular_values(encoder, x_minus, pool, layer_idx)

    diff = sv_plus - sv_minus
    rel_l2 = (diff.norm() / sv_plus.norm()).item()
    max_abs = diff.abs().max().item()
    ref_max = sv_plus.abs().max().item()
    max_rel = max_abs / max(ref_max, 1e-12)

    svp = sv_plus.cpu().numpy()
    svm = sv_minus.cpu().numpy()
    corr = float(np.corrcoef(svp, svm)[0, 1]) if svp.std() > 0 else 1.0

    eps = 1e-8
    score_plus = float(sv_plus.clamp_min(eps).log().sum().item())
    score_minus = float(sv_minus.clamp_min(eps).log().sum().item())

    return {
        "rel_l2_diff": rel_l2,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "sv_pearson_corr": corr,
        "score_plus": score_plus,
        "score_minus": score_minus,
        "score_abs_gap": abs(score_plus - score_minus),
        "score_rel_gap": abs(score_plus - score_minus) / max(abs(score_plus), 1e-6),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jepa", default="results/jepa_iter3_honest.pt")
    p.add_argument("--data", default="data/_iter3_test_pairresid.pt",
                   help="Pair-residualized test set (paired honest/deceptive are +/- mirrors)")
    p.add_argument("--output", default="results/iter3_mirror_invariance.json")
    p.add_argument("--pool", default="mean", choices=["mean", "layer"])
    p.add_argument("--layer_idx", type=int, default=24)
    p.add_argument("--n_pairs", type=int, default=62)
    args = p.parse_args()

    device = "cpu"  # Jacobian needs autograd-friendly ops
    encoder, norm_mean, norm_std = load_jepa(args.jepa, device=device)

    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    traj = blob["trajectories"]
    labels = blob["labels"]
    fact_ids = blob["fact_ids"]
    print(f"data: N={len(traj)}  facts={len(torch.unique(fact_ids))}")

    # Normalize (same transform JEPA was trained with).
    traj_norm = (traj - norm_mean.cpu()) / norm_std.cpu()

    # Build pair mapping.
    pairs: dict[int, dict[int, int]] = {}
    for idx in range(len(traj)):
        fid = int(fact_ids[idx].item())
        lab = int(labels[idx].item())
        pairs.setdefault(fid, {})[lab] = idx
    pair_ids = sorted(pairs.keys())
    if args.n_pairs < len(pair_ids):
        pair_ids = pair_ids[: args.n_pairs]
    print(f"comparing {len(pair_ids)} pairs with pool={args.pool}"
          + (f" layer_idx={args.layer_idx}" if args.pool == "layer" else ""))

    # Sanity: paired residuals should be exact mirrors.
    p0 = pair_ids[0]
    mirror_check = (traj[pairs[p0][0]] + traj[pairs[p0][1]]).abs().mean().item()
    print(f"mirror check (honest + deceptive should sum to ~0): {mirror_check:.6g}")

    per_pair = []
    for fid in tqdm(pair_ids, desc="pairs"):
        i_h = pairs[fid][0]
        i_d = pairs[fid][1]
        x_h = traj_norm[i_h: i_h + 1]
        x_d = traj_norm[i_d: i_d + 1]
        rec = compare_pair(encoder, x_h, x_d, pool=args.pool,
                           layer_idx=args.layer_idx if args.pool == "layer" else None)
        rec["fact_id"] = fid
        per_pair.append(rec)

    def agg(key):
        vals = [r[key] for r in per_pair]
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "max": float(np.max(vals)),
            "min": float(np.min(vals)),
        }

    summary = {
        "jepa": args.jepa,
        "data": args.data,
        "pool": args.pool,
        "layer_idx": args.layer_idx if args.pool == "layer" else None,
        "n_pairs": len(pair_ids),
        "rel_l2_diff": agg("rel_l2_diff"),
        "max_abs_diff": agg("max_abs_diff"),
        "max_rel_diff": agg("max_rel_diff"),
        "sv_pearson_corr": agg("sv_pearson_corr"),
        "score_abs_gap": agg("score_abs_gap"),
        "score_rel_gap": agg("score_rel_gap"),
    }

    print("\n" + "=" * 60)
    print(f"mirror-invariance empirical check  (pool={args.pool})")
    print("-" * 60)
    for metric, stats in [
        ("||sigma(+x) - sigma(-x)||_2 / ||sigma(+x)||_2", summary["rel_l2_diff"]),
        ("max |sigma_k(+x) - sigma_k(-x)| / max sigma", summary["max_rel_diff"]),
        ("pearson corr(sigma(+x), sigma(-x))",           summary["sv_pearson_corr"]),
        ("|JEPA-SCORE(+x) - JEPA-SCORE(-x)| / |score|",  summary["score_rel_gap"]),
    ]:
        print(f"  {metric}")
        print(f"    mean={stats['mean']:.4g}  median={stats['median']:.4g}  "
              f"max={stats['max']:.4g}")
    print("=" * 60)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": summary, "per_pair": per_pair}, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
