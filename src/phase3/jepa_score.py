"""JEPA-SCORE: Jacobian-based density estimator (Balestriero 2025 thm 1 / eq 5).

For a trained JEPA encoder f, the score
    S(x) = sum_k log sigma_k(J_f(x))
is an estimate of the log data density p_X(x) up to a constant. Samples
from the pretraining distribution score high; out-of-distribution samples
score low.

Here we use it for alignment anomaly detection: train JEPA on aligned
trajectories only, then score held-out aligned vs misaligned. The hypothesis
is that misaligned trajectories score lower even though no misaligned
example was seen during JEPA training.

This is the *unsupervised* angle — no alignment labels used during JEPA
pretraining; labels are only used to evaluate the scoring on held-out data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from torch.autograd.functional import jacobian
from tqdm import tqdm

from src.phase2.model import LayerJEPA


def load_jepa(ckpt_path: str, device: str = "cpu") -> tuple[LayerJEPA, torch.Tensor, torch.Tensor, int, int]:
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
    return model, ckpt["norm_mean"].to(device), ckpt["norm_std"].to(device), ckpt["d_in"], ckpt["L"]


def jepa_score_one(encoder: LayerJEPA, x: torch.Tensor, pool: str = "mean") -> float:
    """JEPA-SCORE for a single trajectory x of shape (1, L, d_in).

    Implements eq 5: sum_k log sigma_k(J_f(x)) where J is the Jacobian of
    the pooled encoder output w.r.t. x.
    """
    def enc_pool(xi: torch.Tensor) -> torch.Tensor:
        z = encoder.encode(xi)  # (1, L, d_model)
        if pool == "mean":
            return z.mean(dim=1)  # (1, d_model)
        raise ValueError(pool)

    J = jacobian(enc_pool, x)  # (1, d_model, 1, L, d_in)
    J_flat = J.view(J.size(1), -1)  # (d_model, L*d_in)
    svd = torch.linalg.svdvals(J_flat)
    eps = 1e-8
    return svd.clamp_min(eps).log().sum().item()


def jepa_score_batch(encoder: LayerJEPA, X: torch.Tensor, pool: str = "mean",
                     desc: str = "jepa-score") -> np.ndarray:
    scores = np.zeros(X.size(0), dtype=np.float64)
    for i in tqdm(range(X.size(0)), desc=desc):
        scores[i] = jepa_score_one(encoder, X[i:i + 1], pool=pool)
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--jepa", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--group_field", default="fact_ids")
    p.add_argument("--pool", default="mean")
    args = p.parse_args()

    device = "cpu"  # Jacobian on MPS hits unsupported ops; CPU is fine for small batches
    encoder, norm_mean, norm_std, d_in, L_train = load_jepa(args.jepa, device=device)
    print(f"JEPA loaded: d_in={d_in}  L={L_train}")

    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    traj = blob["trajectories"]
    labels = blob["labels"]
    N, L, d = traj.shape
    print(f"data: N={N} L={L} d={d}  pos={int(labels.sum())} neg={int((1 - labels).sum())}")

    X = ((traj - norm_mean.cpu()) / norm_std.cpu()).to(device)

    scores = jepa_score_batch(encoder, X, pool=args.pool)

    y = labels.numpy()
    # Convention: label 1 = misaligned. JEPA-SCORE is high for in-distribution,
    # so misaligned should score LOWER. For AUROC over "misaligned" class,
    # flip sign so larger score => more misaligned.
    y_scores = -scores
    auc = float(roc_auc_score(y, y_scores))
    fpr, tpr, _ = roc_curve(y, y_scores)
    mask = fpr <= 0.01
    recall_1pct = float(tpr[mask].max()) if mask.any() else 0.0

    print(f"\nJEPA-SCORE distribution:")
    for name, sel in [("label=0 (honest)", y == 0), ("label=1 (deceptive)", y == 1)]:
        s = scores[sel]
        print(f"  {name}:  mean={s.mean():.3f}  std={s.std():.3f}  "
              f"min={s.min():.3f}  max={s.max():.3f}")

    print(f"\nAUROC (label 1 = misaligned vs -JEPA-SCORE):  {auc:.4f}")
    print(f"recall@1%FPR:                                {recall_1pct:.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "data": args.data, "jepa": args.jepa, "pool": args.pool,
        "N": int(N),
        "jepa_score": scores.tolist(),
        "labels": y.tolist(),
        "auroc_misaligned_vs_neg_score": auc,
        "recall_at_1pct_fpr": recall_1pct,
        "mean_aligned": float(scores[y == 0].mean()),
        "mean_misaligned": float(scores[y == 1].mean()),
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
