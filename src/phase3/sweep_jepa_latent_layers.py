"""Sweep linear probes over JEPA latent slots.

This answers a focused question: if we freeze the JEPA encoder and probe each
latent layer slot separately, where does the discriminative signal peak?

That helps distinguish:
  - representation failure: JEPA never encoded the signal
  - readout failure: JEPA encoded the signal, but global pooling diluted it
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split

from src.phase3.pair_transforms import apply_pair_transform
from src.phase3.probes import JEPAFrozenProbe


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = fpr <= fpr_target
    if mask.any():
        return float(tpr[mask].max())
    return 0.0


def cross_val_auc_for_layer(
    traj_train: torch.Tensor,
    y_train: torch.Tensor,
    jepa_ckpt: str,
    layer_idx: int,
    device: str,
    cv: int,
) -> float:
    y = y_train.cpu().numpy()
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    aucs: list[float] = []
    for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
        probe = JEPAFrozenProbe(jepa_ckpt, device=device, pool="layer", layer_idx=layer_idx)
        probe.fit(traj_train[tr_idx], y_train[tr_idx])
        scores = probe.predict_score(traj_train[va_idx])
        aucs.append(float(roc_auc_score(y_train[va_idx].cpu().numpy(), scores)))
    return float(np.mean(aucs))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--jepa", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--group_field", default=None,
                   help="If set (e.g. 'fact_ids'), split by group so paired examples stay together.")
    p.add_argument("--trajectory_transform", default="none",
                   choices=["none", "pair_residualize", "pair_signed_delta"])
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    traj = apply_pair_transform(
        blob["trajectories"],
        blob.get("labels"),
        blob.get("fact_ids"),
        mode=args.trajectory_transform,
    )
    labels = blob["labels"]
    N, L, d = traj.shape
    print(f"loaded {args.data}: N={N} L={L} d={d} transform={args.trajectory_transform}")

    idx = np.arange(N)
    if args.group_field and args.group_field in blob:
        groups = blob[args.group_field].numpy() if torch.is_tensor(blob[args.group_field]) else np.asarray(blob[args.group_field])
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
        tr_idx, te_idx = next(splitter.split(idx, labels.numpy(), groups))
        print(f"group-aware split by '{args.group_field}' ({len(np.unique(groups))} groups)")
    else:
        tr_idx, te_idx = train_test_split(
            idx, test_size=args.test_frac, stratify=labels.numpy(), random_state=args.seed,
        )

    tr_idx_t = torch.tensor(tr_idx)
    te_idx_t = torch.tensor(te_idx)
    Xtr, ytr = traj[tr_idx_t], labels[tr_idx_t]
    Xte, yte = traj[te_idx_t], labels[te_idx_t]
    print(f"train={len(tr_idx)} test={len(te_idx)}")

    cv_auc_per_layer: list[float] = []
    test_auc_per_layer: list[float] = []
    test_recall_per_layer: list[float] = []

    for layer_idx in range(L):
        print(f"[{layer_idx + 1}/{L}] JEPA latent layer {layer_idx}")
        cv_auc = cross_val_auc_for_layer(Xtr, ytr, args.jepa, layer_idx, device, args.cv)
        probe = JEPAFrozenProbe(args.jepa, device=device, pool="layer", layer_idx=layer_idx)
        probe.fit(Xtr, ytr)
        scores = probe.predict_score(Xte)
        test_auc = float(roc_auc_score(yte.numpy(), scores))
        test_recall = recall_at_fpr(yte.numpy(), scores, 0.01)
        cv_auc_per_layer.append(cv_auc)
        test_auc_per_layer.append(test_auc)
        test_recall_per_layer.append(test_recall)
        print(f"  cv_auc={cv_auc:.4f} test_auc={test_auc:.4f} recall@1%FPR={test_recall:.4f}")

    best_cv_layer = int(np.argmax(cv_auc_per_layer))
    best_test_layer = int(np.argmax(test_auc_per_layer))

    print("\n" + "=" * 72)
    print(f"best CV layer:   {best_cv_layer}  auc={cv_auc_per_layer[best_cv_layer]:.4f}")
    print(f"best test layer: {best_test_layer}  auc={test_auc_per_layer[best_test_layer]:.4f}")
    print("=" * 72)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "data": args.data,
            "jepa": args.jepa,
            "trajectory_transform": args.trajectory_transform,
            "N_train": len(tr_idx),
            "N_test": len(te_idx),
            "L": L,
            "d": d,
            "cv_auc_per_layer": cv_auc_per_layer,
            "test_auc_per_layer": test_auc_per_layer,
            "test_recall_at_1pct_fpr_per_layer": test_recall_per_layer,
            "best_cv_layer": best_cv_layer,
            "best_test_layer": best_test_layer,
        }, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
