"""Diagnostic #1: per-slot AUROC sweep on JEPA encoder output.

For a trained JEPA encoder, compute z = encoder.encode(x) of shape (N, L, d_model).
For each layer slot l in 0..L-1, fit L2 logistic regression on z[:, l, :] as
feature and report test AUROC on a fact-grouped train/test split.

This mirrors the single_layer probe sweep on the target-model residual stream,
but on JEPA's re-encoded latents. Peak location tells us where in JEPA's latent
space the honest/deceptive perturbation lives.

Hypothesis: peak at slots around 22-28 (mirroring the target-model 2/3-depth
peak at layer 24). If true, this confirms the signal is in the JEPA
representation but specifically at certain slots — mean-pool buries it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.phase2.model import LayerJEPA


@torch.no_grad()
def encode_all(encoder: LayerJEPA, traj: torch.Tensor, norm_mean: torch.Tensor,
               norm_std: torch.Tensor, device: str) -> torch.Tensor:
    x = ((traj - norm_mean) / norm_std).to(device)
    z = encoder.encode(x)  # (N, L, d_model)
    return z.cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/instructed_pairs.pt")
    p.add_argument("--jepa", default="results/jepa_ip80.pt")
    p.add_argument("--group_field", default="fact_ids")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/iter2b_jepa_layer_sweep.json")
    args = p.parse_args()

    device = "cpu"
    ckpt = torch.load(args.jepa, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    encoder = LayerJEPA(
        d_in=ckpt["d_in"], d_model=cfg["d_model"],
        num_layers=cfg["jepa_layers"], num_heads=cfg["heads"],
        max_L=max(ckpt["L"], 64),
    ).to(device)
    encoder.load_state_dict(ckpt["online_state_dict"])
    encoder.eval()
    for par in encoder.parameters():
        par.requires_grad_(False)

    norm_mean = ckpt["norm_mean"].to("cpu")
    norm_std = ckpt["norm_std"].to("cpu")
    print(f"JEPA loaded: d_in={ckpt['d_in']}  L={ckpt['L']}  d_model={cfg['d_model']}")

    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    traj = blob["trajectories"]
    labels = blob["labels"].numpy()
    N, L, _ = traj.shape
    print(f"data: N={N}  L={L}")

    z = encode_all(encoder, traj, norm_mean, norm_std, device)  # (N, L, d_model)
    print(f"JEPA latents: {tuple(z.shape)}")

    idx = np.arange(N)
    groups = blob[args.group_field].numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    tr_idx, te_idx = next(splitter.split(idx, labels, groups))
    ytr = labels[tr_idx]
    yte = labels[te_idx]
    print(f"train={len(tr_idx)}  test={len(te_idx)}")

    per_slot_auc = []
    for slot in range(z.size(1)):
        Xtr = z[tr_idx, slot, :].numpy()
        Xte = z[te_idx, slot, :].numpy()
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(scaler.transform(Xtr), ytr)
        s = clf.decision_function(scaler.transform(Xte))
        auc = float(roc_auc_score(yte, s))
        per_slot_auc.append(auc)

    best = int(np.argmax(per_slot_auc))
    worst = int(np.argmin(per_slot_auc))
    print(f"\nper-slot AUROC (JEPA latent, L2 logistic, test-set):")
    for slot, auc in enumerate(per_slot_auc):
        marker = "  <-- peak" if slot == best else ("  <-- trough" if slot == worst else "")
        print(f"  slot {slot:2d}: {auc:.4f}{marker}")
    print(f"\npeak:   slot {best}  AUROC {per_slot_auc[best]:.4f}")
    print(f"trough: slot {worst}  AUROC {per_slot_auc[worst]:.4f}")
    print(f"mean:   AUROC {np.mean(per_slot_auc):.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "data": args.data,
            "jepa": args.jepa,
            "N": int(N),
            "L": int(L),
            "N_train": len(tr_idx),
            "N_test": len(te_idx),
            "per_slot_auroc": per_slot_auc,
            "peak_slot": best,
            "peak_auroc": per_slot_auc[best],
            "trough_slot": worst,
            "trough_auroc": per_slot_auc[worst],
            "mean_auroc": float(np.mean(per_slot_auc)),
        }, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
