"""Zero-shot transfer test: train linear probe on IP-true, test on IP-false.

If the InfoNCE JEPA encoder has learned a content-invariant behavioral-state
latent (honest vs deceptive), a probe fit on one fact distribution should
classify correctly on a different fact distribution without refitting.

Method:
  1. Fit JEPAFrozenProbe on the IP-true fact-grouped training subset
     (488 samples from 244 facts, same split used throughout the project).
  2. Encode ALL IP-false trajectories (612 samples from 306 different facts)
     with the SAME JEPA encoder and apply the SAME probe.
  3. Report AUROC + recall@1%FPR on IP-false.

Comparators (same probe):
  - Performance on IP-true test (the 124 held-out IP-true trajectories).
    This is the in-distribution reference.
  - Difference between in-distribution and transfer is the generalization gap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

from src.phase3.probes import JEPAFrozenProbe


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = fpr <= fpr_target
    return float(tpr[mask].max()) if mask.any() else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jepa", default="results/jepa_ip80_infonce_w10.pt")
    p.add_argument("--src", default="data/instructed_pairs.pt",
                   help="IP-true (probe fit + in-distribution test).")
    p.add_argument("--tgt", default="data/instructed_pairs_false.pt",
                   help="IP-false (zero-shot transfer target).")
    p.add_argument("--output", default="results/iter3_transfer_ip_true_to_false.json")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jepa_pool", default="mean",
                   choices=["mean", "last", "mid", "layer", "window"])
    p.add_argument("--jepa_layer_idx", type=int, default=24)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load source (IP-true) and split into train / in-dist test by fact_id.
    src = torch.load(args.src, map_location="cpu", weights_only=False)
    src_traj = src["trajectories"]
    src_labels = src["labels"]
    src_fids = src["fact_ids"].numpy()
    N_src = len(src_traj)
    idx = np.arange(N_src)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    tr_idx, te_idx = next(splitter.split(idx, src_labels.numpy(), src_fids))
    print(f"IP-true: N={N_src}  train={len(tr_idx)}  indist_test={len(te_idx)}")

    # Load target (IP-false). No split — every sample is a transfer target.
    tgt = torch.load(args.tgt, map_location="cpu", weights_only=False)
    tgt_traj = tgt["trajectories"]
    tgt_labels = tgt["labels"]
    print(f"IP-false: N={len(tgt_traj)}  (all used for transfer)")

    # Fit probe on IP-true train only.
    probe = JEPAFrozenProbe(
        args.jepa, device=device, pool=args.jepa_pool, layer_idx=args.jepa_layer_idx
    )
    Xtr = src_traj[torch.tensor(tr_idx)]
    ytr = src_labels[torch.tensor(tr_idx)]
    probe.fit(Xtr, ytr)
    print(f"probe fit on {len(tr_idx)} IP-true train samples (pool={args.jepa_pool}, "
          f"layer_idx={args.jepa_layer_idx})")

    # In-distribution: IP-true test.
    Xte_id = src_traj[torch.tensor(te_idx)]
    yte_id = src_labels[torch.tensor(te_idx)].numpy()
    s_id = probe.predict_score(Xte_id)
    auc_id = float(roc_auc_score(yte_id, s_id))
    rec_id = recall_at_fpr(yte_id, s_id, 0.01)

    # Transfer: IP-false, all samples.
    yte_tr = tgt_labels.numpy()
    s_tr = probe.predict_score(tgt_traj)
    auc_tr = float(roc_auc_score(yte_tr, s_tr))
    rec_tr = recall_at_fpr(yte_tr, s_tr, 0.01)

    gap = auc_id - auc_tr

    print("\n" + "=" * 60)
    print(f"{'setting':<32s} {'AUROC':>8s} {'recall@1%FPR':>14s}")
    print("-" * 60)
    print(f"{'IP-true test (in-dist)':<32s} {auc_id:>8.4f} {rec_id:>14.4f}")
    print(f"{'IP-false (zero-shot transfer)':<32s} {auc_tr:>8.4f} {rec_tr:>14.4f}")
    print("-" * 60)
    print(f"{'generalization gap (AUROC)':<32s} {gap:>8.4f}")
    print("=" * 60)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "jepa": args.jepa,
            "src": args.src,
            "tgt": args.tgt,
            "pool": args.jepa_pool,
            "layer_idx": args.jepa_layer_idx,
            "N_src_train": int(len(tr_idx)),
            "N_src_test": int(len(te_idx)),
            "N_tgt": int(len(tgt_traj)),
            "indist": {"auroc": auc_id, "recall_at_1pct_fpr": rec_id},
            "transfer": {"auroc": auc_tr, "recall_at_1pct_fpr": rec_tr},
            "gap_auroc": gap,
        }, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
