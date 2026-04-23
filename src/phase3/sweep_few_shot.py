"""Few-shot sweep: vary N_train for each probe, compare AUROC.

JEPA's hypothetical selling point is that unlabeled pretraining helps in the
low-label regime. If that's real, the JEPAFrozenProbe curve dominates the
others at small N_train even when they converge at large N_train.

For each (N_train, probe, seed): subsample N_train labeled examples
stratified by class, fit probe, score fixed test set. Report mean +/- std
across seeds.

Note: SingleLayerProbe uses a FIXED layer (best from the full-data CV) —
running 5-fold CV with N_train=5 is degenerate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.phase3.probes import (
    SingleLayerProbe, AllLayersConcatProbe, TransformerOverLayersProbe, JEPAFrozenProbe,
)


def stratified_subsample(idx_pool: np.ndarray, labels: np.ndarray,
                         n: int, rng: np.random.Generator) -> np.ndarray:
    pos = idx_pool[labels[idx_pool] == 1]
    neg = idx_pool[labels[idx_pool] == 0]
    n_pos = n // 2
    n_neg = n - n_pos
    pick_pos = rng.choice(pos, size=min(n_pos, len(pos)), replace=False)
    pick_neg = rng.choice(neg, size=min(n_neg, len(neg)), replace=False)
    return np.concatenate([pick_pos, pick_neg])


def build_probes(d: int, L: int, jepa_ckpt: str, single_layer: int, device: str,
                 transformer_epochs: int) -> dict:
    return {
        "single_layer": lambda: SingleLayerProbe(layer=single_layer),
        "all_layers_concat": lambda: AllLayersConcatProbe(),
        "transformer_over_layers": lambda: TransformerOverLayersProbe(
            d_in=d, epochs=transformer_epochs, device=device),
        "jepa_frozen": lambda: JEPAFrozenProbe(jepa_ckpt, device=device),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/_labeled_smoke200.pt")
    p.add_argument("--jepa", default="results/_jepa_smoke50.pt")
    p.add_argument("--output", default="results/iter2_few_shot.json")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--single_layer", type=int, default=0)  # best from full-data CV
    p.add_argument("--transformer_epochs", type=int, default=30)
    p.add_argument("--n_trains", type=int, nargs="+",
                   default=[5, 10, 20, 40, 80, 160])
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    traj = blob["trajectories"]
    labels = blob["labels"]
    N, L, d = traj.shape
    y_np = labels.numpy()
    print(f"N={N} L={L} d={d}  pos={int(labels.sum())}  neg={int((1-labels).sum())}")

    idx = np.arange(N)
    tr_pool, te_idx = train_test_split(
        idx, test_size=args.test_frac, stratify=y_np, random_state=42,
    )
    te_idx_t = torch.tensor(te_idx)
    Xte = traj[te_idx_t]
    yte = labels[te_idx_t]
    print(f"train_pool={len(tr_pool)}  test={len(te_idx)}")

    probe_factories = build_probes(d, L, args.jepa, args.single_layer, device, args.transformer_epochs)

    results: dict = {p: {n: [] for n in args.n_trains} for p in probe_factories}

    for n_train in args.n_trains:
        if n_train > len(tr_pool):
            print(f"  (skipping n_train={n_train}: only {len(tr_pool)} in pool)")
            continue
        print(f"\nN_train={n_train}")
        for seed in range(args.n_seeds):
            rng = np.random.default_rng(seed)
            sub = stratified_subsample(tr_pool, y_np, n_train, rng)
            Xtr = traj[torch.tensor(sub)]
            ytr = labels[torch.tensor(sub)]
            for pname, pfactory in probe_factories.items():
                probe = pfactory()
                probe.fit(Xtr, ytr)
                s = probe.predict_score(Xte)
                auc = float(roc_auc_score(yte.numpy(), s))
                results[pname][n_train].append(auc)
            print(f"  seed={seed}:  "
                  + "  ".join(f"{p}={results[p][n_train][-1]:.3f}" for p in probe_factories))

    # Aggregate
    print("\n" + "=" * 78)
    print(f"{'probe':<28s}" + "".join(f"{n:>8d}" for n in args.n_trains))
    print("-" * 78)
    for pname in probe_factories:
        row = f"{pname:<28s}"
        for n in args.n_trains:
            xs = results[pname][n]
            if xs:
                row += f" {np.mean(xs):.2f}±{np.std(xs):.2f}"
            else:
                row += f"{'--':>8s}"
        print(row)
    print("=" * 78)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "data": args.data, "jepa": args.jepa,
            "n_trains": args.n_trains, "n_seeds": args.n_seeds,
            "single_layer": args.single_layer,
            "N_test": len(te_idx),
            "results": results,
        }, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
