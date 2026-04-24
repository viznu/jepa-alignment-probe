"""Build a diverse-corpus JEPA training file.

Combines:
  - The 80% fact-grouped train subset of instructed_pairs.pt (true facts,
    244 facts x 2 conditions = 488 trajectories; matches the test split
    already used for probe evaluation).
  - All 612 false-fact trajectories (new diagnostic #2 extraction).

Saves a single file usable by src.phase2.train. fact_ids are offset so
true-fact and false-fact ids never collide.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ip_true", default="data/instructed_pairs.pt")
    p.add_argument("--ip_false", default="data/instructed_pairs_false.pt")
    p.add_argument("--output", default="data/diverse_corpus.pt")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    true_blob = torch.load(args.ip_true, map_location="cpu", weights_only=False)
    true_traj = true_blob["trajectories"]
    true_labels = true_blob["labels"].numpy()
    true_fids = true_blob["fact_ids"].numpy()
    print(f"true:  {tuple(true_traj.shape)}  facts={len(np.unique(true_fids))}")

    # Reproduce the existing fact-grouped split to isolate train from test.
    idx = np.arange(len(true_traj))
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    tr_idx, te_idx = next(splitter.split(idx, true_labels, true_fids))
    print(f"  using train subset: {len(tr_idx)} / test held out: {len(te_idx)}")

    true_tr = true_traj[torch.tensor(tr_idx)]
    true_tr_fids = torch.tensor(true_fids[tr_idx])
    true_tr_labels = torch.tensor(true_labels[tr_idx], dtype=torch.long)

    false_blob = torch.load(args.ip_false, map_location="cpu", weights_only=False)
    false_traj = false_blob["trajectories"]
    false_labels = false_blob["labels"]
    false_fids = false_blob["fact_ids"]
    print(f"false: {tuple(false_traj.shape)}  facts={len(torch.unique(false_fids))}")

    # Offset false fact_ids so they don't collide with true ones.
    offset = int(true_fids.max()) + 1
    false_fids_shifted = false_fids + offset

    combined_traj = torch.cat([true_tr, false_traj], dim=0)
    combined_labels = torch.cat([true_tr_labels, false_labels], dim=0)
    combined_fids = torch.cat([true_tr_fids, false_fids_shifted], dim=0)
    combined_prompts = (
        [true_blob["prompts"][i] for i in tr_idx]
        + false_blob["prompts"]
    )
    combined_source = ["true"] * len(true_tr) + ["false"] * len(false_traj)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "trajectories": combined_traj,
        "labels": combined_labels,
        "fact_ids": combined_fids,
        "prompts": combined_prompts,
        "sources": combined_source,
        "model_name": true_blob["model_name"],
        "num_layers": combined_traj.size(1),
        "hidden_dim": combined_traj.size(2),
        "token_position": true_blob.get("token_position", "last_prefilled_token"),
        "notes": "Diverse corpus: 80%-train of IP-true + all IP-false (fact_ids offset)",
    }, out)
    print(f"saved combined shape={tuple(combined_traj.shape)} -> {out}")


if __name__ == "__main__":
    main()
