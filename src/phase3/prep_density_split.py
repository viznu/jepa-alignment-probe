"""Prepare data for the JEPA-SCORE density experiment.

Splits instructed_pairs.pt by fact_id (80/20). Saves:
  data/_iter3_train_honest.pt : honest-only examples from train facts
  data/_iter3_test.pt         : all examples (honest + deceptive) from test facts

JEPA is trained (self-supervised) on the train-honest file. At scoring time,
JEPA-SCORE is computed per example in the test file, and separation between
label=0 (honest) and label=1 (deceptive) is measured with AUROC.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/instructed_pairs.pt")
    p.add_argument("--train_out", default="data/_iter3_train_honest.pt")
    p.add_argument("--test_out", default="data/_iter3_test.pt")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    traj = blob["trajectories"]
    labels = blob["labels"]
    fact_ids = blob["fact_ids"].numpy()
    N, L, d = traj.shape
    y = labels.numpy()

    idx = np.arange(N)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    tr_idx, te_idx = next(splitter.split(idx, y, fact_ids))

    # Honest-only train
    tr_honest_mask = labels[tr_idx] == 0
    tr_honest_idx = tr_idx[tr_honest_mask.numpy()]
    tr_traj = traj[torch.tensor(tr_honest_idx)]
    tr_fids = torch.tensor(fact_ids[tr_honest_idx])
    tr_prompts = [blob["prompts"][i] for i in tr_honest_idx]
    tr_facts = blob.get("facts", None)

    # Mixed test (both labels)
    te_traj = traj[torch.tensor(te_idx)]
    te_labels = labels[torch.tensor(te_idx)]
    te_fids = torch.tensor(fact_ids[te_idx])
    te_prompts = [blob["prompts"][i] for i in te_idx]

    print(f"original: N={N}  facts={len(np.unique(fact_ids))}")
    print(f"  train pool (all labels):    {len(tr_idx)} examples across {len(np.unique(fact_ids[tr_idx]))} facts")
    print(f"  test (fact-held-out):       {len(te_idx)} examples across {len(np.unique(fact_ids[te_idx]))} facts")
    print(f"  train honest-only (JEPA):   {len(tr_honest_idx)} examples")
    print(f"  test honest/deceptive:      honest={int((te_labels == 0).sum())}  "
          f"deceptive={int((te_labels == 1).sum())}")

    for outp, payload in [
        (args.train_out, {
            "trajectories": tr_traj,
            "prompts": tr_prompts,
            "fact_ids": tr_fids,
            "model_name": blob["model_name"],
            "num_layers": L,
            "hidden_dim": d,
            "token_position": blob.get("token_position", "?"),
            "split": "train_honest",
        }),
        (args.test_out, {
            "trajectories": te_traj,
            "labels": te_labels,
            "fact_ids": te_fids,
            "prompts": te_prompts,
            "model_name": blob["model_name"],
            "num_layers": L,
            "hidden_dim": d,
            "token_position": blob.get("token_position", "?"),
            "split": "test_mixed",
        }),
    ]:
        path = Path(outp)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        print(f"  saved -> {path}  shape={tuple(payload['trajectories'].shape)}")


if __name__ == "__main__":
    main()
