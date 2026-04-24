"""Utilities for pair-aware trajectory transforms.

These transforms are for datasets like Instructed-Pairs where each fact_id
has exactly one honest example (label 0) and one deceptive example (label 1).
They remove the fact-specific common-mode component so downstream models see
the perturbation more directly.
"""
from __future__ import annotations

import torch


def _index_pairs(
    labels: torch.Tensor,
    fact_ids: torch.Tensor,
    *,
    require_complete: bool = True,
) -> dict[int, tuple[int, int]]:
    grouped: dict[int, dict[int, int]] = {}
    for idx, (label, fact_id) in enumerate(zip(labels.tolist(), fact_ids.tolist())):
        by_label = grouped.setdefault(int(fact_id), {})
        by_label[int(label)] = idx

    pairs: dict[int, tuple[int, int]] = {}
    for fact_id, by_label in grouped.items():
        if 0 not in by_label or 1 not in by_label:
            if require_complete:
                raise ValueError(
                    f"pair transform requires one honest and one deceptive example per fact_id; "
                    f"fact_id={fact_id} has labels={sorted(by_label)}"
                )
            continue
        pairs[fact_id] = (by_label[0], by_label[1])
    return pairs


def get_paired_indices(
    labels: torch.Tensor,
    fact_ids: torch.Tensor,
    *,
    require_complete: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return aligned honest/deceptive indices for exact pairs."""
    pairs = _index_pairs(labels, fact_ids, require_complete=require_complete)
    honest_idx = [pair[0] for pair in pairs.values()]
    deceptive_idx = [pair[1] for pair in pairs.values()]
    return torch.tensor(honest_idx, dtype=torch.long), torch.tensor(deceptive_idx, dtype=torch.long)


def apply_pair_transform(
    trajectories: torch.Tensor,
    labels: torch.Tensor | None,
    fact_ids: torch.Tensor | None,
    mode: str = "none",
) -> torch.Tensor:
    """Apply a pair-aware transform while preserving sample order.

    Modes:
      - none: return input unchanged
      - pair_residualize: subtract the within-fact pair mean from each example
      - pair_signed_delta: replace each example with +/- half the within-fact
        honest/deceptive difference, preserving the original label ordering
    """
    if mode == "none":
        return trajectories
    if labels is None or fact_ids is None:
        raise ValueError(f"trajectory transform '{mode}' requires labels and fact_ids")
    if not torch.is_tensor(labels) or not torch.is_tensor(fact_ids):
        raise TypeError("labels and fact_ids must be tensors for pair transforms")

    pairs = _index_pairs(labels, fact_ids)
    out = torch.empty_like(trajectories)

    for honest_idx, deceptive_idx in pairs.values():
        honest = trajectories[honest_idx]
        deceptive = trajectories[deceptive_idx]
        delta = deceptive - honest

        if mode == "pair_residualize":
            center = 0.5 * (honest + deceptive)
            out[honest_idx] = honest - center
            out[deceptive_idx] = deceptive - center
        elif mode == "pair_signed_delta":
            out[honest_idx] = -0.5 * delta
            out[deceptive_idx] = 0.5 * delta
        else:
            raise ValueError(f"unknown trajectory transform: {mode}")

    return out
