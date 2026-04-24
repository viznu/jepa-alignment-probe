"""Train LayerJEPA on saved activation trajectories.

Loss: MSE between predictor output and EMA-target encoder output, measured
only at masked layer positions. Target encoder is a copy whose weights EMA
the online encoder (no gradient flow through target).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.phase2.model import LayerJEPA, random_block_mask
from src.phase3.pair_transforms import apply_pair_transform, get_paired_indices


def ema_update(target: torch.nn.Module, online: torch.nn.Module, tau: float) -> None:
    for p_t, p in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(tau).add_(p.data, alpha=1.0 - tau)


def compute_loss(online: LayerJEPA, target: LayerJEPA,
                 x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = online(x, mask)  # (B, L, d_model)
    with torch.no_grad():
        tgt = target.encode(x)  # (B, L, d_model), EMA target — no mask
    diff = (pred - tgt)[mask]  # (n_masked, d_model)
    return diff.pow(2).mean()


def pool_latents(
    z: torch.Tensor,
    pool: str = "mean",
    layer_idx: int | None = None,
    window_start: int | None = None,
    window_end: int | None = None,
) -> torch.Tensor:
    if pool == "mean":
        return z.mean(dim=1)
    if pool == "last":
        return z[:, -1, :]
    if pool == "mid":
        return z[:, z.size(1) // 2, :]
    if pool == "layer":
        if layer_idx is None:
            raise ValueError("pool='layer' requires layer_idx")
        return z[:, layer_idx, :]
    if pool == "window":
        if window_start is None or window_end is None:
            raise ValueError("pool='window' requires window_start and window_end")
        return z[:, window_start:window_end, :].mean(dim=1)
    raise ValueError(f"unknown latent pool: {pool}")


def compute_pair_loss(
    online: LayerJEPA,
    x: torch.Tensor,
    honest_idx: torch.Tensor,
    deceptive_idx: torch.Tensor,
    pool: str,
    layer_idx: int | None,
    window_start: int | None,
    window_end: int | None,
    margin: float,
) -> torch.Tensor:
    """Cosine-margin pair loss (Codex variant).

    Penalizes cos(z_honest, z_deceptive) - margin when positive. Weak signal
    in practice; see compute_pair_infonce for the stronger alternative.
    """
    z = online.encode(x)
    pooled = pool_latents(z, pool=pool, layer_idx=layer_idx,
                          window_start=window_start, window_end=window_end)
    honest = pooled[honest_idx]
    deceptive = pooled[deceptive_idx]
    cosine = F.cosine_similarity(honest, deceptive, dim=-1)
    return F.relu(cosine - margin).mean()


def compute_pair_infonce(
    online: LayerJEPA,
    x: torch.Tensor,
    labels: torch.Tensor,
    pool: str,
    layer_idx: int | None,
    window_start: int | None,
    window_end: int | None,
    temperature: float,
) -> torch.Tensor:
    """SupCon-style InfoNCE loss on pooled JEPA latents.

    For each anchor, positives are same-label samples across the batch and
    negatives are different-label samples (includes the within-fact partner,
    which sits geometrically close after vanilla JEPA training — the hardest
    negative). L2-normalized embeddings; standard log-softmax over similarity.
    """
    z = online.encode(x)
    pooled = pool_latents(z, pool=pool, layer_idx=layer_idx,
                          window_start=window_start, window_end=window_end)
    pooled = F.normalize(pooled, dim=-1)

    B = pooled.size(0)
    sim = pooled @ pooled.T / temperature  # (B, B)

    self_mask = torch.eye(B, dtype=torch.bool, device=sim.device)
    # Use a large negative (not -inf) for self-similarity: -inf interacts with
    # the bool mask multiply below to produce NaN (-inf * 0 = NaN). -1e9 is
    # effectively 0 under softmax but avoids the NaN.
    sim = sim.masked_fill(self_mask, -1e9)

    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = same_label & ~self_mask

    # log p(positive) under softmax over all non-self samples
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n_pos = pos_mask.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / n_pos

    # Only include anchors that actually have a positive in this batch.
    valid = pos_mask.any(dim=1)
    if not valid.any():
        return torch.zeros((), device=sim.device)
    return -mean_log_prob_pos[valid].mean()


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    traj = blob["trajectories"]  # (N, L, d)
    traj = apply_pair_transform(
        traj,
        blob.get("labels"),
        blob.get("fact_ids"),
        mode=args.trajectory_transform,
    )
    N, L, d = traj.shape
    print(f"loaded {args.data}: N={N} L={L} d={d} transform={args.trajectory_transform}")

    # Z-normalize per layer across the dataset — stabilizes training given
    # the 30x norm growth from layer 0 to layer 34.
    mean = traj.mean(dim=0, keepdim=True)
    std = traj.std(dim=0, keepdim=True).clamp_min(1e-6)
    traj_norm = (traj - mean) / std

    n_val = max(1, int(N * args.val_frac))
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr = traj_norm[tr_idx]
    vl = traj_norm[val_idx]
    labels = blob.get("labels")
    fact_ids = blob.get("fact_ids")
    pair_train = None
    if args.pair_loss_weight > 0.0:
        if labels is None:
            raise ValueError("pair-aware loss requires labels in the dataset")
        tr_labels = labels[tr_idx]
        pair_train = {
            "x": tr.to(device),
            "labels": tr_labels.to(device),
        }
        if args.pair_loss_kind == "cosine":
            if fact_ids is None:
                raise ValueError("pair_loss_kind=cosine requires fact_ids")
            tr_fact_ids = fact_ids[tr_idx]
            honest_idx, deceptive_idx = get_paired_indices(tr_labels, tr_fact_ids)
            pair_train["honest_idx"] = honest_idx.to(device)
            pair_train["deceptive_idx"] = deceptive_idx.to(device)
    print(f"train={len(tr)} val={len(vl)}")

    loader = DataLoader(TensorDataset(tr), batch_size=args.batch_size, shuffle=True, drop_last=False)

    online = LayerJEPA(d_in=d, d_model=args.d_model, num_layers=args.jepa_layers,
                       num_heads=args.heads, max_L=max(L, 64)).to(device)
    target = copy.deepcopy(online).to(device)
    for p in target.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in online.parameters())
    print(f"params={n_params:,}  d_model={args.d_model}  jepa_layers={args.jepa_layers}")

    opt = torch.optim.AdamW(online.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log = []
    for epoch in range(args.epochs):
        t0 = time.time()
        online.train()
        tr_losses = []
        pair_losses = []
        for (x,) in loader:
            x = x.to(device)
            mask = random_block_mask(x.size(0), L, ratio=args.mask_ratio, device=device)
            loss = compute_loss(online, target, x, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
            opt.step()
            ema_update(target, online, tau=args.ema_tau)
            tr_losses.append(loss.item())
        if pair_train is not None:
            if args.pair_loss_kind == "infonce":
                pair_loss = compute_pair_infonce(
                    online,
                    pair_train["x"],
                    pair_train["labels"],
                    pool=args.pair_pool,
                    layer_idx=args.pair_layer_idx,
                    window_start=args.pair_window_start,
                    window_end=args.pair_window_end,
                    temperature=args.pair_temperature,
                )
            else:
                pair_loss = compute_pair_loss(
                    online,
                    pair_train["x"],
                    pair_train["honest_idx"],
                    pair_train["deceptive_idx"],
                    pool=args.pair_pool,
                    layer_idx=args.pair_layer_idx,
                    window_start=args.pair_window_start,
                    window_end=args.pair_window_end,
                    margin=args.pair_margin,
                )
            total_pair_loss = args.pair_loss_weight * pair_loss
            opt.zero_grad()
            total_pair_loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
            opt.step()
            ema_update(target, online, tau=args.ema_tau)
            pair_losses.append(pair_loss.item())
        sched.step()

        online.eval()
        with torch.no_grad():
            xv = vl.to(device)
            mv = random_block_mask(xv.size(0), L, ratio=args.mask_ratio, device=device)
            val_loss = compute_loss(online, target, xv, mv).item()

        tr_loss = sum(tr_losses) / len(tr_losses)
        lr_now = opt.param_groups[0]["lr"]
        dt = time.time() - t0
        pair_loss_mean = sum(pair_losses) / len(pair_losses) if pair_losses else None
        record = {"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, "lr": lr_now, "time": dt}
        if pair_loss_mean is not None:
            record["pair_loss"] = pair_loss_mean
        log.append(record)
        pair_msg = f"  pair={pair_loss_mean:.4f}" if pair_loss_mean is not None else ""
        print(f"epoch {epoch:3d}  train={tr_loss:.4f}  val={val_loss:.4f}{pair_msg}  lr={lr_now:.2e}  ({dt:.1f}s)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "online_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "config": vars(args),
        "log": log,
        "norm_mean": mean,
        "norm_std": std,
        "d_in": d,
        "L": L,
    }, out)
    with open(out.with_suffix(".log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/activations_iter1_smoke.pt")
    p.add_argument("--output", default="results/jepa_iter1.pt")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--jepa_layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mask_ratio", type=float, default=0.4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--ema_tau", type=float, default=0.996)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trajectory_transform", default="none",
                   choices=["none", "pair_residualize", "pair_signed_delta"])
    p.add_argument("--pair_loss_weight", type=float, default=0.0)
    p.add_argument("--pair_loss_kind", default="cosine",
                   choices=["cosine", "infonce"],
                   help="cosine=Codex's cosine-margin; infonce=SupCon-style on pooled latents.")
    p.add_argument("--pair_margin", type=float, default=0.0,
                   help="cosine only: margin above which cos(honest, deceptive) is penalized.")
    p.add_argument("--pair_temperature", type=float, default=0.07,
                   help="infonce only: softmax temperature.")
    p.add_argument("--pair_pool", default="layer",
                   choices=["mean", "last", "mid", "layer", "window"])
    p.add_argument("--pair_layer_idx", type=int, default=24)
    p.add_argument("--pair_window_start", type=int, default=None)
    p.add_argument("--pair_window_end", type=int, default=None)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
