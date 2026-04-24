"""Four probe families for misalignment classification from activation trajectories.

Common interface:
  fit(traj_train, y_train, traj_val=None, y_val=None) -> None
  predict_score(traj) -> np.ndarray of shape (N,)  (higher = more "harmful")

traj is torch.Tensor of shape (N, L, d); y is torch.Tensor of shape (N,) with {0, 1}.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _as_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


class SingleLayerProbe:
    """Logistic regression on one chosen layer's activation vector."""
    def __init__(self, layer: int, C: float = 0.1, max_iter: int = 2000):
        self.layer = layer
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=max_iter)

    def fit(self, traj, y, traj_val=None, y_val=None):
        X = _as_np(traj[:, self.layer, :])
        y = _as_np(y)
        self.clf.fit(self.scaler.fit_transform(X), y)

    def predict_score(self, traj):
        X = self.scaler.transform(_as_np(traj[:, self.layer, :]))
        return self.clf.decision_function(X)


class AllLayersConcatProbe:
    """Logistic regression on flattened (L*d,) per-sample concatenation."""
    def __init__(self, C: float = 0.01, max_iter: int = 5000):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=max_iter)

    def fit(self, traj, y, traj_val=None, y_val=None):
        X = _as_np(traj.flatten(1))
        y = _as_np(y)
        self.clf.fit(self.scaler.fit_transform(X), y)

    def predict_score(self, traj):
        X = self.scaler.transform(_as_np(traj.flatten(1)))
        return self.clf.decision_function(X)


class TransformerOverLayersProbe:
    """Small transformer over the L-layer sequence + classification head.
    Supervised, trained with BCE. Layer inputs are z-normalized per layer.
    """
    def __init__(self, d_in: int, d_model: int = 256, n_heads: int = 4, n_layers: int = 4,
                 max_L: int = 64, epochs: int = 30, batch_size: int = 32, lr: float = 1e-3,
                 wd: float = 0.05, device: str = "cpu", seed: int = 42):
        self.device = device
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        torch.manual_seed(seed)
        self.model = _TransformerClassifier(d_in, d_model, n_heads, n_layers, max_L).to(device)
        self.mean = None
        self.std = None

    def fit(self, traj, y, traj_val=None, y_val=None):
        self.mean = traj.mean(dim=0, keepdim=True)
        self.std = traj.std(dim=0, keepdim=True).clamp_min(1e-6)
        X = ((traj - self.mean) / self.std).to(self.device)
        yt = y.float().to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        N = X.size(0)
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(N, device=self.device)
            losses = []
            for i in range(0, N, self.batch_size):
                idx = perm[i:i + self.batch_size]
                logits = self.model(X[idx])
                loss = F.binary_cross_entropy_with_logits(logits, yt[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())

    def predict_score(self, traj):
        self.model.eval()
        X = ((traj - self.mean) / self.std).to(self.device)
        with torch.no_grad():
            logits = self.model(X)
        return logits.detach().cpu().numpy()


class _TransformerClassifier(nn.Module):
    def __init__(self, d_in, d_model, n_heads, n_layers, max_L):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.zeros(max_L, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L, _ = x.shape
        h = self.proj(x) + self.pos[:L]
        z = self.enc(h)
        return self.head(z.mean(dim=1)).squeeze(-1)


class JEPAFrozenProbe:
    """Use a frozen trained JEPA encoder as feature extractor, then logistic on pooled latent."""
    def __init__(self, jepa_ckpt_path: str, device: str = "cpu", C: float = 0.1, max_iter: int = 2000,
                 pool: str = "mean", layer_idx: int | None = None,
                 window_start: int | None = None, window_end: int | None = None):
        from src.phase2.model import LayerJEPA
        ckpt = torch.load(jepa_ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        self.device = device
        self.pool = pool
        self.layer_idx = layer_idx
        self.window_start = window_start
        self.window_end = window_end
        self.encoder = LayerJEPA(
            d_in=ckpt["d_in"], d_model=cfg["d_model"],
            num_layers=cfg["jepa_layers"], num_heads=cfg["heads"],
            max_L=max(ckpt["L"], 64),
        ).to(device)
        self.encoder.load_state_dict(ckpt["online_state_dict"])
        self.encoder.eval()
        self.jepa_mean = ckpt["norm_mean"].to("cpu")
        self.jepa_std = ckpt["norm_std"].to("cpu")
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=max_iter)

    @torch.no_grad()
    def _featurize(self, traj):
        x = ((traj.cpu() - self.jepa_mean) / self.jepa_std).to(self.device)
        z = self.encoder.encode(x)  # (N, L, d_model)
        if self.pool == "mean":
            feat = z.mean(dim=1)
        elif self.pool == "last":
            feat = z[:, -1, :]
        elif self.pool == "mid":
            feat = z[:, z.size(1) // 2, :]
        elif self.pool == "layer":
            if self.layer_idx is None:
                raise ValueError("pool='layer' requires layer_idx")
            feat = z[:, self.layer_idx, :]
        elif self.pool == "window":
            if self.window_start is None or self.window_end is None:
                raise ValueError("pool='window' requires window_start and window_end")
            feat = z[:, self.window_start:self.window_end, :].mean(dim=1)
        else:
            raise ValueError(f"unknown JEPA pooling mode: {self.pool}")
        return feat.cpu().numpy()

    def fit(self, traj, y, traj_val=None, y_val=None):
        X = self._featurize(traj)
        y = _as_np(y)
        self.clf.fit(self.scaler.fit_transform(X), y)

    def predict_score(self, traj):
        X = self.scaler.transform(self._featurize(traj))
        return self.clf.decision_function(X)
