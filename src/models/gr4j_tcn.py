import os
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════


def get_config():
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument(
        "--root",
        type=str,
        default=os.environ.get(
            "WATERSHED_ROOT", str(Path(__file__).resolve().parent.parent.parent)
        ),
    )
    p.add_argument(
        "--precip_idx",
        type=int,
        default=0,
        help="Precipitation feature index (raw mm/day)",
    )
    p.add_argument(
        "--pet_idx", type=int, default=27, help="PET feature index (raw mm/day)"
    )
    p.add_argument("--hidden_dim", type=int, default=96, help="TCN hidden channels")
    p.add_argument(
        "--num_layers", type=int, default=4, help="TCN depth (dilations = 1,2,4,8,...)"
    )
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument(
        "--loss",
        type=str,
        default="bias_huber",
        choices=["huber", "nse", "peak", "bias_huber"],
    )
    p.add_argument("--peak_percentile", type=float, default=85.0)
    p.add_argument("--peak_loss_weight", type=float, default=2.5)
    p.add_argument("--huber_delta", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--run_tag", type=str, default=None)
    p.add_argument(
        "--seq_suffix",
        type=str,
        default="",
        help="Extra suffix on sequence files, e.g. '_hybrid'",
    )
    return p.parse_args()


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Speed: deterministic=False + benchmark=True gives 10-15% speedup
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ═══════════════════════════════════════════════════════════════════════════════
# Losses
# ═══════════════════════════════════════════════════════════════════════════════


class StableNSELoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        var = ((y_true - y_true.mean()) ** 2).sum()
        denom = torch.clamp(var, min=self.eps)
        sq_err = ((y_pred - y_true) ** 2).sum()
        return sq_err / denom


class PeakFocusedLoss(nn.Module):
    def __init__(
        self,
        threshold: float,
        huber_delta: float = 0.15,
        peak_weight: float = 2.5,
    ):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=huber_delta, reduction="none")
        self.register_buffer("threshold", torch.tensor(threshold, dtype=torch.float32))
        self.peak_weight = peak_weight

    def forward(self, y_pred, y_true):
        per_sample = self.huber(y_pred, y_true)
        with torch.no_grad():
            w = torch.ones_like(y_true)
            ramp_lo = self.threshold * 0.85
            ramp_hi = self.threshold
            mid = (y_true >= ramp_lo) & (y_true < ramp_hi)
            w = torch.where(y_true >= ramp_hi, torch.full_like(w, self.peak_weight), w)
            w = torch.where(
                mid,
                1.0
                + (self.peak_weight - 1.0)
                * (y_true - ramp_lo)
                / (ramp_hi - ramp_lo + 1e-8),
                w,
            )
        return (per_sample * w).sum() / (w.sum() + 1e-8)


class BiasPenalizedHuber(nn.Module):
    """
    Huber + explicit volume-bias penalty.
    bias_weight = 0.5 is a good starting point.
    """

    def __init__(self, huber_delta: float = 0.15, bias_weight: float = 0.5):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=huber_delta)
        self.bias_weight = bias_weight

    def forward(self, y_pred, y_true):
        huber = self.huber(y_pred, y_true)
        bias_penalty = torch.abs(y_pred.mean() - y_true.mean())
        return huber + self.bias_weight * bias_penalty


# ═══════════════════════════════════════════════════════════════════════════════
# GR4J Feature Extractor (differentiable, X4 controls routing delay)
# ═══════════════════════════════════════════════════════════════════════════════


class GR4JFeatureExtractor(nn.Module):
    def __init__(self, precip_idx: int = 0, pet_idx: int = 27, eps: float = 1e-6):
        super().__init__()
        self.precip_idx = precip_idx
        self.pet_idx = pet_idx
        self.eps = eps

        # X1: production store capacity [mm]  (softplus > 1)
        # X2: groundwater exchange [mm/day]     (positive for gaining karst springs)
        # X3: routing store capacity [mm]       (softplus > 1)
        # X4: time constant [days]              (softplus > 0.5)
        self.X1 = nn.Parameter(torch.tensor(300.0))
        self.X2 = nn.Parameter(torch.tensor(1.0))  # CHANGED: positive init for karst
        self.X3 = nn.Parameter(torch.tensor(100.0))
        self.X4 = nn.Parameter(torch.tensor(2.2))  # typical GR4J value

    def forward(self, x):
        # x: (B, T, n_feat)
        B, T, n_feat = x.shape
        P = x[:, :, self.precip_idx]  # (B, T)
        E = x[:, :, self.pet_idx]  # (B, T)

        # Constrain physical parameters
        X1 = F.softplus(self.X1) + 1.0
        X2 = self.X2
        X3 = F.softplus(self.X3) + 1.0
        X4 = F.softplus(self.X4) + 0.5

        # X4 controls the fraction of Q9 that enters the routing store (delayed).
        store_frac = 1.0 - torch.exp(-X4 / 0.956)
        store_frac = torch.clamp(store_frac, 0.05, 0.99)

        # Initialize stores
        S = torch.zeros(B, device=x.device, dtype=x.dtype)
        R = torch.zeros(B, device=x.device, dtype=x.dtype)

        features = []
        q_raw = []  # NEW: collect raw GR4J discharge (mm/day)
        for t in range(T):
            P_t = P[:, t]
            E_t = E[:, t]

            Pn = torch.clamp(P_t - E_t, min=0.0)
            En = torch.clamp(E_t - P_t, min=0.0)

            # --- Production store ---
            ratio_s = S / (X1 + self.eps)
            tanh_ps = torch.tanh(Pn / (X1 + self.eps))
            Ps = X1 * (1 - ratio_s**2) * tanh_ps / (1 + ratio_s * tanh_ps + self.eps)
            Ps = torch.clamp(Ps, min=0.0)

            tanh_es = torch.tanh(En / (X1 + self.eps))
            Es = S * (2 - ratio_s) * tanh_es / (1 + (1 - ratio_s) * tanh_es + self.eps)
            Es = torch.minimum(Es, S).clamp(min=0.0)

            S = S + torch.where(P_t >= E_t, Ps, -Es)

            # Percolation
            perc_ratio = S / (5.25 * X1 + self.eps)
            Perc = S * (1.0 - (1.0 + perc_ratio**4 + self.eps) ** (-0.25))
            Perc = torch.minimum(Perc, S).clamp(min=0.0)
            S = S - Perc

            # Routing input
            Pr = torch.where(P_t >= E_t, Pn - Ps, torch.zeros_like(Pn))
            Q9 = Perc + Pr
            Q9 = torch.clamp(Q9, min=0.0)

            # X4-scaled split: store_frac to routing store, rest direct
            Q1 = store_frac * Q9
            Q0 = (1.0 - store_frac) * Q9

            # --- Routing store ---
            ratio_r = R / (X3 + self.eps)
            Fex = X2 * (ratio_r**3.5)
            Fex = torch.clamp(Fex, min=-X3 * 2, max=X3 * 2)

            R_in = R + Q1 + Fex
            R_in = torch.clamp(R_in, min=0.0)

            qr_ratio = R_in / (X3 + self.eps)
            Qr = R_in * (1.0 - (1.0 + qr_ratio**4 + self.eps) ** (-0.25))
            Qr = torch.clamp(Qr, min=0.0)

            R = R_in - Qr
            # Total discharge (mm/day) — standard GR4J formulation
            Q = Qr + Q0 + torch.clamp(-Fex, min=0.0)

            feat = torch.stack(
                [
                    torch.log1p(S),
                    torch.log1p(R),
                    torch.log1p(Ps),
                    torch.log1p(Es),
                    torch.log1p(Perc),
                    torch.log1p(Pr),
                    torch.log1p(Q9),
                    torch.tanh(Fex / 10.0),
                    torch.log1p(Qr),
                    torch.log1p(Q),
                ],
                dim=-1,
            )

            features.append(feat)
            q_raw.append(Q)  # (B)

        features = torch.stack(features, dim=1)  # (B, T, 10)
        q_raw = torch.stack(q_raw, dim=1)  # (B, T)
        return features, q_raw


# ═══════════════════════════════════════════════════════════════════════════════
# TCN Backbone
# ═══════════════════════════════════════════════════════════════════════════════


class TCNLayer(nn.Module):
    """Causal dilated conv with residual."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, padding=pad, dilation=dilation
        )
        self.conv = nn.utils.parametrizations.weight_norm(self.conv)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.chomp = pad

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.out_relu = nn.ReLU()

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv(x)
        if self.chomp > 0:
            out = out[:, :, : -self.chomp]
        out = self.dropout(self.relu(out))

        res = x if self.downsample is None else self.downsample(x)
        if out.shape[-1] != res.shape[-1]:
            min_len = min(out.shape[-1], res.shape[-1])
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]
        return self.out_relu(out + res)


class TCNBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            layers.append(
                TCNLayer(in_ch, hidden_dim, kernel_size, dilation=2**i, dropout=dropout)
            )
            in_ch = hidden_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # x: (B, C, T)
        for layer in self.layers:
            x = layer(x)
        return x  # (B, hidden, T)


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid Model
# ═══════════════════════════════════════════════════════════════════════════════


class HybridGR4J_TCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.30,
        precip_idx: int = 0,
        pet_idx: int = 27,
    ):
        super().__init__()
        self.gr4j = GR4JFeatureExtractor(precip_idx, pet_idx)
        self.tcn = TCNBackbone(
            input_dim + 10, hidden_dim, num_layers, kernel_size, dropout
        )
        self.tcn_norm = nn.LayerNorm(hidden_dim)

        # NEW: project raw GR4J discharge (mean over time) to output space
        self.q_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Head: last timestep + global average pool over TCN output
        head_dim = hidden_dim * 2
        self.head = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x: (B, T, F)
        gr4j_feat, q_raw = self.gr4j(x)  # NEW: unpack q_raw
        x = torch.cat([x, gr4j_feat], dim=-1)  # (B, T, F+10)

        x = x.transpose(1, 2)  # (B, F+10, T)
        h = self.tcn(x)  # (B, hidden, T)
        h = h.transpose(1, 2)  # (B, T, hidden)
        h = self.tcn_norm(h)

        last = h[:, -1, :]  # (B, hidden)
        pool = h.mean(dim=1)  # (B, hidden)
        fused = torch.cat([last, pool], dim=-1)

        neural_out = self.head(fused)  # (B, 1)

        # NEW: physical prior from GR4J raw discharge
        q_mean = q_raw.mean(dim=1, keepdim=True)  # (B, 1)
        q_prior = self.q_proj(q_mean)  # (B, 1)

        return q_prior + neural_out


# ═══════════════════════════════════════════════════════════════════════════════
# EMA
# ═══════════════════════════════════════════════════════════════════════════════


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = deepcopy(model.state_dict())
        for k in self.shadow:
            self.shadow[k] = self.shadow[k].detach().clone()

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def apply_to(self, model):
        return _EMAContext(model, self.shadow)


class _EMAContext:
    def __init__(self, model, shadow):
        self.model = model
        self.shadow = shadow
        self.backup = None

    def __enter__(self):
        self.backup = deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.shadow)
        return self.model

    def __exit__(self, *args):
        self.model.load_state_dict(self.backup)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════


def safe_kge(obs, sim, eps=1e-8):
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    obs_std, sim_std = obs.std(), sim.std()
    obs_mean = obs.mean()
    if obs_std < eps or sim_std < eps or abs(obs_mean) < eps:
        return np.nan, np.nan, np.nan, np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim_std / obs_std
    beta = sim.mean() / obs_mean
    if np.isnan(r):
        return np.nan, np.nan, alpha, beta
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge, r, alpha, beta


def nse(obs, sim):
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom < 1e-12:
        return np.nan
    return 1 - np.sum((obs - sim) ** 2) / denom


def log_nse(obs, sim, eps=1e-3):
    obs = np.maximum(obs, 0) + eps
    sim = np.maximum(sim, 0) + eps
    lo, ls = np.log(obs), np.log(sim)
    denom = np.sum((lo - lo.mean()) ** 2)
    if denom < 1e-12:
        return np.nan
    return 1 - np.sum((lo - ls) ** 2) / denom


# ═══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ═══════════════════════════════════════════════════════════════════════════════


def light_style(ax):
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#333333")
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    ax.title.set_color("#111111")
    ax.xaxis.label.set_color("#444444")
    ax.yaxis.label.set_color("#444444")
    ax.grid(alpha=0.2, color="#cccccc")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    cfg = get_config()
    set_seed(cfg.seed)

    # Auto batch size: larger batches for shorter sequences (h=1/h=3)
    if cfg.batch_size is None:
        cfg.batch_size = 512 if cfg.lookback <= 60 else 256

    tag = cfg.run_tag if cfg.run_tag else f"seed{cfg.seed}"

    ROOT = Path(cfg.root)
    SEQ_DIR = ROOT / "data" / "sequences"
    MODEL_DIR = ROOT / "models"
    LOG_DIR = ROOT / "logs"
    FIG_DIR = ROOT / "results" / "figures"
    MET_DIR = ROOT / "results" / "metrics"
    PRED_DIR = ROOT / "results" / "predictions"

    for d in [
        MODEL_DIR / "trained",
        MODEL_DIR / "checkpoints",
        MODEL_DIR / "configs",
        LOG_DIR,
        FIG_DIR,
        MET_DIR,
        PRED_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(
            f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        # Speed: TF32 on Ampere/Ada (RTX 3050 Ti) — ~2× matmul speedup
        torch.set_float32_matmul_precision("high")
        print("  TF32 matmul precision enabled")

    # ── Load sequences ─────────────────────────────────────────────────────────
    print("\nLoading sequences...")
    suffix = f"_h{cfg.horizon}_lb{cfg.lookback}{cfg.seq_suffix}"
    X_train = np.load(SEQ_DIR / f"X_train{suffix}.npy")
    y_train = np.load(SEQ_DIR / f"y_train{suffix}.npy")
    X_val = np.load(SEQ_DIR / f"X_val{suffix}.npy")
    y_val = np.load(SEQ_DIR / f"y_val{suffix}.npy")
    X_test = np.load(SEQ_DIR / f"X_test{suffix}.npy")
    y_test = np.load(SEQ_DIR / f"y_test{suffix}.npy")
    dates_test = np.load(SEQ_DIR / f"dates_test{suffix}.npy", allow_pickle=True)

    total = len(X_train) + len(X_val) + len(X_test)
    print(
        f"  Seq suffix: '{cfg.seq_suffix}'"
        if cfg.seq_suffix
        else "  Seq suffix: (none)"
    )
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(
        f"  Split:   train={len(X_train)/total*100:.1f}% "
        f"val={len(X_val)/total*100:.1f}% test={len(X_test)/total*100:.1f}%"
    )

    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]

    # ── Sanity check: P and PET must be raw mm/day ─────────────────────────────
    p_mean = float(X_train[:, :, cfg.precip_idx].mean())
    e_mean = float(X_train[:, :, cfg.pet_idx].mean())
    print(
        f"\n  Input check — P(ch{cfg.precip_idx}) mean={p_mean:.3f}, "
        f"PET(ch{cfg.pet_idx}) mean={e_mean:.3f}"
    )
    if abs(p_mean) < 0.5 and np.std(X_train[:, :, cfg.precip_idx]) < 2.0:
        print(
            "  ⚠️  WARNING: Precipitation looks normalized (mean≈0, std≈1). "
            "GR4J needs raw mm/day. Use --seq_suffix '_hybrid'."
        )
    if abs(e_mean) < 0.5 and np.std(X_train[:, :, cfg.pet_idx]) < 2.0:
        print(
            "  ⚠️  WARNING: PET looks normalized (mean≈0, std≈1). "
            "GR4J needs raw mm/day. Use --seq_suffix '_hybrid'."
        )

    # ── Dataloaders ────────────────────────────────────────────────────────────
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
    )

    pin = device.type == "cuda"
    persistent = cfg.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = HybridGR4J_TCN(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
        precip_idx=cfg.precip_idx,
        pet_idx=cfg.pet_idx,
    ).to(device)

    # Speed: torch.compile graph optimization (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="default")
            print("  torch.compile enabled")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,} (trainable: {n_trainable:,})")

    # Print GR4J initial parameters
    with torch.no_grad():
        x1 = F.softplus(model.gr4j.X1) + 1.0
        x2 = model.gr4j.X2.item()
        x3 = F.softplus(model.gr4j.X3) + 1.0
        x4 = (F.softplus(model.gr4j.X4) + 0.5).item()
        sf = 1.0 - np.exp(-x4 / 0.956)
        print(
            f"  GR4J initial: X1={x1:.2f}mm, X2={x2:.4f}, X3={x3:.2f}mm, X4={x4:.2f}d"
        )
        print(f"  Routing store frac (X4={x4:.2f}): {sf:.3f}")

    # ── Loss ───────────────────────────────────────────────────────────────────
    peak_threshold = float(np.percentile(y_train, cfg.peak_percentile))
    print(
        f"  Peak threshold (p{cfg.peak_percentile:.0f} of y_train): {peak_threshold:.4f}"
    )

    if cfg.loss == "nse":
        print("  Loss: Stable NSE (batch-wise)")
        criterion = StableNSELoss().to(device)
    elif cfg.loss == "peak":
        print(f"  Loss: Peak-weighted Huber (weight={cfg.peak_loss_weight}×)")
        criterion = PeakFocusedLoss(
            threshold=peak_threshold,
            huber_delta=cfg.huber_delta,
            peak_weight=cfg.peak_loss_weight,
        ).to(device)
    elif cfg.loss == "bias_huber":
        print(f"  Loss: Huber(δ={cfg.huber_delta}) + bias penalty (w=0.5)")
        criterion = BiasPenalizedHuber(
            huber_delta=cfg.huber_delta,
            bias_weight=0.5,
        ).to(device)
    else:
        print(f"  Loss: plain Huber (δ={cfg.huber_delta}) — peak weighting OFF")
        criterion = nn.SmoothL1Loss(beta=cfg.huber_delta).to(device)

    # ── Optimizer (modest GR4J LR multiplier) ─────────────────────────────────
    gr4j_params = list(model.gr4j.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("gr4j.")]
    print(
        f"  GR4J params: {sum(p.numel() for p in gr4j_params)} "
        f"(lr × 10 = {cfg.lr * 10:.1e})"
    )
    print(
        f"  Other params: {sum(p.numel() for p in other_params):,} "
        f"(lr = {cfg.lr:.1e})"
    )

    optimizer = optim.AdamW(
        [
            {"params": gr4j_params, "lr": cfg.lr * 10, "weight_decay": 0.0},
            {"params": other_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        ],
        betas=(0.9, 0.999),
    )

    # Linear warmup then ReduceLROnPlateau
    plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
        min_lr=1e-6,
    )

    scaler = GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = ModelEMA(model, decay=cfg.ema_decay)

    # ── Training loop ──────────────────────────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*72}\nTraining run {run_id}\n{'='*72}")
    print(
        f"  Arch: Hybrid GR4J-TCN | GR4J(P_idx={cfg.precip_idx},PET_idx={cfg.pet_idx}) → "
        f"TCN {cfg.hidden_dim}×{cfg.num_layers} + GAP"
    )
    print(
        f"  Dropout: {cfg.dropout} | WD: {cfg.weight_decay} | Batch: {cfg.batch_size}"
    )
    print(
        f"  LR: {cfg.lr} (warmup={cfg.warmup_epochs}, then ReduceLROnPlateau) | EMA: {cfg.ema_decay}"
    )
    print(f"  Input: {seq_len}-day lookback × {input_dim} raw + 10 GR4J features\n")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mae_ema": [],
        "lr": [],
    }
    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        # Warmup
        if epoch <= cfg.warmup_epochs:
            lr_scale = epoch / cfg.warmup_epochs
            optimizer.param_groups[0]["lr"] = cfg.lr * 10 * lr_scale
            optimizer.param_groups[1]["lr"] = cfg.lr * lr_scale

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                pred = model(xb)
                loss = criterion(pred, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # NEW: Clamp X2 to physical range (karst springs are gaining)
            with torch.no_grad():
                model.gr4j.X2.clamp_(min=0.0, max=5.0)

            ema.update(model)

            bs = yb.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

        train_loss = train_loss_sum / max(train_n, 1)

        # ── Validate (EMA weights) ─────────────────────────────────────────────
        model.eval()
        with ema.apply_to(model):
            val_loss_sum = 0.0
            val_mae_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    with autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                        pred = model(xb)
                        loss = criterion(pred, yb)
                    val_loss_sum += loss.item() * yb.size(0)
                    val_mae_sum += torch.abs(pred - yb).sum().item()
                    val_n += yb.size(0)

        val_loss = val_loss_sum / max(val_n, 1)
        val_mae = val_mae_sum / max(val_n, 1)

        # LR scheduling after warmup
        if epoch > cfg.warmup_epochs:
            plateau.step(val_loss)

        current_lr = optimizer.param_groups[1]["lr"]  # backbone LR
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_ema"].append(val_mae)
        history["lr"].append(current_lr)

        # ── Checkpointing & early stopping ─────────────────────────────────────
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best EMA weights
            ckpt_path = MODEL_DIR / "checkpoints" / f"gr4j_tcn_best_{tag}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": ema.shadow,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": vars(cfg),
                },
                ckpt_path,
            )
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            print(
                f"  Epoch {epoch:03d} | train={train_loss:.5f} | val={val_loss:.5f} "
                f"| val_MAE={val_mae:.5f} | lr={current_lr:.2e} | best={best_epoch}"
            )

        if patience_counter >= cfg.patience:
            print(f"\n  Early stopping triggered at epoch {epoch} (best={best_epoch})")
            break

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed/60:.1f} min ({elapsed:.0f} s)")

    # ═══════════════════════════════════════════════════════════════════════════════
    # Test evaluation (best EMA model)
    # ═══════════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}\nTest evaluation (best EMA epoch {best_epoch})\n{'='*72}")

    with ema.apply_to(model):
        model.eval()
        all_pred = []
        all_true = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device, non_blocking=True)
                with autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                    pred = model(xb)
                all_pred.append(pred.cpu().numpy())
                all_true.append(yb.numpy())

    pred_norm = np.concatenate(all_pred, axis=0).flatten()
    true_norm = np.concatenate(all_true, axis=0).flatten()

    # ── Denormalise (and invert log-transform if it was applied in windowing.py) ──
    suffix = f"_h{cfg.horizon}_lb{cfg.lookback}{cfg.seq_suffix}"
    scaler_path = ROOT / "data" / "splits" / f"scaler_params{suffix}.csv"
    scaler_df = pd.read_csv(scaler_path, index_col=0)
    q_min = float(scaler_df.loc["__target__", "min"])
    q_max = float(scaler_df.loc["__target__", "max"])

    # Detect log transform from windowing metadata if present
    meta_path = ROOT / "data" / "splits" / f"window_meta{suffix}.json"
    log_transformed = False
    log_eps = 1e-3
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        log_transformed = meta.get("log_transform", False)
        log_eps = meta.get("log_eps", 1e-3)

    # Invert min-max
    pred_raw = pred_norm * (q_max - q_min) + q_min
    true_raw = true_norm * (q_max - q_min) + q_min

    # Invert log1p if needed
    if log_transformed:
        pred_raw = np.expm1(pred_raw) - log_eps
        true_raw = np.expm1(true_raw) - log_eps
        # Clamp negatives from numerical noise
        pred_raw = np.maximum(pred_raw, 0.0)
        true_raw = np.maximum(true_raw, 0.0)

    # ═══════════════════════════════════════════════════════════════════════════════
    # Metrics
    # ═══════════════════════════════════════════════════════════════════════════════
    def pbias(obs, sim):
        return 100.0 * np.sum(sim - obs) / np.sum(obs)

    def peak_bias(obs, sim, percentile=95):
        thresh = np.percentile(obs, percentile)
        mask = obs >= thresh
        if mask.sum() == 0:
            return np.nan
        return 100.0 * (np.mean(sim[mask]) - np.mean(obs[mask])) / np.mean(obs[mask])

    nse_val = nse(true_raw, pred_raw)
    kge_val, r_val, alpha_val, beta_val = safe_kge(true_raw, pred_raw)
    mae_val = mean_absolute_error(true_raw, pred_raw)
    rmse_val = np.sqrt(mean_squared_error(true_raw, pred_raw))
    r2_val = r2_score(true_raw, pred_raw)
    pbias_val = pbias(true_raw, pred_raw)
    peak_bias_val = peak_bias(true_raw, pred_raw, percentile=95)

    print(f"  NSE   : {nse_val:.4f}")
    print(
        f"  KGE   : {kge_val:.4f}  (r={r_val:.4f}, α={alpha_val:.4f}, β={beta_val:.4f})"
    )
    print(f"  MAE   : {mae_val:.4f} m³/s")
    print(f"  RMSE  : {rmse_val:.4f} m³/s")
    print(f"  R²    : {r2_val:.4f}")
    print(f"  PBIAS : {pbias_val:+.2f} %")
    print(f"  Peak% : {peak_bias_val:+.2f} %")

    metrics = {
        "model": "GR4J-TCN",
        "tag": tag,
        "horizon": cfg.horizon,
        "lookback": cfg.lookback,
        "seed": cfg.seed,
        "nse": float(nse_val),
        "kge": float(kge_val),
        "r": float(r_val),
        "alpha": float(alpha_val),
        "beta": float(beta_val),
        "mae": float(mae_val),
        "rmse": float(rmse_val),
        "r2": float(r2_val),
        "pbias": float(pbias_val),
        "peak_bias_pct": float(peak_bias_val),
        "best_epoch": best_epoch,
        "train_time_s": elapsed,
        "loss": cfg.loss,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "input_dim": input_dim,
        "seq_len": seq_len,
        "log_transform": log_transformed,
        "log_eps": log_eps,
        "q_min": q_min,
        "q_max": q_max,
    }

    # Save metrics
    met_path = MET_DIR / f"metrics_gr4j_tcn_{tag}.json"
    with open(met_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {met_path}")

    # Save predictions
    pred_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates_test),
            "observed": true_raw,
            "predicted": pred_raw,
            "residual": true_raw - pred_raw,
        }
    )
    pred_path = PRED_DIR / f"predictions_gr4j_tcn_{tag}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved → {pred_path}")

    # Save final model (non-EMA, for inspection)
    final_path = MODEL_DIR / "trained" / f"gr4j_tcn_final_{tag}.pt"
    torch.save(
        {
            "epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.shadow,
            "config": vars(cfg),
        },
        final_path,
    )

    # Save config
    cfg_path = MODEL_DIR / "configs" / f"gr4j_tcn_config_{tag}.json"
    with open(cfg_path, "w") as f:
        json.dump(vars(cfg), f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════════════
    # Training history plot
    # ═══════════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    fig.patch.set_facecolor("#ffffff")

    # Loss curves
    ax = axes[0, 0]
    ax.plot(
        history["epoch"], history["train_loss"], label="Train", color="#1f77b4", lw=1.2
    )
    ax.plot(history["epoch"], history["val_loss"], label="Val", color="#ff7f0e", lw=1.2)
    ax.axvline(
        best_epoch, color="#2ca02c", ls="--", alpha=0.7, label=f"Best ({best_epoch})"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    light_style(ax)

    # LR schedule
    ax = axes[0, 1]
    ax.plot(history["epoch"], history["lr"], color="#9467bd", lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule")
    ax.set_yscale("log")
    light_style(ax)

    # Observed vs Predicted scatter
    ax = axes[1, 0]
    ax.scatter(true_raw, pred_raw, alpha=0.25, s=8, c="#1f77b4", edgecolors="none")
    lim = [min(true_raw.min(), pred_raw.min()), max(true_raw.max(), pred_raw.max())]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Observed (m³/s)")
    ax.set_ylabel("Predicted (m³/s)")
    ax.set_title(f"Scatter  |  NSE={nse_val:.3f}  KGE={kge_val:.3f}")
    light_style(ax)

    # Time series (last 365 days of test)
    ax = axes[1, 1]
    plot_len = min(365, len(true_raw))
    dates_plot = pd.to_datetime(dates_test[-plot_len:])
    ax.plot(dates_plot, true_raw[-plot_len:], label="Observed", color="#1f77b4", lw=1.2)
    ax.plot(
        dates_plot, pred_raw[-plot_len:], label="Predicted", color="#ff7f0e", lw=1.2
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_title(f"Test Period (last {plot_len} days)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    light_style(ax)

    fig.tight_layout()
    fig_path = FIG_DIR / f"gr4j_tcn_{tag}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved → {fig_path}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # GR4J learned parameters report
    # ═══════════════════════════════════════════════════════════════════════════════
    with torch.no_grad():
        x1 = F.softplus(model.gr4j.X1) + 1.0
        x2 = model.gr4j.X2.item()
        x3 = F.softplus(model.gr4j.X3) + 1.0
        x4 = (F.softplus(model.gr4j.X4) + 0.5).item()
        sf = 1.0 - np.exp(-x4 / 0.956)

    print(f"\n{'='*72}")
    print("Learned GR4J parameters")
    print(f"{'='*72}")
    print(f"  X1 (production capacity) : {x1:.2f} mm")
    print(f"  X2 (groundwater exchange) : {x2:.4f} mm/day  (clamped ≥ 0)")
    print(f"  X3 (routing capacity)    : {x3:.2f} mm")
    print(f"  X4 (time constant)       : {x4:.2f} days")
    print(f"  Routing store fraction   : {sf:.3f}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
