import os
import sys
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
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument(
        "--loss",
        type=str,
        default="huber",
        choices=["huber", "nse", "peak", "bias_huber"],
    )
    p.add_argument("--peak_percentile", type=float, default=90.0)
    p.add_argument("--peak_loss_weight", type=float, default=2.0)
    p.add_argument("--huber_delta", type=float, default=0.15)
    p.add_argument(
        "--x2_min",
        type=float,
        default=-5.0,
        help="GR4J X2 lower bound (allow negative for groundwater loss)",
    )
    p.add_argument("--x2_max", type=float, default=5.0)
    p.add_argument(
        "--freeze_gr4j_epochs",
        type=int,
        default=0,
        help="Freeze GR4J X1-X4 for this many epochs at start (0 = no freeze)",
    )
    p.add_argument(
        "--jit_gr4j",
        action="store_true",
        default=True,
        help="JIT-script GR4J extractor for speed",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
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
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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
        peak_weight: float = 2.0,
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
    """Huber + EMA-smoothed bias penalty.

    The per-batch (pred_mean - true_mean) is noisy when peak events are rare,
    so we track an exponential moving average of the bias across batches.
    """

    def __init__(
        self,
        huber_delta: float = 0.15,
        bias_weight: float = 0.5,
        ema_decay: float = 0.9,
    ):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=huber_delta)
        self.bias_weight = bias_weight
        self.ema_decay = ema_decay
        self.register_buffer("bias_ema", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, y_pred, y_true):
        huber = self.huber(y_pred, y_true)
        with torch.no_grad():
            batch_bias = (y_pred.mean() - y_true.mean()).detach()
            if not self.initialized:
                self.bias_ema.copy_(batch_bias)
                self.initialized.fill_(True)
            else:
                self.bias_ema.mul_(self.ema_decay).add_(
                    batch_bias, alpha=1.0 - self.ema_decay
                )
        # Use current batch bias for gradient, scaled by EMA magnitude for stability.
        current_bias = y_pred.mean() - y_true.mean()
        scale = torch.abs(self.bias_ema).clamp(min=1e-4)
        bias_penalty = torch.abs(current_bias) * (scale / (scale + 1e-2))
        return huber + self.bias_weight * bias_penalty


class GR4JFeatureExtractor(nn.Module):
    """Differentiable GR4J-like reservoir features.

    Note: this is a simplified hybrid formulation, NOT canonical GR4J.
    The canonical GR4J uses two unit hydrographs (UH1 over X4 timesteps,
    UH2 over 2*X4 timesteps) with a 90/10 split between routing and direct
    runoff. Here we collapse the unit hydrographs into a single learned
    `store_frac` derived from X4, which is sufficient for feature extraction
    in the hybrid model but is not a strict GR4J implementation.
    """

    def __init__(self, precip_idx: int = 0, pet_idx: int = 27, eps: float = 1e-6):
        super().__init__()
        self.precip_idx = precip_idx
        self.pet_idx = pet_idx
        self.eps = eps

        self.X1 = nn.Parameter(torch.tensor(300.0))
        self.X2 = nn.Parameter(torch.tensor(1.0))
        self.X3 = nn.Parameter(torch.tensor(100.0))
        self.X4 = nn.Parameter(torch.tensor(2.2))

    def forward(self, x):
        B, T, n_feat = x.shape
        P = x[:, :, self.precip_idx]
        E = x[:, :, self.pet_idx]

        X1 = F.softplus(self.X1) + 1.0
        X2 = self.X2
        X3 = F.softplus(self.X3) + 1.0
        X4 = F.softplus(self.X4) + 0.5

        store_frac = 1.0 - torch.exp(-X4 / 0.956)
        store_frac = torch.clamp(store_frac, 0.05, 0.99)

        S = torch.zeros(B, device=x.device, dtype=x.dtype)
        R = torch.zeros(B, device=x.device, dtype=x.dtype)

        features = []
        for t in range(T):
            P_t = P[:, t]
            E_t = E[:, t]

            Pn = torch.clamp(P_t - E_t, min=0.0)
            En = torch.clamp(E_t - P_t, min=0.0)

            ratio_s = S / (X1 + self.eps)
            tanh_ps = torch.tanh(Pn / (X1 + self.eps))
            Ps = X1 * (1 - ratio_s**2) * tanh_ps / (1 + ratio_s * tanh_ps + self.eps)
            Ps = torch.clamp(Ps, min=0.0)

            tanh_es = torch.tanh(En / (X1 + self.eps))
            Es = S * (2 - ratio_s) * tanh_es / (1 + (1 - ratio_s) * tanh_es + self.eps)
            Es = torch.minimum(Es, S).clamp(min=0.0)

            S = S + torch.where(P_t >= E_t, Ps, -Es)

            perc_ratio = S / (5.25 * X1 + self.eps)
            Perc = S * (1.0 - (1.0 + perc_ratio**4 + self.eps) ** (-0.25))
            Perc = torch.minimum(Perc, S).clamp(min=0.0)
            S = S - Perc

            Pr = torch.where(P_t >= E_t, Pn - Ps, torch.zeros_like(Pn))
            Q9 = Perc + Pr
            Q9 = torch.clamp(Q9, min=0.0)

            Q1 = store_frac * Q9
            Q0 = (1.0 - store_frac) * Q9

            ratio_r = R / (X3 + self.eps)
            Fex = X2 * (ratio_r**3.5)
            Fex = torch.clamp(Fex, min=-X3 * 2, max=X3 * 2)

            R_in = R + Q1 + Fex
            R_in = torch.clamp(R_in, min=0.0)

            qr_ratio = R_in / (X3 + self.eps)
            Qr = R_in * (1.0 - (1.0 + qr_ratio**4 + self.eps) ** (-0.25))
            Qr = torch.clamp(Qr, min=0.0)

            R = R_in - Qr
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

        features = torch.stack(features, dim=1)
        return features


class TCNLayer(nn.Module):
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
        for layer in self.layers:
            x = layer(x)
        return x


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
        self.meteo_dim = 32

        # Separate LayerNorm for GR4J features so they don't get scale-dominated
        # by raw meteo features when concatenated. log1p outputs can be 0..5+,
        # while normalized meteo is ~N(0,1).
        self.gr4j_norm = nn.LayerNorm(10)

        self.tcn = TCNBackbone(
            self.meteo_dim + 10, hidden_dim, num_layers, kernel_size, dropout
        )
        self.tcn_norm = nn.LayerNorm(hidden_dim)

        head_dim = hidden_dim * 2
        self.head = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        meteo = x[:, :, : self.meteo_dim]
        gr4j_feat = self.gr4j(meteo)
        gr4j_feat = self.gr4j_norm(gr4j_feat)
        x = torch.cat([meteo, gr4j_feat], dim=-1)

        x = x.transpose(1, 2)
        h = self.tcn(x)
        h = h.transpose(1, 2)
        h = self.tcn_norm(h)

        last = h[:, -1, :]
        pool = h.mean(dim=1)
        fused = torch.cat([last, pool], dim=-1)

        return self.head(fused)


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


def light_style(ax):
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#333333")
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    ax.title.set_color("#111111")
    ax.xaxis.label.set_color("#444444")
    ax.yaxis.label.set_color("#444444")
    ax.grid(alpha=0.2, color="#cccccc")


def set_gr4j_requires_grad(model, requires_grad: bool):
    for p in model.gr4j.parameters():
        p.requires_grad = requires_grad


def main():
    cfg = get_config()
    set_seed(cfg.seed)

    if cfg.batch_size is None:
        cfg.batch_size = 1024 if cfg.lookback <= 60 else 512

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
        torch.set_float32_matmul_precision("high")
        print("  TF32 matmul precision enabled")

    print("\nLoading sequences...")
    seq_suffix = getattr(cfg, "seq_suffix", "")
    suffix = f"_h{cfg.horizon}_lb{cfg.lookback}{seq_suffix}"
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

    p_mean = float(X_train[:, :, cfg.precip_idx].mean())
    e_mean = float(X_train[:, :, cfg.pet_idx].mean())
    print(
        f"\n  Input check — P(ch{cfg.precip_idx}) mean={p_mean:.3f}, "
        f"PET(ch{cfg.pet_idx}) mean={e_mean:.3f}"
    )
    if abs(p_mean) < 0.5 and np.std(X_train[:, :, cfg.precip_idx]) < 2.0:
        print(
            "  WARNING: Precipitation looks normalized (mean≈0, std≈1). "
            "GR4J needs raw mm/day. Use --seq_suffix '_hybrid'."
        )
    if abs(e_mean) < 0.5 and np.std(X_train[:, :, cfg.pet_idx]) < 2.0:
        print(
            "  WARNING: PET looks normalized (mean≈0, std≈1). "
            "GR4J needs raw mm/day. Use --seq_suffix '_hybrid'."
        )

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

    model = HybridGR4J_TCN(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
        precip_idx=cfg.precip_idx,
        pet_idx=cfg.pet_idx,
    ).to(device)

    # JIT-script the GR4J extractor — the loop is static, so scripting can give
    # a notable speedup on the t-loop without changing semantics. If it fails
    # (e.g. unsupported op on this torch version), we fall back silently.
    if cfg.jit_gr4j:
        try:
            model.gr4j = torch.jit.script(model.gr4j)
            print("  GR4J extractor JIT-scripted")
        except Exception as e:
            print(f"  GR4J JIT-script skipped: {e}")

    # torch.compile on the whole model — note that the GR4J Python loop may
    # cause graph breaks; this is logged for diagnosis if needed.
    if hasattr(torch, "compile") and sys.platform != "win32":
        try:
            model = torch.compile(model, mode="default")
            print("  torch.compile enabled (use TORCH_LOGS=graph_breaks to debug)")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")
    else:
        print("  torch.compile skipped: Windows platform (Triton unavailable)")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,} (trainable: {n_trainable:,})")

    # Resolve the underlying GR4J module robustly (after possible compile/script).
    def _get_gr4j(m):
        # torch.compile wraps with OptimizedModule that exposes ._orig_mod
        base = getattr(m, "_orig_mod", m)
        return base.gr4j

    gr4j_mod = _get_gr4j(model)

    with torch.no_grad():
        x1 = F.softplus(gr4j_mod.X1) + 1.0
        x2 = gr4j_mod.X2.item()
        x3 = F.softplus(gr4j_mod.X3) + 1.0
        x4 = (F.softplus(gr4j_mod.X4) + 0.5).item()
        sf = 1.0 - np.exp(-x4 / 0.956)
        print(
            f"  GR4J initial: X1={x1:.2f}mm, X2={x2:.4f}, X3={x3:.2f}mm, X4={x4:.2f}d"
        )
        print(f"  Routing store frac (X4={x4:.2f}): {sf:.3f}")

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
        print(f"  Loss: Huber(δ={cfg.huber_delta}) + EMA-smoothed bias penalty")
        criterion = BiasPenalizedHuber(
            huber_delta=cfg.huber_delta,
            bias_weight=0.5,
            ema_decay=0.9,
        ).to(device)
    else:
        print(f"  Loss: plain Huber (δ={cfg.huber_delta})")
        criterion = nn.SmoothL1Loss(beta=cfg.huber_delta).to(device)

    # Resolve GR4J params via the underlying module (handles compile/script wrapping).
    gr4j_param_ids = {id(p) for p in gr4j_mod.parameters()}
    gr4j_params = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in gr4j_param_ids:
            gr4j_params.append(p)
        else:
            other_params.append(p)

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

    plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
        min_lr=1e-6,
    )

    scaler = GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = ModelEMA(model, decay=cfg.ema_decay)

    # Optional two-stage training: freeze GR4J for an initial warm-start phase.
    if cfg.freeze_gr4j_epochs > 0:
        set_gr4j_requires_grad(_get_gr4j(model), False)
        print(
            f"  GR4J X1-X4 frozen for first {cfg.freeze_gr4j_epochs} epochs (warm-start)"
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'=' * 72}\nTraining run {run_id}\n{'=' * 72}")
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
    print(f"  X2 clamp: [{cfg.x2_min}, {cfg.x2_max}] (allows groundwater loss)")
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
        # Unfreeze GR4J after warm-start phase
        if cfg.freeze_gr4j_epochs > 0 and epoch == cfg.freeze_gr4j_epochs + 1:
            set_gr4j_requires_grad(_get_gr4j(model), True)
            print(f"  Epoch {epoch}: GR4J X1-X4 unfrozen, joint training resumes")

        if epoch <= cfg.warmup_epochs:
            lr_scale = epoch / cfg.warmup_epochs
            optimizer.param_groups[0]["lr"] = cfg.lr * 10 * lr_scale
            optimizer.param_groups[1]["lr"] = cfg.lr * lr_scale

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

            with torch.no_grad():
                # Allow X2 in [x2_min, x2_max] — negative = groundwater loss
                _get_gr4j(model).X2.clamp_(min=cfg.x2_min, max=cfg.x2_max)

            ema.update(model)

            bs = yb.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

        train_loss = train_loss_sum / max(train_n, 1)

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

        if epoch > cfg.warmup_epochs:
            plateau.step(val_loss)

        current_lr = optimizer.param_groups[1]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_ema"].append(val_mae)
        history["lr"].append(current_lr)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0

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

    print(f"\n{'=' * 72}\nTest evaluation (best EMA epoch {best_epoch})\n{'=' * 72}")

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

    scaler_path = ROOT / "data" / "splits" / f"scaler_params{suffix}.csv"
    scaler_df = pd.read_csv(scaler_path, index_col=0)
    q_min = float(scaler_df.loc["__target__", "min"])
    q_max = float(scaler_df.loc["__target__", "max"])

    meta_path = ROOT / "data" / "sequences" / f"window_meta{suffix}.json"
    log_transformed = False
    log_eps = 1e-3
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        log_transformed = meta.get("log_transform", False)
        log_eps = meta.get("log_eps", 1e-3)

    pred_raw = pred_norm * (q_max - q_min) + q_min
    true_raw = true_norm * (q_max - q_min) + q_min

    if log_transformed:
        pred_raw = np.expm1(pred_raw) - log_eps
        true_raw = np.expm1(true_raw) - log_eps
        pred_raw = np.maximum(pred_raw, 0.0)
        true_raw = np.maximum(true_raw, 0.0)

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
    log_nse_val = log_nse(true_raw, pred_raw)

    print(f"  NSE     : {nse_val:.4f}")
    print(f"  logNSE  : {log_nse_val:.4f}")
    print(
        f"  KGE     : {kge_val:.4f}  (r={r_val:.4f}, α={alpha_val:.4f}, β={beta_val:.4f})"
    )
    print(f"  MAE     : {mae_val:.4f} m³/s")
    print(f"  RMSE    : {rmse_val:.4f} m³/s")
    print(f"  R²      : {r2_val:.4f}")
    print(f"  PBIAS   : {pbias_val:+.2f} %")
    print(f"  Peak%   : {peak_bias_val:+.2f} %")

    metrics = {
        "model": "GR4J-TCN",
        "tag": tag,
        "horizon": cfg.horizon,
        "lookback": cfg.lookback,
        "seed": cfg.seed,
        "nse": float(nse_val),
        "log_nse": float(log_nse_val),
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

    met_path = MET_DIR / f"metrics_gr4j_tcn_{tag}.json"
    with open(met_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {met_path}")

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

    cfg_path = MODEL_DIR / "configs" / f"gr4j_tcn_config_{tag}.json"
    with open(cfg_path, "w") as f:
        json.dump(vars(cfg), f, indent=2)

    with torch.no_grad():
        gr4j_mod = _get_gr4j(model)
        x1 = F.softplus(gr4j_mod.X1) + 1.0
        x2 = gr4j_mod.X2.item()
        x3 = F.softplus(gr4j_mod.X3) + 1.0
        x4 = (F.softplus(gr4j_mod.X4) + 0.5).item()
        sf = 1.0 - np.exp(-x4 / 0.956)

    print(f"\n{'=' * 72}")
    print("Learned GR4J parameters")
    print(f"{'=' * 72}")
    print(f"  X1 (production capacity) : {x1:.2f} mm")
    print(
        f"  X2 (groundwater exchange) : {x2:+.4f} mm/day  (clamped to [{cfg.x2_min}, {cfg.x2_max}])"
    )
    print(f"  X3 (routing capacity)    : {x3:.2f} mm")
    print(f"  X4 (time constant)       : {x4:.2f} days")
    print(f"  Routing store fraction   : {sf:.3f}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
