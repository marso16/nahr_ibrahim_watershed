import os
import json
import time
import random
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler  # newer API; torch.cuda.amp is deprecated
from pathlib import Path
from datetime import datetime
from copy import deepcopy

# Newer parametrize-based weight_norm. If you're on PyTorch < 1.12 swap to:
#   from torch.nn.utils import weight_norm
from torch.nn.utils.parametrizations import weight_norm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats


class NSELoss(nn.Module):
    """
    Batch-wise NSE-style loss. Minimising this maximises NSE per batch.

    NSE = 1 - sum((y_pred - y_true)^2) / sum((y_true - y_mean)^2)
    Loss = sum((y_pred - y_true)^2) / sum((y_true - y_mean)^2)

    Sensitive to batch composition — use larger batches (>= 128) for stability,
    since per-batch variance of y_true is noisier with small batches.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        var = ((y_true - y_true.mean()) ** 2).sum() + self.eps
        sq_err = ((y_pred - y_true) ** 2).sum()
        return sq_err / var


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════
def get_config():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        default=os.environ.get(
            "WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed"
        ),
        help="Project root directory",
    )
    # Architecture
    # Stage 1: levels_1 dilated blocks at dilations 1, 2, 4, ... 2^(levels_1-1)
    # Stage 2: levels_2 dilated blocks at dilations 2^levels_1, ... continuing
    # Receptive field = 1 + 2*(kernel_size-1) * sum(dilations)
    p.add_argument("--channels_1", type=int, default=128)
    p.add_argument("--channels_2", type=int, default=64)
    p.add_argument("--levels_1", type=int, default=3)
    p.add_argument("--levels_2", type=int, default=3)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--attention_dim", type=int, default=128)
    p.add_argument("--attention_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument(
        "--spatial_dropout",
        type=float,
        default=0.15,
        help="Channel-wise dropout inside TCN blocks (analog of LSTM recurrent dropout)",
    )

    # Training
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--ema_decay", type=float, default=0.999)

    # Peak handling (loss-only, NOT sampling)
    # Set --use_peak_loss to False (or --peak_loss_weight 1.0) for an
    # ablation that uses plain Huber. Peak weighting helps Peak_MAE but
    # often hurts NSE/KGE by inflating the predicted variance.
    p.add_argument(
        "--use_peak_loss",
        action="store_true",
        default=False,
        help="Enable peak-weighted Huber loss (default: off, plain Huber)",
    )

    p.add_argument(
        "--use_nse_loss",
        action="store_true",
        default=False,
        help="Train with batch-wise NSE-loss instead of Huber",
    )

    p.add_argument("--peak_percentile", type=float, default=85.0)
    p.add_argument("--peak_loss_weight", type=float, default=2.5)
    p.add_argument("--huber_delta", type=float, default=0.05)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════
class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over a sequence."""

    def __init__(self, input_dim: int, attention_dim: int, dropout: float = 0.0):
        super().__init__()
        self.W = nn.Linear(input_dim, attention_dim, bias=True)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        scores = self.v(torch.tanh(self.W(x))).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        context = (x * weights).sum(dim=1)  # (B, D)
        return self.dropout(context), weights.squeeze(-1)


class Chomp1d(nn.Module):
    """Trim the right side after causal padding so output length == input length."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TCNBlock(nn.Module):
    """
    Standard dilated residual TCN block: two causal convs with weight norm,
    SiLU activations, channel-wise dropout, and a residual connection
    (1x1 conv if channel counts differ).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.SiLU()
        # Dropout1d drops entire channels — variational-style for 1D conv data.
        # If you're on PyTorch < 1.12, swap to nn.Dropout(p=dropout).
        self.drop1 = nn.Dropout1d(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout1d(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.final_act = nn.SiLU()

        self._init_weights()

    def _init_weights(self):
        # Kaiming init suitable for SiLU/ReLU-family activations
        for c in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(c.weight, nonlinearity="relu")
            if c.bias is not None:
                nn.init.zeros_(c.bias)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x):
        # x: (B, C, T)
        out = self.drop1(self.act1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.act2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.final_act(out + res)


class WatershedTCN(nn.Module):
    """
    Two-stage causal TCN with additive attention + multi-head self-attention,
    designed to mirror the structure of the WatershedLSTM:

      stage_1 (low-level features, larger #channels) → MHA branch
      stage_2 (high-level features, smaller #channels) → AddAttn branch
      + last-step pooling on stage_2
      + residual projection from last input timestep
      → fusion → MLP head
    """

    def __init__(
        self,
        input_dim: int,
        channels_1: int = 128,
        channels_2: int = 64,
        levels_1: int = 3,
        levels_2: int = 3,
        kernel_size: int = 3,
        attention_dim: int = 128,
        attention_heads: int = 4,
        dropout: float = 0.30,
        spatial_dropout: float = 0.15,
    ):
        super().__init__()

        # ── Stage 1: input_dim → channels_1, dilations 1, 2, 4, ... ──────────
        s1 = []
        in_ch = input_dim
        for i in range(levels_1):
            dilation = 2**i
            s1.append(
                TCNBlock(in_ch, channels_1, kernel_size, dilation, spatial_dropout)
            )
            in_ch = channels_1
        self.stage_1 = nn.Sequential(*s1)

        # ── Stage 2: channels_1 → channels_2, dilations continue ─────────────
        s2 = []
        in_ch = channels_1
        for i in range(levels_2):
            dilation = 2 ** (levels_1 + i)
            s2.append(
                TCNBlock(in_ch, channels_2, kernel_size, dilation, spatial_dropout)
            )
            in_ch = channels_2
        self.stage_2 = nn.Sequential(*s2)

        self.layer_norm_1 = nn.LayerNorm(channels_1)
        self.layer_norm_2 = nn.LayerNorm(channels_2)

        # Multi-head self-attention over stage_1 outputs (low-level / dense)
        self.mha = nn.MultiheadAttention(
            channels_1,
            num_heads=attention_heads,
            dropout=dropout / 2,
            batch_first=True,
        )
        self.mha_norm = nn.LayerNorm(channels_1)

        # Additive attention over stage_2 outputs (high-level / sparse)
        self.attention = AdditiveAttention(
            channels_2, attention_dim, dropout=dropout / 2
        )

        # Residual projection from last input timestep
        self.input_residual = nn.Sequential(
            nn.Linear(input_dim, channels_2),
            nn.SiLU(),
        )

        # Fusion: last_step (C2) + attn_context (C2) + mha_pool (C1) + residual (C2)
        fusion_dim = channels_2 + channels_2 + channels_1 + channels_2

        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

        self._init_head_weights()

        # Cache effective receptive field for logging
        self.receptive_field = self._compute_receptive_field(
            levels_1 + levels_2, kernel_size
        )

    @staticmethod
    def _compute_receptive_field(num_blocks: int, kernel_size: int) -> int:
        # Each block contains 2 dilated convs of the same dilation.
        # RF contribution per block = 2 * (k - 1) * dilation
        rf = 1
        for i in range(num_blocks):
            rf += 2 * (kernel_size - 1) * (2**i)
        return rf

    def _init_head_weights(self):
        for module in [self.input_residual, self.head, self.attention]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, T, F)
        x_t = x.transpose(1, 2)  # (B, F, T) for conv

        h1 = self.stage_1(x_t)  # (B, C1, T)
        h1_seq = self.layer_norm_1(h1.transpose(1, 2))  # (B, T, C1)

        h2 = self.stage_2(h1_seq.transpose(1, 2))  # (B, C2, T)
        h2_seq = self.layer_norm_2(h2.transpose(1, 2))  # (B, T, C2)

        # MHA on stage_1 (residual + norm)
        mha_out, _ = self.mha(h1_seq, h1_seq, h1_seq)
        mha_out = self.mha_norm(h1_seq + mha_out)
        mha_pool = mha_out.mean(dim=1)

        # Additive attention on stage_2
        attn_context, _ = self.attention(h2_seq)

        # Last step of stage_2
        last_step = h2_seq[:, -1, :]

        # Residual from last input timestep
        residual = self.input_residual(x[:, -1, :])

        # Fuse and predict
        fused = torch.cat([last_step, attn_context, mha_pool, residual], dim=-1)
        return self.head(fused)


# ═══════════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════════
class PeakFocusedLoss(nn.Module):
    """Huber + peak-weighted MSE. Threshold is passed in explicitly."""

    def __init__(
        self,
        threshold: float,
        huber_delta: float = 0.05,
        peak_weight: float = 2.5,
    ):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=huber_delta, reduction="none")
        self.register_buffer("threshold", torch.tensor(threshold, dtype=torch.float32))
        self.peak_weight = peak_weight

    def forward(self, y_pred, y_true):
        per_sample = self.huber(y_pred, y_true)  # (B, 1)
        # Per-sample weights: 1 below threshold, peak_weight above (smooth ramp)
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


# ═══════════════════════════════════════════════════════════════════════════════
# EMA
# ═══════════════════════════════════════════════════════════════════════════════
class ModelEMA:
    """Exponential moving average of model parameters."""

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
        """Returns a context manager that swaps EMA weights in temporarily."""
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
# Hydrological metrics (with NaN guards)
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
# Plot styling helpers
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

    # ── Paths ─────────────────────────────────────────────────────────────────
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

    # ── Load sequences ────────────────────────────────────────────────────────
    print("\nLoading sequences...")
    X_train = np.load(SEQ_DIR / "X_train.npy")
    y_train = np.load(SEQ_DIR / "y_train.npy")
    X_val = np.load(SEQ_DIR / "X_val.npy")
    y_val = np.load(SEQ_DIR / "y_val.npy")
    X_test = np.load(SEQ_DIR / "X_test.npy")
    y_test = np.load(SEQ_DIR / "y_test.npy")
    dates_test = np.load(SEQ_DIR / "dates_test.npy", allow_pickle=True)

    total = len(X_train) + len(X_val) + len(X_test)
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(
        f"  Split:   train={len(X_train)/total*100:.1f}% "
        f"val={len(X_val)/total*100:.1f}% test={len(X_test)/total*100:.1f}%"
    )

    assert len(X_train) / total > 0.5, "training set seems too small"
    assert len(X_test) / total > 0.1, "test set seems too small"

    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]

    # Peak threshold computed from TRAIN only (no leakage)
    peak_threshold = float(np.percentile(y_train, cfg.peak_percentile))
    print(
        f"  Peak threshold (p{cfg.peak_percentile:.0f} of y_train): "
        f"{peak_threshold:.4f}"
    )

    # ── Dataloaders ───────────────────────────────────────────────────────────
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
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = WatershedTCN(
        input_dim=input_dim,
        channels_1=cfg.channels_1,
        channels_2=cfg.channels_2,
        levels_1=cfg.levels_1,
        levels_2=cfg.levels_2,
        kernel_size=cfg.kernel_size,
        attention_dim=cfg.attention_dim,
        attention_heads=cfg.attention_heads,
        dropout=cfg.dropout,
        spatial_dropout=cfg.spatial_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,} (trainable: {n_trainable:,})")
    print(
        f"  Receptive field: {model.receptive_field} timesteps "
        f"(seq_len = {seq_len})"
    )
    if model.receptive_field < seq_len:
        print(
            "  NOTE: receptive field is smaller than sequence length — "
            "consider increasing --levels_1/--levels_2 or --kernel_size"
        )

    # Loss, optimizer, scheduler, EMA
    if cfg.use_nse_loss:
        print(f"  Loss: NSE-loss (batch-wise)")
        criterion = NSELoss().to(device)
    elif cfg.use_peak_loss:
        print(f"  Loss: peak-weighted Huber (peak_weight={cfg.peak_loss_weight}×)")
        criterion = PeakFocusedLoss(
            threshold=peak_threshold,
            huber_delta=cfg.huber_delta,
            peak_weight=cfg.peak_loss_weight,
        ).to(device)
    else:
        print(f"  Loss: plain Huber (δ={cfg.huber_delta}) — peak weighting disabled")
        criterion = nn.SmoothL1Loss(beta=cfg.huber_delta).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    # Linear warmup then ReduceLROnPlateau
    plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    scaler = GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = ModelEMA(model, decay=cfg.ema_decay)

    # ── Training loop ─────────────────────────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'=' * 72}\nTraining run {run_id}\n{'=' * 72}")
    print(
        f"  Arch: TCN {cfg.channels_1}×{cfg.levels_1} → {cfg.channels_2}×{cfg.levels_2} "
        f"(k={cfg.kernel_size}) + AddAttn({cfg.attention_dim}) "
        f"+ MHA({cfg.attention_heads}h)"
    )
    print(
        f"  Dropout: spatial={cfg.dropout} channel-wise={cfg.spatial_dropout} | "
        f"WD: {cfg.weight_decay}"
    )
    print(
        f"  LR: {cfg.lr} (warmup={cfg.warmup_epochs}, then ReduceLROnPlateau) | "
        f"Batch: {cfg.batch_size} | EMA: {cfg.ema_decay}"
    )
    if cfg.use_peak_loss:
        print(
            f"  Loss: Huber(δ={cfg.huber_delta}) with peak weight={cfg.peak_loss_weight}× "
            f"above p{cfg.peak_percentile:.0f}"
        )
    else:
        print(f"  Loss: plain Huber(δ={cfg.huber_delta}) — peak weighting OFF")
    print(f"  Input: {seq_len}-day lookback × {input_dim} features\n")

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
        # Linear warmup
        if epoch <= cfg.warmup_epochs:
            lr_scale = epoch / cfg.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * lr_scale

        # ── Train ─────────────────────────────────────────────────────────────
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

            ema.update(model)

            bs = yb.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs

        train_loss = train_loss_sum / max(train_n, 1)

        # ── Validate (with EMA weights) ───────────────────────────────────────
        model.eval()
        val_loss_sum, val_mae_sum, val_n = 0.0, 0.0, 0
        with ema.apply_to(model), torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                    pred = model(xb)
                    loss = criterion(pred, yb)
                bs = yb.size(0)
                val_loss_sum += loss.item() * bs
                val_mae_sum += torch.abs(pred - yb).sum().item()
                val_n += bs

        val_loss = val_loss_sum / max(val_n, 1)
        val_mae = val_mae_sum / max(val_n, 1)

        if epoch > cfg.warmup_epochs:
            plateau.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_ema"].append(val_mae)
        history["lr"].append(current_lr)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  Epoch {epoch:3d}/{cfg.epochs} | "
                f"train={train_loss:.5f}  val={val_loss:.5f}  "
                f"val_mae(ema)={val_mae:.5f}  lr={current_lr:.2e}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save EMA weights as the "best" model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": ema.shadow,
                    "val_loss": val_loss,
                    "config": vars(cfg),
                    "peak_threshold": peak_threshold,
                },
                MODEL_DIR / "checkpoints" / "tcn_best.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            print(
                f"\n  Early stopping at epoch {epoch} "
                f"(best epoch {best_epoch}, val={best_val:.5f})"
            )
            break

    train_time = time.time() - start
    print(f"\nTraining time: {train_time/60:.1f} min")
    print(f"Best val loss: {best_val:.6f} @ epoch {best_epoch}")

    # ── Load best (EMA) weights ───────────────────────────────────────────────
    ckpt = torch.load(MODEL_DIR / "checkpoints" / "tcn_best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    torch.save(model.state_dict(), MODEL_DIR / "trained" / "tcn_final.pt")

    # ── History CSV ───────────────────────────────────────────────────────────
    history_df = pd.DataFrame(history)
    history_df.to_csv(MODEL_DIR / "configs" / "tcn_training_log.csv", index=False)

    # ── Evaluate on test ──────────────────────────────────────────────────────
    print(f"\n{'=' * 72}\nEvaluation on test set\n{'=' * 72}")

    y_pred_list = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            with autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                pred = model(xb)
            y_pred_list.append(pred.float().cpu().numpy())
    y_pred = np.concatenate(y_pred_list).flatten()

    # Denormalise (and invert log-transform if it was applied in split.py)
    scaler_df = pd.read_csv(ROOT / "data" / "splits" / "scaler_params.csv", index_col=0)
    q_min = float(scaler_df.loc["discharge_m3s", "min"])
    q_max = float(scaler_df.loc["discharge_m3s", "max"])

    # Check for log-transform metadata (added by updated split.py)
    if "__meta__" in scaler_df.index:
        log_transform = bool(float(scaler_df.loc["__meta__", "min"]))
        log_eps = float(scaler_df.loc["__meta__", "max"])
    else:
        log_transform = False
        log_eps = 0.0

    # Step 1: inverse min-max → log-space (or real-space if not log-transformed)
    y_test_lin = y_test * (q_max - q_min) + q_min
    y_pred_lin = y_pred * (q_max - q_min) + q_min

    # Step 2: if log-transform was applied, exp it back
    if log_transform:
        print(f"  Inverting log-transform (eps={log_eps})")
        y_test_real = np.exp(y_test_lin) - log_eps
        y_pred_real = np.exp(y_pred_lin) - log_eps
        # Clip any tiny negatives from numerical noise
        y_test_real = np.maximum(y_test_real, 0.0)
        y_pred_real = np.maximum(y_pred_real, 0.0)
    else:
        y_test_real = y_test_lin
        y_pred_real = y_pred_lin

    # Metrics
    r2 = r2_score(y_test_real, y_pred_real)
    mae_real = mean_absolute_error(y_test_real, y_pred_real)
    rmse_real = float(np.sqrt(mean_squared_error(y_test_real, y_pred_real)))
    nse_val = nse(y_test_real, y_pred_real)
    kge_val, kge_r, kge_alpha, kge_beta = safe_kge(y_test_real, y_pred_real)
    log_nse_val = log_nse(y_test_real, y_pred_real)
    pbias = 100 * np.sum(y_pred_real - y_test_real) / np.sum(y_test_real)

    peak_mask = y_test_real >= np.percentile(y_test_real, 95)
    if peak_mask.sum() > 0:
        peak_bias = (
            100
            * (y_pred_real[peak_mask].mean() - y_test_real[peak_mask].mean())
            / y_test_real[peak_mask].mean()
        )
        peak_mae = mean_absolute_error(y_test_real[peak_mask], y_pred_real[peak_mask])
        peak_rmse = float(
            np.sqrt(mean_squared_error(y_test_real[peak_mask], y_pred_real[peak_mask]))
        )
    else:
        peak_bias = peak_mae = peak_rmse = np.nan

    print(f"  R²:     {r2:.4f}")
    print(f"  MAE:    {mae_real:.3f} m³/s")
    print(f"  RMSE:   {rmse_real:.3f} m³/s")
    print(f"  NSE:    {nse_val:.4f}  (target > 0.8)")
    print(f"  KGE:    {kge_val:.4f}  (target > 0.8)")
    print(f"    r={kge_r:.3f} | α={kge_alpha:.3f} | β={kge_beta:.3f}")
    print(f"  logNSE: {log_nse_val:.4f}")
    print(f"  PBIAS:  {pbias:+.2f}%")
    print(f"  Peak Bias: {peak_bias:+.2f}%  |  Peak MAE: {peak_mae:.3f}")

    # Save metrics CSV
    pd.DataFrame(
        [
            {
                "split": "Test",
                "NSE": round(float(nse_val), 4),
                "KGE": round(float(kge_val), 4),
                "RMSE": round(rmse_real, 4),
                "MAE": round(mae_real, 4),
                "R2": round(float(r2), 4),
                "PBIAS_%": round(float(pbias), 2),
                "Peak_Bias_%": round(float(peak_bias), 2),
                "Log_NSE": round(float(log_nse_val), 4),
                "Peak_MAE": round(float(peak_mae), 4),
                "Peak_RMSE": round(float(peak_rmse), 4),
                "KGE_r": round(float(kge_r), 4),
                "KGE_alpha": round(float(kge_alpha), 4),
                "KGE_beta": round(float(kge_beta), 4),
            }
        ]
    ).to_csv(MET_DIR / "tcn_metrics.csv", index=False)

    # Save predictions
    pred_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates_test),
            "observed": y_test_real,
            "predicted": y_pred_real,
            "residual": y_test_real - y_pred_real,
            "is_peak": peak_mask,
        }
    )
    pred_df.to_csv(PRED_DIR / "tcn_predictions_test.csv", index=False)

    # Save hparams + results
    hparams = {
        **vars(cfg),
        "run_id": run_id,
        "device": str(device),
        "epochs_trained": int(history_df["epoch"].iloc[-1]),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "training_time_min": float(train_time / 60),
        "n_params": int(n_params),
        "receptive_field": int(model.receptive_field),
        "peak_threshold_normalised": float(peak_threshold),
        "test_r2": float(r2),
        "test_mae": float(mae_real),
        "test_rmse": float(rmse_real),
        "test_nse": float(nse_val),
        "test_kge": float(kge_val),
        "test_log_nse": float(log_nse_val),
        "test_pbias_pct": float(pbias),
        "test_peak_bias_pct": float(peak_bias),
        "kge_r": float(kge_r),
        "kge_alpha": float(kge_alpha),
        "kge_beta": float(kge_beta),
    }
    with open(LOG_DIR / f"hparams_tcn_{run_id}.json", "w") as f:
        json.dump(hparams, f, indent=2)

    # ── Plot dashboard ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#ffffff")
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: timeseries
    ax1 = fig.add_subplot(gs[0, :2])
    light_style(ax1)
    ax1.plot(
        pred_df["date"], pred_df["observed"], color="#1f77b4", lw=1.2, label="Observed"
    )
    ax1.plot(
        pred_df["date"],
        pred_df["predicted"],
        color="#ff7f0e",
        lw=1.2,
        alpha=0.85,
        label="Predicted",
    )
    ax1.scatter(
        pred_df.loc[peak_mask, "date"],
        pred_df.loc[peak_mask, "observed"],
        color="#d62728",
        s=15,
        alpha=0.7,
        label="Peaks (p95)",
        zorder=5,
    )
    ax1.fill_between(pred_df["date"], pred_df["observed"], alpha=0.08, color="#1f77b4")
    ax1.set_ylabel("Discharge (m³/s)")
    ax1.set_title(
        f"Forecast — R²={r2:.3f} | NSE={nse_val:.3f} | KGE={kge_val:.3f} | "
        f"MAE={mae_real:.2f} | Peak MAE={peak_mae:.2f}"
    )
    ax1.legend(
        loc="upper right",
        facecolor="#ffffff",
        edgecolor="#cccccc",
        labelcolor="#333333",
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # Panel 2: metrics box
    ax2 = fig.add_subplot(gs[0, 2])
    light_style(ax2)
    ax2.axis("off")
    nse_mark = "✓" if nse_val > 0.8 else "✗"
    kge_mark = "✓" if kge_val > 0.8 else "✗"
    txt = (
        f"Test Metrics\n{'─'*22}\n"
        f"R²:     {r2:.4f}\n"
        f"NSE:    {nse_val:.4f}  {nse_mark}\n"
        f"KGE:    {kge_val:.4f}  {kge_mark}\n"
        f"logNSE: {log_nse_val:.4f}\n"
        f"{'─'*22}\n"
        f"MAE:    {mae_real:.3f} m³/s\n"
        f"RMSE:   {rmse_real:.3f} m³/s\n"
        f"PBIAS:  {pbias:+.2f}%\n"
        f"{'─'*22}\n"
        f"Peak Bias: {peak_bias:+.2f}%\n"
        f"Peak MAE:  {peak_mae:.2f}\n"
        f"Peak RMSE: {peak_rmse:.2f}\n"
        f"{'─'*22}\n"
        f"KGE breakdown\n"
        f"  r = {kge_r:.3f}\n"
        f"  α = {kge_alpha:.3f}\n"
        f"  β = {kge_beta:.3f}"
    )
    ax2.text(
        0.05,
        0.95,
        txt,
        transform=ax2.transAxes,
        fontsize=10,
        fontfamily="monospace",
        color="#111111",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f8f9fa",
            edgecolor="#cccccc",
            alpha=0.95,
        ),
    )

    # Panel 3: scatter
    ax3 = fig.add_subplot(gs[1, 0])
    light_style(ax3)
    ax3.scatter(
        pred_df["observed"], pred_df["predicted"], alpha=0.35, s=12, color="#9467bd"
    )
    ax3.scatter(
        pred_df.loc[peak_mask, "observed"],
        pred_df.loc[peak_mask, "predicted"],
        alpha=0.6,
        s=20,
        color="#d62728",
        label="Peaks",
    )
    lims = [
        min(pred_df["observed"].min(), pred_df["predicted"].min()),
        max(pred_df["observed"].max(), pred_df["predicted"].max()),
    ]
    ax3.plot(lims, lims, "--", color="#888888", alpha=0.7, lw=1.2, label="1:1")
    ax3.set_xlabel("Observed (m³/s)")
    ax3.set_ylabel("Predicted (m³/s)")
    ax3.set_title("Observed vs Predicted")
    ax3.legend(
        loc="upper left",
        facecolor="#ffffff",
        edgecolor="#cccccc",
        labelcolor="#333333",
    )

    # Panel 4: residuals time series
    ax4 = fig.add_subplot(gs[1, 1])
    light_style(ax4)
    ax4.axhline(0, color="#888888", lw=0.8, alpha=0.6)
    ax4.plot(pred_df["date"], pred_df["residual"], color="#d62728", lw=0.7, alpha=0.7)
    ax4.scatter(
        pred_df.loc[peak_mask, "date"],
        pred_df.loc[peak_mask, "residual"],
        color="#d62728",
        s=15,
        alpha=0.8,
        zorder=5,
    )
    ax4.fill_between(
        pred_df["date"], pred_df["residual"], 0, alpha=0.15, color="#d62728"
    )
    ax4.set_ylabel("Residual (m³/s)")
    ax4.set_title("Residuals (Observed − Predicted)")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # Panel 5: residual histogram
    ax5 = fig.add_subplot(gs[1, 2])
    light_style(ax5)
    ax5.hist(
        pred_df["residual"],
        bins=60,
        color="#2ca02c",
        alpha=0.7,
        edgecolor="#ffffff",
        density=True,
    )
    ax5.hist(
        pred_df.loc[peak_mask, "residual"],
        bins=30,
        color="#d62728",
        alpha=0.5,
        edgecolor="#ffffff",
        density=True,
        label="Peak residuals",
    )
    ax5.axvline(0, color="#888888", ls="--", lw=1)
    ax5.set_xlabel("Residual (m³/s)")
    ax5.set_ylabel("Density")
    ax5.set_title("Residual Distribution")
    ax5.legend(
        loc="upper right",
        facecolor="#ffffff",
        edgecolor="#cccccc",
        labelcolor="#333333",
    )

    # Panel 6: peak scatter
    ax6 = fig.add_subplot(gs[2, 0])
    light_style(ax6)
    pdf_peak = pred_df[peak_mask]
    if len(pdf_peak) > 0:
        ax6.scatter(
            pdf_peak["observed"],
            pdf_peak["predicted"],
            alpha=0.6,
            s=25,
            color="#d62728",
        )
        plims = [pdf_peak["observed"].min() * 0.9, pdf_peak["observed"].max() * 1.05]
        ax6.plot(plims, plims, "--", color="#888888", alpha=0.7, lw=1.2)
    ax6.set_xlabel("Observed (m³/s)")
    ax6.set_ylabel("Predicted (m³/s)")
    ax6.set_title(f"Peak Flows (p95)\nBias: {peak_bias:+.1f}% | MAE: {peak_mae:.2f}")

    # Panel 7: Q-Q plot
    ax7 = fig.add_subplot(gs[2, 1])
    light_style(ax7)
    stats.probplot(pred_df["residual"], dist="norm", plot=ax7)
    ax7.get_lines()[0].set_markerfacecolor("#9467bd")
    ax7.get_lines()[0].set_markersize(4)
    ax7.get_lines()[0].set_alpha(0.5)
    ax7.get_lines()[1].set_color("#888888")
    ax7.set_title("Residual Q-Q Plot (Normality)")

    # Panel 8: flow duration curve
    ax8 = fig.add_subplot(gs[2, 2])
    light_style(ax8)
    obs_s = np.sort(pred_df["observed"].values)[::-1]
    sim_s = np.sort(pred_df["predicted"].values)[::-1]
    exc_o = np.arange(1, len(obs_s) + 1) / len(obs_s) * 100
    exc_s = np.arange(1, len(sim_s) + 1) / len(sim_s) * 100
    ax8.semilogy(
        exc_o, np.maximum(obs_s, 1e-3), color="#1f77b4", lw=1.5, label="Observed"
    )
    ax8.semilogy(
        exc_s, np.maximum(sim_s, 1e-3), color="#ff7f0e", lw=1.5, label="Predicted"
    )
    ax8.set_xlabel("Exceedance Probability (%)")
    ax8.set_ylabel("Discharge (m³/s)")
    ax8.set_title("Flow Duration Curve")
    ax8.legend(
        loc="upper right",
        facecolor="#ffffff",
        edgecolor="#cccccc",
        labelcolor="#333333",
    )

    # Panel 9: training history
    ax9 = fig.add_subplot(gs[3, :])
    light_style(ax9)
    ax9.plot(
        history_df["epoch"],
        history_df["train_loss"],
        color="#1f77b4",
        lw=1.5,
        label="Train loss",
    )
    ax9.plot(
        history_df["epoch"],
        history_df["val_loss"],
        color="#ff7f0e",
        lw=1.5,
        label="Val loss (EMA)",
    )
    ax9.plot(
        history_df["epoch"],
        history_df["val_mae_ema"],
        color="#2ca02c",
        lw=1.0,
        alpha=0.7,
        label="Val MAE (EMA)",
    )
    ax9.axvline(
        best_epoch,
        color="#2ca02c",
        ls="--",
        lw=1.2,
        alpha=0.7,
        label=f"Best epoch ({best_epoch})",
    )
    ax9.set_xlabel("Epoch")
    ax9.set_ylabel("Loss / MAE")
    ax9.set_title("Training History (log scale)")
    ax9.set_yscale("log")
    ax9.legend(
        loc="upper right",
        facecolor="#ffffff",
        edgecolor="#cccccc",
        labelcolor="#333333",
        ncol=2,
    )

    fig.suptitle(
        f"WatershedTCN — {run_id} | {input_dim} features × {seq_len}-day lookback | "
        f"TCN {cfg.channels_1}×{cfg.levels_1}→{cfg.channels_2}×{cfg.levels_2} "
        f"(k={cfg.kernel_size}, RF={model.receptive_field}) + Attn + EMA",
        color="#111111",
        fontsize=13,
        fontfamily="monospace",
        y=0.98,
    )

    fig_path = FIG_DIR / "tcn_results.png"
    plt.savefig(
        fig_path, dpi=150, bbox_inches="tight", facecolor="#ffffff", edgecolor="none"
    )
    plt.close()
    print(f"\nDashboard → {fig_path}")

    print(f"\n{'=' * 72}")
    nse_status = "PASS" if nse_val > 0.8 else "FAIL"
    kge_status = "PASS" if kge_val > 0.8 else "FAIL"
    print(
        f"TARGET CHECK:  NSE={nse_val:.3f} [{nse_status}]  |  "
        f"KGE={kge_val:.3f} [{kge_status}]"
    )
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
