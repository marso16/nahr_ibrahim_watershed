import os
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
# Loss
# ═══════════════════════════════════════════════════════════════════════════════
class NSELoss(nn.Module):
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

    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--lookback", type=int, default=30)
    p.add_argument(
        "--root",
        type=str,
        default=os.environ.get(
            "WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed"
        ),
        help="Project root directory",
    )

    # TCN architecture
    p.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="TCN hidden channels (alias: tcn_base_channels)",
    )
    p.add_argument(
        "--num_layers", type=int, default=3, help="Number of TCN dilated blocks"
    )
    p.add_argument(
        "--tcn_base_channels",
        type=int,
        default=None,
        help="Deprecated alias for --hidden_dim",
    )
    p.add_argument(
        "--tcn_levels", type=int, default=None, help="Deprecated alias for --num_layers"
    )
    p.add_argument("--kernel_size", type=int, default=3, help="TCN kernel size")
    p.add_argument("--attention_dim", type=int, default=128)
    p.add_argument("--attention_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.20)

    # Training
    p.add_argument("--lr", type=float, default=0.000125)
    p.add_argument("--weight_decay", type=float, default=0.000234)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--ema_decay", type=float, default=0.999)

    # Loss options
    p.add_argument("--use_peak_loss", action="store_true", default=False)
    p.add_argument("--use_nse_loss", action="store_true", default=True)
    p.add_argument("--peak_percentile", type=float, default=85.0)
    p.add_argument("--peak_loss_weight", type=float, default=2.5)
    p.add_argument("--huber_delta", type=float, default=0.15)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional tag appended to all output filenames. "
        "If omitted, defaults to 'seed{seed}'.",
    )
    p.add_argument(
        "--seq_suffix",
        type=str,
        default="",
        help="Extra suffix on sequence files, e.g. '_hybrid'",
    )
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
# TCN Components
# ═══════════════════════════════════════════════════════════════════════════════
class Chomp1d(nn.Module):
    """Remove the extra padding from the right to preserve causality."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBackbone(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size, dropout):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            dilation = 2**i
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        return self.network(x)  # (B, C_out, T)


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


class WatershedTCN(nn.Module):
    def __init__(
        self,
        input_dim,
        tcn_channels=None,
        kernel_size=3,
        attention_dim=128,
        attention_heads=4,
        dropout=0.20,
        hidden_dim=None,
        dilations=None,
    ):
        # Support both tcn_channels (training script) and
        # hidden_dim+dilations (Optuna script) interfaces
        if tcn_channels is None:
            if hidden_dim is not None and dilations is not None:
                tcn_channels = [hidden_dim] * len(dilations)
            else:
                raise ValueError(
                    "Provide either tcn_channels or both hidden_dim and dilations"
                )
        super().__init__()
        self.tcn = TCNBackbone(input_dim, tcn_channels, kernel_size, dropout)
        out_channels = tcn_channels[-1]

        # Additive attention over TCN outputs
        self.attention = AdditiveAttention(
            out_channels, attention_dim, dropout=dropout / 2
        )

        # Residual projection from last input timestep
        self.input_residual = nn.Sequential(
            nn.Linear(input_dim, out_channels),
            nn.SiLU(),
        )

        # Fusion: last_step + attention + mha_pool + residual
        fusion_dim = out_channels * 4

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

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, T, F)
        h = self.tcn(x)  # (B, C, T)
        h = h.transpose(1, 2)  # (B, T, C)

        mha_pool = h.mean(dim=1)

        # Additive attention on h
        attn_context, _ = self.attention(h)

        # Last step of h
        last_step = h[:, -1, :]

        # Residual from last input timestep
        residual = self.input_residual(x[:, -1, :])

        # Fuse
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

    obs_std, sim_std = obs.std(), obs.std()
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
    if cfg.tcn_base_channels is not None:
        cfg.hidden_dim = cfg.tcn_base_channels
    if cfg.tcn_levels is not None:
        cfg.num_layers = cfg.tcn_levels

    set_seed(cfg.seed)

    tag = cfg.run_tag if cfg.run_tag else f"seed{cfg.seed}"

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
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(
        f"  Split:   train={len(X_train)/total*100:.1f}% "
        f"val={len(X_val)/total*100:.1f}% test={len(X_test)/total*100:.1f}%"
    )

    assert len(X_train) / total > 0.3, "training set seems too small"
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
    tcn_channels = [cfg.hidden_dim] * cfg.num_layers
    model = WatershedTCN(
        input_dim=input_dim,
        tcn_channels=tcn_channels,
        kernel_size=cfg.kernel_size,
        attention_dim=cfg.attention_dim,
        attention_heads=cfg.attention_heads,
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,} (trainable: {n_trainable:,})")

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
        print(f"  Loss: plain Huber (δ={cfg.huber_delta})")
        criterion = nn.SmoothL1Loss(beta=cfg.huber_delta).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50, min_lr=1e-6
    )

    scaler = GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = ModelEMA(model, decay=cfg.ema_decay)

    # ── Training loop ─────────────────────────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'=' * 72}\nTraining run {run_id}\n{'=' * 72}")
    print(
        f"  Arch: TCN {cfg.hidden_dim}×{cfg.num_layers} (k={cfg.kernel_size}) "
        f"+ AddAttn({cfg.attention_dim}) + MeanPool"
    )
    print(f"  Dropout: spatial={cfg.dropout} | WD: {cfg.weight_decay}")
    print(
        f"  LR: {cfg.lr} (warmup={cfg.warmup_epochs}, then ReduceLROnPlateau) | "
        f"Batch: {cfg.batch_size} | EMA: {cfg.ema_decay}"
    )
    if cfg.use_peak_loss:
        print(
            f"  Loss: Huber(δ={cfg.huber_delta}) with peak weight={cfg.peak_loss_weight}× "
            f"above p{cfg.peak_percentile:.0f}"
        )
    elif cfg.use_nse_loss:
        print(f"  Loss: NSE-loss (batch-wise)")
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
                MODEL_DIR / "checkpoints" / f"tcn_best_{tag}.pt",
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
    ckpt = torch.load(
        MODEL_DIR / "checkpoints" / f"tcn_best_{tag}.pt", weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── History CSV ───────────────────────────────────────────────────────────
    history_df = pd.DataFrame(history)
    history_df.to_csv(
        MODEL_DIR / "configs" / f"tcn_training_log_{tag}.csv", index=False
    )

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

    # Denormalise (and invert log-transform if it was applied in windowing.py)
    scaler_path = ROOT / "data" / "splits" / f"scaler_params{suffix}.csv"
    scaler_df = pd.read_csv(scaler_path, index_col=0)

    if "__target__" in scaler_df.index:
        q_min = float(scaler_df.loc["__target__", "min"])
        q_max = float(scaler_df.loc["__target__", "max"])
    else:
        q_min = float(scaler_df.loc["discharge_m3s", "min"])
        q_max = float(scaler_df.loc["discharge_m3s", "max"])

    # Read log-transform metadata from window_meta.json (new format) or CSV __meta__ (old format)
    meta_path = ROOT / "data" / "splits" / f"window_meta{suffix}.json"
    log_transform = False
    log_eps = 0.0
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        log_transform = meta.get("log_transform", False)
        log_eps = meta.get("log_eps", 0.0)
    elif "__meta__" in scaler_df.index:
        log_transform = bool(float(scaler_df.loc["__meta__", "min"]))
        log_eps = float(scaler_df.loc["__meta__", "max"])

    y_test_lin = y_test * (q_max - q_min) + q_min
    y_pred_lin = y_pred * (q_max - q_min) + q_min

    if log_transform:
        print(f"  Inverting log-transform (eps={log_eps})")
        y_test_real = np.exp(y_test_lin) - log_eps
        y_pred_real = np.exp(y_pred_lin) - log_eps
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
    print(f"  NSE:    {nse_val:.4f}")
    print(f"  KGE:    {kge_val:.4f}")
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
    ).to_csv(MET_DIR / f"tcn_metrics_{tag}.csv", index=False)

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
    pred_df.to_csv(PRED_DIR / f"tcn_predictions_test_{tag}.csv", index=False)

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

    print(f"\n{'=' * 72}")
    print(f"TARGET CHECK:  NSE={nse_val:.3f}  |  " f"KGE={kge_val:.3f}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
