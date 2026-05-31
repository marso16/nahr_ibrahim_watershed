"""
Train and evaluate the GRU baseline. Mirrors lstm.py and mlp.py structure
so results are directly comparable.

Usage:
  python src/models/gru.py --horizon 1 --lookback 30 --run_tag gru_strategy_a_h1
"""

import os
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from gru_model import WatershedGRU


# ═══════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════
class NSELoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        var = ((y_true - y_true.mean()) ** 2).sum() + self.eps
        sq_err = ((y_pred - y_true) ** 2).sum()
        return sq_err / var


# ═══════════════════════════════════════════════════════════════════════════
# EMA
# ═══════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════
# Hydrological metrics
# ═══════════════════════════════════════════════════════════════════════════
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
    return (
        1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2),
        r,
        alpha,
        beta,
    )


def nse(obs, sim):
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    denom = np.sum((obs - obs.mean()) ** 2)
    return np.nan if denom < 1e-12 else 1 - np.sum((obs - sim) ** 2) / denom


def log_nse(obs, sim, eps=1e-3):
    obs = np.maximum(obs, 0) + eps
    sim = np.maximum(sim, 0) + eps
    lo, ls = np.log(obs), np.log(sim)
    denom = np.sum((lo - lo.mean()) ** 2)
    return np.nan if denom < 1e-12 else 1 - np.sum((lo - ls) ** 2) / denom


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
def get_config():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        default=os.environ.get(
            "WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed"
        ),
    )
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--lookback", type=int, default=30)

    # GRU architecture
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.30)

    # Training (mirrors lstm.py defaults for fair comparison)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--use_nse_loss", action="store_true", default=True)
    p.add_argument("--huber_delta", type=float, default=0.15)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--run_tag", type=str, default=None)
    return p.parse_args()


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    cfg = get_config()
    set_seed(cfg.seed)
    tag = cfg.run_tag if cfg.run_tag else f"gru_seed{cfg.seed}_h{cfg.horizon}"

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
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load sequences (same arrays as lstm.py uses) ──────────────────────────
    print("\nLoading sequences...")
    suffix = f"_h{cfg.horizon}_lb{cfg.lookback}"
    X_train = np.load(SEQ_DIR / f"X_train{suffix}.npy")
    y_train = np.load(SEQ_DIR / f"y_train{suffix}.npy")
    X_val = np.load(SEQ_DIR / f"X_val{suffix}.npy")
    y_val = np.load(SEQ_DIR / f"y_val{suffix}.npy")
    X_test = np.load(SEQ_DIR / f"X_test{suffix}.npy")
    y_test = np.load(SEQ_DIR / f"y_test{suffix}.npy")
    dates_test = np.load(SEQ_DIR / f"dates_test{suffix}.npy", allow_pickle=True)

    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    input_dim = X_train.shape[2]

    # ── Dataloaders ──────────────────────────────────────────────────────────
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

    # ── Model ────────────────────────────────────────────────────────────────
    model = WatershedGRU(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\nModel: WatershedGRU hidden={cfg.hidden_dim} layers={cfg.num_layers} | "
        f"Parameters: {n_params:,}"
    )

    criterion = (
        NSELoss().to(device)
        if cfg.use_nse_loss
        else nn.SmoothL1Loss(beta=cfg.huber_delta).to(device)
    )
    print(
        f"  Loss: {'NSE-loss' if cfg.use_nse_loss else f'Huber(δ={cfg.huber_delta})'}"
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999)
    )
    plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    scaler = GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = ModelEMA(model, decay=cfg.ema_decay)

    # ── Training loop ────────────────────────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'=' * 72}\nTraining run {run_id} (GRU h={cfg.horizon})\n{'=' * 72}\n")

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
        if epoch <= cfg.warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * (epoch / cfg.warmup_epochs)

        model.train()
        train_loss_sum, train_n = 0.0, 0
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
            train_loss_sum += loss.item() * yb.size(0)
            train_n += yb.size(0)
        train_loss = train_loss_sum / max(train_n, 1)

        model.eval()
        val_loss_sum, val_mae_sum, val_n = 0.0, 0.0, 0
        with ema.apply_to(model), torch.no_grad():
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

        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_ema"].append(val_mae)
        history["lr"].append(current_lr)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  Epoch {epoch:3d}/{cfg.epochs} | train={train_loss:.5f}  "
                f"val={val_loss:.5f}  val_mae(ema)={val_mae:.5f}  lr={current_lr:.2e}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": ema.shadow,
                    "val_loss": val_loss,
                    "config": vars(cfg),
                },
                MODEL_DIR / "checkpoints" / f"gru_best_{tag}.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            print(
                f"\n  Early stopping at epoch {epoch} (best epoch {best_epoch}, val={best_val:.5f})"
            )
            break

    train_time = time.time() - start
    print(f"\nTraining time: {train_time/60:.1f} min")
    print(f"Best val loss: {best_val:.6f} @ epoch {best_epoch}")

    # ── Load best (EMA) weights ──────────────────────────────────────────────
    ckpt = torch.load(
        MODEL_DIR / "checkpoints" / f"gru_best_{tag}.pt", weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    torch.save(model.state_dict(), MODEL_DIR / "trained" / f"gru_final_{tag}.pt")
    pd.DataFrame(history).to_csv(
        MODEL_DIR / "configs" / f"gru_training_log_{tag}.csv", index=False
    )

    # ── Evaluate on test ─────────────────────────────────────────────────────
    print(f"\n{'=' * 72}\nEvaluation on test set\n{'=' * 72}")
    y_pred_list = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device, non_blocking=True)
            with autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                pred = model(xb)
            y_pred_list.append(pred.float().cpu().numpy())
    y_pred = np.concatenate(y_pred_list).flatten()

    # Denormalise and invert log
    scaler_df = pd.read_csv(ROOT / "data" / "splits" / "scaler_params.csv", index_col=0)
    q_min = float(scaler_df.loc["discharge_m3s", "min"])
    q_max = float(scaler_df.loc["discharge_m3s", "max"])
    if "__meta__" in scaler_df.index:
        log_transform = bool(float(scaler_df.loc["__meta__", "min"]))
        log_eps = float(scaler_df.loc["__meta__", "max"])
    else:
        log_transform = False
        log_eps = 0.0

    y_test_lin = y_test * (q_max - q_min) + q_min
    y_pred_lin = y_pred * (q_max - q_min) + q_min

    if log_transform:
        y_test_real = np.maximum(np.exp(y_test_lin) - log_eps, 0.0)
        y_pred_real = np.maximum(np.exp(y_pred_lin) - log_eps, 0.0)
    else:
        y_test_real = y_test_lin
        y_pred_real = y_pred_lin

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

    # Save metrics
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
    ).to_csv(MET_DIR / f"gru_metrics_{tag}.csv", index=False)

    pd.DataFrame(
        {
            "date": pd.to_datetime(dates_test),
            "observed": y_test_real,
            "predicted": y_pred_real,
            "residual": y_test_real - y_pred_real,
            "is_peak": peak_mask,
        }
    ).to_csv(PRED_DIR / f"gru_predictions_test_{tag}.csv", index=False)

    print(f"\nMetrics    → results/metrics/gru_metrics_{tag}.csv")
    print(f"Predictions → results/predictions/gru_predictions_test_{tag}.csv")

    # Quick comparison plot
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(
        pd.to_datetime(dates_test),
        y_test_real,
        color="#1f77b4",
        lw=1.0,
        label="Observed",
    )
    ax.plot(
        pd.to_datetime(dates_test),
        y_pred_real,
        color="#2ca02c",
        lw=1.0,
        alpha=0.8,
        label="GRU Predicted",
    )
    ax.set_title(
        f"GRU {tag} — NSE={nse_val:.3f} | KGE={kge_val:.3f} | MAE={mae_real:.2f} m³/s"
    )
    ax.set_ylabel("Discharge (m³/s)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"gru_results_{tag}.png", dpi=130, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
