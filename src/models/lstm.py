import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
ROOT      = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")
SEQ_DIR   = ROOT / "data" / "sequences"
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models"
FIG_DIR   = ROOT / "results" / "figures"
MET_DIR   = ROOT / "results" / "metrics"
PRED_DIR  = ROOT / "results" / "predictions"

for d in [MODEL_DIR / "trained", MODEL_DIR / "checkpoints",
          MODEL_DIR / "configs", FIG_DIR, MET_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CFG = {
    "model_name"    : "LSTM",
    "lookback"      : 30,
    "n_features"    : 16,
    "lstm_units"    : [128, 64],
    "dense_units"   : [32],
    "dropout"       : 0.3,
    "recurrent_drop": 0.2,
    "learning_rate" : 1e-3,
    "batch_size"    : 32,
    "epochs"        : 150,
    "patience"      : 20,
    "min_delta"     : 1e-5,
    "seed"          : 42,
}

tf.random.set_seed(CFG["seed"])
np.random.seed(CFG["seed"])

print("=" * 65)
print("  Nahr Ibrahim — LSTM")
print("=" * 65)
print(f"  Architecture : {CFG['lstm_units']} units")
print(f"  Dropout      : {CFG['dropout']} | Recurrent: {CFG['recurrent_drop']}")
print(f"  Learning rate: {CFG['learning_rate']}")
print(f"  Loss         : MSE")

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1/6] Loading sequences ...")

X_train = np.load(SEQ_DIR / "X_train.npy")
y_train = np.load(SEQ_DIR / "y_train.npy")
X_val   = np.load(SEQ_DIR / "X_val.npy")
y_val   = np.load(SEQ_DIR / "y_val.npy")
X_test  = np.load(SEQ_DIR / "X_test.npy")
y_test  = np.load(SEQ_DIR / "y_test.npy")

dates_train = np.load(SEQ_DIR / "dates_train.npy", allow_pickle=True)
dates_val   = np.load(SEQ_DIR / "dates_val.npy",   allow_pickle=True)
dates_test  = np.load(SEQ_DIR / "dates_test.npy",  allow_pickle=True)

scaler = pd.read_csv(SPLIT_DIR / "scaler_params.csv", index_col=0)
q_min  = scaler.loc["discharge_m3s", "min"]
q_max  = scaler.loc["discharge_m3s", "max"]

print(f"  X_train : {X_train.shape}")
print(f"  X_val   : {X_val.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  Q range : [{q_min:.3f}, {q_max:.3f}] m³/s")

# =============================================================================
# 2. METRICS
# =============================================================================
def nse_metric(y_true, y_pred):
    """Nash-Sutcliffe Efficiency as a Keras metric."""
    num = tf.reduce_sum(tf.square(y_true - y_pred))
    den = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - num / (den + tf.keras.backend.epsilon())

# =============================================================================
# 3. BUILD MODEL
# =============================================================================

print("\n[2/6] Building LSTM model ...")

def build_lstm(cfg: dict) -> tf.keras.Model:
    """
    Stacked LSTM with dropout regularization.
    Architecture:
        Input (30, 12)
          → LSTM(128, return_sequences=True) → Dropout(0.3)
          → LSTM(64,  return_sequences=False) → Dropout(0.3)
          → Dense(32) → ReLU → Dropout(0.15)
          → Dense(1)  → Linear
    """
    inputs = tf.keras.Input(
        shape=(cfg["lookback"], cfg["n_features"]),
        name="input_sequence"
    )

    x = inputs

    for i, units in enumerate(cfg["lstm_units"]):
        return_seq = (i < len(cfg["lstm_units"]) - 1)
        x = tf.keras.layers.LSTM(
            units,
            return_sequences   = return_seq,
            dropout            = cfg["dropout"],
            recurrent_dropout  = cfg["recurrent_drop"],
            kernel_regularizer = tf.keras.regularizers.l2(1e-4),
            name               = f"lstm_{i+1}"
        )(x)
        x = tf.keras.layers.Dropout(
            cfg["dropout"], name=f"dropout_lstm_{i+1}"
        )(x)

    for i, units in enumerate(cfg["dense_units"]):
        x = tf.keras.layers.Dense(units, name=f"dense_{i+1}")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu_{i+1}")(x)
        x = tf.keras.layers.Dropout(
            cfg["dropout"] / 2, name=f"dropout_dense_{i+1}"
        )(x)

    outputs = tf.keras.layers.Dense(
        1, activation="linear", name="output"
    )(x)

    return tf.keras.Model(
        inputs=inputs, outputs=outputs,
        name="LSTM_NahrIbrahim"
    )


model = build_lstm(CFG)
model.summary()

# =============================================================================
# 4. COMPILE
# =============================================================================
print("\n[3/6] Compiling ...")

optimizer = tf.keras.optimizers.Adam(
    learning_rate = CFG["learning_rate"],
    clipnorm      = 1.0
)

model.compile(
    optimizer = optimizer,
    loss      = "mse",
    metrics   = [
        nse_metric,
        "mae",
        tf.keras.metrics.RootMeanSquaredError(name="rmse")
    ]
)

# =============================================================================
# 5. CALLBACKS
# =============================================================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor             = "val_loss",
        mode                = "min",
        patience            = CFG["patience"],
        min_delta           = CFG["min_delta"],
        restore_best_weights= True,
        verbose             = 1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath       = str(MODEL_DIR / "checkpoints" / "lstm_best.keras"),
        monitor        = "val_loss",
        mode           = "min",
        save_best_only = True,
        verbose        = 0
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = "val_loss",
        mode     = "min",
        factor   = 0.5,
        patience = 10,
        min_lr   = 1e-6,
        verbose  = 1
    ),
    tf.keras.callbacks.CSVLogger(
        str(MODEL_DIR / "configs" / "lstm_training_log.csv"),
        append=False
    ),
]

# =============================================================================
# 6. TRAIN
# =============================================================================
print("\n[4/6] Training ...")
print(f"  Max epochs : {CFG['epochs']} | Patience : {CFG['patience']}")
print(f"  Batch size : {CFG['batch_size']}")
print()

start_time = datetime.now()

history = model.fit(
    X_train, y_train,
    validation_data = (X_val, y_val),
    epochs          = CFG["epochs"],
    batch_size      = CFG["batch_size"],
    callbacks       = callbacks,
    shuffle         = False,
    verbose         = 1
)

elapsed = datetime.now() - start_time
print(f"\n  Training complete in {elapsed}")

model.save(str(MODEL_DIR / "trained" / "lstm_final.keras"))
pd.DataFrame([CFG]).to_csv(
    MODEL_DIR / "configs" / "lstm_config.csv", index=False
)
print(f"  Model saved → models/trained/lstm_final.keras")

# =============================================================================
# 7. EVALUATION
# =============================================================================
print("\n[5/6] Evaluating ...")

def inverse_transform_q(q_norm, q_min, q_max):
    return q_norm * (q_max - q_min) + q_min


def compute_metrics(y_true_norm, y_pred_norm, q_min, q_max, split_name):
    y_true = inverse_transform_q(y_true_norm, q_min, q_max)
    y_pred = np.clip(
        inverse_transform_q(y_pred_norm.flatten(), q_min, q_max), 0, None
    )

    # NSE
    nse_val = 1 - np.sum((y_true - y_pred)**2) / \
                  np.sum((y_true - np.mean(y_true))**2)

    # KGE
    r       = np.corrcoef(y_true, y_pred)[0, 1]
    alpha   = np.std(y_pred)  / np.std(y_true)
    beta    = np.mean(y_pred) / np.mean(y_true)
    kge_val = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

    # RMSE & MAE
    rmse_val = np.sqrt(np.mean((y_true - y_pred)**2))
    mae_val  = np.mean(np.abs(y_true - y_pred))

    # PBIAS
    pbias_val = 100 * np.sum(y_pred - y_true) / np.sum(y_true)

    # Peak flow bias (top 5%)
    threshold = np.percentile(y_true, 95)
    peak_mask = y_true >= threshold
    peak_bias = 100 * (np.mean(y_pred[peak_mask]) -
                        np.mean(y_true[peak_mask])) / \
                       np.mean(y_true[peak_mask])

    # Log-NSE
    eps         = 0.001
    log_nse_val = 1 - np.sum(
        (np.log(y_true + eps) - np.log(y_pred + eps))**2
    ) / np.sum(
        (np.log(y_true + eps) - np.mean(np.log(y_true + eps)))**2
    )

    return {
        "split"       : split_name,
        "NSE"         : round(nse_val,   4),
        "KGE"         : round(kge_val,   4),
        "RMSE"        : round(rmse_val,  4),
        "MAE"         : round(mae_val,   4),
        "PBIAS_%"     : round(pbias_val, 2),
        "Peak_Bias_%" : round(peak_bias, 2),
        "Log_NSE"     : round(log_nse_val, 4),
        "r"           : round(r,     4),
        "alpha"       : round(alpha, 4),
        "beta"        : round(beta,  4),
    }, y_true, y_pred


y_pred_train = model.predict(X_train, batch_size=64, verbose=0)
y_pred_val   = model.predict(X_val,   batch_size=64, verbose=0)
y_pred_test  = model.predict(X_test,  batch_size=64, verbose=0)

m_train, qt_true, qt_pred = compute_metrics(
    y_train, y_pred_train, q_min, q_max, "Train")
m_val, qv_true, qv_pred = compute_metrics(
    y_val, y_pred_val, q_min, q_max, "Validation")
m_test, qtest_true, qtest_pred = compute_metrics(
    y_test, y_pred_test, q_min, q_max, "Test")

all_metrics = [m_train, m_val, m_test]
metrics_df  = pd.DataFrame(all_metrics)

print(f"\n  {'Metric':<14} {'Train':>10} {'Validation':>12} {'Test':>10}")
print(f"  {'-'*50}")
for metric in ["NSE", "KGE", "RMSE", "MAE",
               "PBIAS_%", "Peak_Bias_%", "Log_NSE"]:
    print(f"  {metric:<14} "
          f"{metrics_df[metrics_df.split=='Train'][metric].values[0]:>10} "
          f"{metrics_df[metrics_df.split=='Validation'][metric].values[0]:>12} "
          f"{metrics_df[metrics_df.split=='Test'][metric].values[0]:>10}")

metrics_df.to_csv(MET_DIR / "lstm_metrics.csv", index=False)

for sname, dates, yt, yp in [
    ("train", dates_train, qt_true,    qt_pred),
    ("val",   dates_val,   qv_true,    qv_pred),
    ("test",  dates_test,  qtest_true, qtest_pred),
]:
    pd.DataFrame({
        "date"          : pd.to_datetime(dates),
        "observed_m3s"  : yt,
        "predicted_m3s" : yp,
        "residual_m3s"  : yt - yp,
    }).to_csv(PRED_DIR / f"lstm_predictions_{sname}.csv", index=False)

print(f"\n  Metrics     → results/metrics/lstm_metrics.csv")
print(f"  Predictions → results/predictions/")

# =============================================================================
# 8. VISUALIZATION
# =============================================================================
print("\n[6/6] Generating plots ...")

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#080f1a")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

log        = pd.read_csv(MODEL_DIR / "configs" / "lstm_training_log.csv")
best_epoch = log["val_loss"].idxmin()

# ── Training history — NSE ──
ax0 = fig.add_subplot(gs[0, :2])
ax0.set_facecolor("#0d1825")
ax0.plot(log["epoch"], log["nse_metric"],
         color="#3b9eff", linewidth=1.8, label="Train NSE")
ax0.plot(log["epoch"], log["val_nse_metric"],
         color="#f4a261", linewidth=1.8, linestyle="--", label="Val NSE")
ax0.axvline(best_epoch, color="#00b4a0", linestyle=":",
            linewidth=1.5, label=f"Best epoch ({best_epoch})")
ax0.axhline(0.75, color="#00b4a0", linewidth=0.8,
            linestyle="--", alpha=0.4, label="Good (0.75)")
ax0.axhline(0, color="#4a6a82", linewidth=0.8, linestyle="--")
ax0.set_title("Training History — NSE", color="#e8f4f8", fontsize=11)
ax0.set_xlabel("Epoch", color="#8aafc4")
ax0.set_ylabel("NSE", color="#8aafc4")
ax0.tick_params(colors="#4a6a82")
ax0.spines[:].set_color("#1e3448")
ax0.legend(facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4")
ax0.set_facecolor("#0d1825")

# ── Training history — Loss ──
ax0b = fig.add_subplot(gs[0, 2])
ax0b.set_facecolor("#0d1825")
ax0b.plot(log["epoch"], log["loss"],
          color="#3b9eff", linewidth=1.5, label="Train MSE")
ax0b.plot(log["epoch"], log["val_loss"],
          color="#f4a261", linewidth=1.5, linestyle="--", label="Val MSE")
ax0b.set_title("Training Loss (MSE)", color="#e8f4f8", fontsize=11)
ax0b.set_xlabel("Epoch", color="#8aafc4")
ax0b.set_ylabel("MSE", color="#8aafc4")
ax0b.tick_params(colors="#4a6a82")
ax0b.spines[:].set_color("#1e3448")
ax0b.legend(facecolor="#0d1825", edgecolor="#1e3448",
            labelcolor="#8aafc4", fontsize=8)
ax0b.set_facecolor("#0d1825")

# ── Test hydrograph ──
ax1 = fig.add_subplot(gs[1, :])
ax1.set_facecolor("#0d1825")
dates_t = pd.to_datetime(dates_test)
ax1.plot(dates_t, qtest_true,
         color="#8aafc4", linewidth=1.2, label="Observed", alpha=0.9)
ax1.plot(dates_t, qtest_pred,
         color="#3b9eff", linewidth=1.2, linestyle="--",
         label="LSTM Predicted", alpha=0.9)
ax1.fill_between(dates_t, qtest_true, qtest_pred,
                 alpha=0.15, color="#3b9eff", label="Error")
ax1.set_title(
    f"Test Hydrograph (2021–2025) — "
    f"NSE={m_test['NSE']:.3f}  KGE={m_test['KGE']:.3f}  "
    f"Peak Bias={m_test['Peak_Bias_%']:.1f}%",
    color="#e8f4f8", fontsize=11
)
ax1.set_ylabel("Discharge (m³/s)", color="#8aafc4")
ax1.tick_params(colors="#4a6a82")
ax1.spines[:].set_color("#1e3448")
ax1.legend(facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4")
ax1.set_facecolor("#0d1825")

# ── Scatter ──
ax2 = fig.add_subplot(gs[2, 0])
ax2.set_facecolor("#0d1825")
ax2.scatter(qtest_true, qtest_pred,
            alpha=0.4, s=8, color="#3b9eff", edgecolors="none")
lim = max(qtest_true.max(), qtest_pred.max()) * 1.05
ax2.plot([0, lim], [0, lim], color="#e76f51",
         linewidth=1.5, linestyle="--", label="1:1 line")
ax2.set_xlabel("Observed (m³/s)", color="#8aafc4")
ax2.set_ylabel("Predicted (m³/s)", color="#8aafc4")
ax2.set_title("Observed vs Predicted", color="#e8f4f8", fontsize=11)
ax2.tick_params(colors="#4a6a82")
ax2.spines[:].set_color("#1e3448")
ax2.legend(facecolor="#0d1825", edgecolor="#1e3448",
           labelcolor="#8aafc4", fontsize=8)
ax2.set_facecolor("#0d1825")

# ── Residuals ──
ax3 = fig.add_subplot(gs[2, 1])
ax3.set_facecolor("#0d1825")
residuals = qtest_true - qtest_pred
ax3.hist(residuals, bins=50, color="#3b9eff", alpha=0.75, density=True)
ax3.axvline(0, color="#e76f51", linewidth=1.5, linestyle="--")
ax3.axvline(residuals.mean(), color="#f4a261", linewidth=1.2,
            linestyle=":", label=f"Mean={residuals.mean():.3f}")
ax3.set_xlabel("Residual (m³/s)", color="#8aafc4")
ax3.set_ylabel("Density", color="#8aafc4")
ax3.set_title("Residual Distribution", color="#e8f4f8", fontsize=11)
ax3.tick_params(colors="#4a6a82")
ax3.spines[:].set_color("#1e3448")
ax3.legend(facecolor="#0d1825", edgecolor="#1e3448",
           labelcolor="#8aafc4", fontsize=8)
ax3.set_facecolor("#0d1825")

# ── Metrics bar chart ──
ax4 = fig.add_subplot(gs[2, 2])
ax4.set_facecolor("#0d1825")
metric_names = ["NSE", "KGE", "Log_NSE"]
splits_plot  = ["Train", "Validation", "Test"]
colors_bar   = ["#3b9eff", "#f4a261", "#e76f51"]
x = np.arange(len(metric_names))
w = 0.25

for i, (split, color) in enumerate(zip(splits_plot, colors_bar)):
    row  = metrics_df[metrics_df.split == split]
    vals = [row[m].values[0] for m in metric_names]
    ax4.bar(x + i * w, vals, w, label=split, color=color, alpha=0.8)

ax4.axhline(0.75, color="#00b4a0", linewidth=1,
            linestyle=":", alpha=0.7, label="Good (0.75)")
ax4.set_xticks(x + w)
ax4.set_xticklabels(metric_names, color="#8aafc4", fontsize=9)
ax4.set_ylabel("Score", color="#8aafc4")
ax4.set_title("Performance Metrics", color="#e8f4f8", fontsize=11)
ax4.tick_params(colors="#4a6a82")
ax4.spines[:].set_color("#1e3448")
ax4.legend(facecolor="#0d1825", edgecolor="#1e3448",
           labelcolor="#8aafc4", fontsize=7)
ax4.set_ylim(-0.1, 1.1)
ax4.set_facecolor("#0d1825")

fig.suptitle("LSTM— Nahr Ibrahim Rainfall-Runoff Model",
             color="#e8f4f8", fontsize=14, y=0.98,
             fontfamily="monospace")

plt.savefig(FIG_DIR / "lstm_results.png",
            dpi=150, bbox_inches="tight", facecolor="#080f1a")
plt.show()

# =============================================================================
# 9. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  LSTM SUMMARY")
print("=" * 65)
print(f"\n  Best epoch  : {best_epoch}")
print(f"  Training time : {elapsed}")
print(f"\n  Test Performance:")
print(f"    NSE         : {m_test['NSE']:.4f}  (>0.75 = good)")
print(f"    KGE         : {m_test['KGE']:.4f}  (>0.75 = good)")
print(f"    RMSE        : {m_test['RMSE']:.4f} m³/s")
print(f"    MAE         : {m_test['MAE']:.4f} m³/s")
print(f"    PBIAS       : {m_test['PBIAS_%']:.2f}%  (|<10%| = good)")
print(f"    Peak Bias   : {m_test['Peak_Bias_%']:.2f}%")
print(f"    Log-NSE     : {m_test['Log_NSE']:.4f}")
print(f"\n  Files saved:")
print(f"    models/trained/lstm_final.keras")
print(f"    models/checkpoints/lstm_best.keras")
print(f"    models/configs/lstm_config.csv")
print(f"    models/configs/lstm_training_log.csv")
print(f"    results/metrics/lstm_metrics.csv")
print(f"    results/predictions/lstm_predictions_*.csv")
print(f"    results/figures/lstm_results.png")
print("=" * 65)