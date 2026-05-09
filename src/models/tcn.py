import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")
SEQ_DIR = ROOT / "data" / "sequences"
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models" / "trained"
FIG_DIR = ROOT / "results" / "figures"
MET_DIR = ROOT / "results" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

# ── Load sequences ─────────────────────────────────────────────────────────────
print("Loading sequences ...")
X_train = np.load(SEQ_DIR / "X_train.npy")
y_train = np.load(SEQ_DIR / "y_train.npy")
X_val = np.load(SEQ_DIR / "X_val.npy")
y_val = np.load(SEQ_DIR / "y_val.npy")
X_test = np.load(SEQ_DIR / "X_test.npy")
y_test = np.load(SEQ_DIR / "y_test.npy")
dates = np.load(SEQ_DIR / "dates_test.npy", allow_pickle=True)
dates = pd.to_datetime(dates)

print(f"  Train {X_train.shape}  Val {X_val.shape}  Test {X_test.shape}\n")

# ── Scaler ─────────────────────────────────────────────────────────────────────
scaler = pd.read_csv(SPLIT_DIR / "scaler_params.csv", index_col=0)
q_min = scaler.loc["discharge_m3s", "min"]
q_max = scaler.loc["discharge_m3s", "max"]


def inverse_q(q_norm):
    return np.clip(q_norm * (q_max - q_min) + q_min, 0, None)


# ── Metrics ────────────────────────────────────────────────────────────────────
def nse_metric(y_true, y_pred):
    num = tf.reduce_sum(tf.square(y_true - y_pred))
    den = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - num / (den + tf.keras.backend.epsilon())


def compute_metrics(obs, pred):
    nse = 1 - np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    r = pearsonr(obs, pred)[0]
    alpha = np.std(pred) / np.std(obs)
    beta = np.mean(pred) / np.mean(obs)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    mae = np.mean(np.abs(obs - pred))
    pbias = 100 * np.sum(pred - obs) / np.sum(obs)
    p95 = np.percentile(obs, 95)
    mask = obs >= p95
    peak = 100 * (np.mean(pred[mask]) - np.mean(obs[mask])) / np.mean(obs[mask])
    return {
        "NSE": round(nse, 4),
        "KGE": round(kge, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "PBIAS_%": round(pbias, 2),
        "Peak_Bias_%": round(peak, 2),
    }


# ── TCN architecture ───────────────────────────────────────────────────────────
# Temporal Convolutional Network using dilated causal convolutions.
# Dilation grows exponentially (1, 2, 4, 8) so the receptive field covers
# the full 30-day lookback window with relatively few parameters.
# Residual connections stabilise gradients across deep stacks.
def build_tcn(input_shape, n_filters=64, kernel_size=3, n_stacks=4, dropout=0.2):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for stack in range(n_stacks):
        dilation = 2**stack  # 1, 2, 4, 8
        residual = x

        x = tf.keras.layers.Conv1D(
            n_filters,
            kernel_size,
            dilation_rate=dilation,
            padding="causal",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(
            n_filters,
            kernel_size,
            dilation_rate=dilation,
            padding="causal",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(x)
        x = tf.keras.layers.Dropout(dropout)(x)

        # Match dimensions for residual addition
        if residual.shape[-1] != n_filters:
            residual = tf.keras.layers.Conv1D(n_filters, 1, padding="same")(residual)

        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, output, name="TCN")


# ── Build and train ────────────────────────────────────────────────────────────
print("Building TCN ...")
model = build_tcn(input_shape=(X_train.shape[1], X_train.shape[2]))
model.summary()
print(f"\n  Parameters: {model.count_params():,}\n")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
    loss="mse",
    metrics=[nse_metric],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=10, factor=0.5, min_lr=1e-5, verbose=1
    ),
]

print("Training ...")
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluate ───────────────────────────────────────────────────────────────────
print("\nEvaluating on test set ...")
y_pred_norm = model.predict(X_test, batch_size=512, verbose=0).flatten()
y_pred = inverse_q(y_pred_norm)
y_obs = inverse_q(y_test)

metrics = compute_metrics(y_obs, y_pred)
print(f"\n  TCN — Test Period 2021–2025")
print(f"  NSE       : {metrics['NSE']}")
print(f"  KGE       : {metrics['KGE']}")
print(f"  RMSE      : {metrics['RMSE']} m³/s")
print(f"  MAE       : {metrics['MAE']} m³/s")
print(f"  PBIAS     : {metrics['PBIAS_%']}%")
print(f"  Peak Bias : {metrics['Peak_Bias_%']}%")

# Save metrics
pd.DataFrame([{"model": "TCN", **metrics}]).to_csv(
    MET_DIR / "tcn_metrics.csv", index=False
)

# ── Save model ─────────────────────────────────────────────────────────────────
model.save(MODEL_DIR / "tcn_final.keras")
print(f"\n  Saved → models/trained/tcn_final.keras")

# ── Figures ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#080f1a")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

# Training history
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_facecolor("#0d1825")
ax0.plot(history.history["loss"], color="#3b9eff", linewidth=1.5, label="Train loss")
ax0.plot(history.history["val_loss"], color="#e76f51", linewidth=1.5, label="Val loss")
ax0.set_title("Training history", color="#e8f4f8", fontsize=11)
ax0.set_xlabel("Epoch", color="#8aafc4")
ax0.set_ylabel("MSE loss", color="#8aafc4")
ax0.tick_params(colors="#4a6a82")
ax0.spines[:].set_color("#1e3448")
ax0.legend(facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4")
ax0.grid(alpha=0.08)

# Scatter
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_facecolor("#0d1825")
ax1.scatter(y_obs, y_pred, alpha=0.3, s=6, color="#3b9eff")
lim = max(y_obs.max(), y_pred.max()) * 1.05
ax1.plot(
    [0, lim], [0, lim], color="#e76f51", linewidth=1.5, linestyle="--", label="1:1 line"
)
ax1.set_title(
    f"Observed vs Predicted\nNSE={metrics['NSE']}  KGE={metrics['KGE']}",
    color="#e8f4f8",
    fontsize=11,
)
ax1.set_xlabel("Observed Q (m³/s)", color="#8aafc4")
ax1.set_ylabel("Predicted Q (m³/s)", color="#8aafc4")
ax1.tick_params(colors="#4a6a82")
ax1.spines[:].set_color("#1e3448")
ax1.legend(facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4")
ax1.grid(alpha=0.08)

# Time series — full test period
ax2 = fig.add_subplot(gs[1, :])
ax2.set_facecolor("#0d1825")
ax2.plot(
    dates, y_obs, color="#ffffff", linewidth=1.0, alpha=0.7, label="Observed (GloFAS)"
)
ax2.plot(
    dates, y_pred, color="#3b9eff", linewidth=1.2, alpha=0.85, label="TCN predicted"
)
ax2.fill_between(dates, y_obs, y_pred, alpha=0.12, color="#e76f51", label="Error")
ax2.set_title("Discharge time series — Test 2021–2025", color="#e8f4f8", fontsize=11)
ax2.set_ylabel("Discharge (m³/s)", color="#8aafc4")
ax2.tick_params(colors="#4a6a82")
ax2.spines[:].set_color("#1e3448")
ax2.legend(facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4", fontsize=8)
ax2.grid(alpha=0.08)

fig.suptitle(
    f"TCN — Nahr Ibrahim Watershed\n"
    f"NSE={metrics['NSE']}  KGE={metrics['KGE']}  "
    f"RMSE={metrics['RMSE']} m³/s",
    color="#e8f4f8",
    fontsize=13,
    y=1.01,
    fontfamily="monospace",
)
plt.savefig(
    FIG_DIR / "tcn_results.png", dpi=150, bbox_inches="tight", facecolor="#080f1a"
)
plt.show()
print(f"  Figure saved → results/figures/tcn_results.png")
print(f"\n{'=' * 55}")
print(f"  TCN complete  NSE={metrics['NSE']}  KGE={metrics['KGE']}")
print(f"{'=' * 55}")
