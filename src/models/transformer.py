"""
=============================================================================
Nahr Ibrahim Watershed — Transformer Model
=============================================================================
Architecture : Temporal Transformer with multi-head self-attention
Framework    : TensorFlow / Keras (CPU-optimized)
Input        : (samples, 30 timesteps, 12 features)
Output       : 1-day ahead discharge (normalized)

Key differences from LSTM:
  - No recurrent connections — processes all timesteps in parallel
  - Multi-head attention captures long-range temporal dependencies
  - Positional encoding injects time-order information
  - Layer normalization instead of batch normalization
  - Feed-forward sublayers after each attention block
=============================================================================
"""
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
    "model_name"      : "Transformer",
    "lookback"        : 30,
    "n_features"      : 15,
    # Transformer-specific
    "d_model"         : 64,      # embedding dimension (must be divisible by n_heads)
    "n_heads"         : 4,       # number of attention heads
    "n_encoder_blocks": 3,       # number of transformer encoder blocks
    "ffn_dim"         : 128,     # feed-forward network inner dimension
    "dropout"         : 0.2,
    # Dense head
    "dense_units"     : [64, 32],
    # Training
    "learning_rate"   : 1e-3,
    "warmup_epochs"   : 10,
    "batch_size"      : 32,
    "epochs"          : 300,
    "patience"        : 40,
    "min_delta"       : 1e-6,
    "peak_weight"     : 3.0,
    "peak_threshold"  : 0.75,
    "log_transform"   : True,
    "seed"            : 42,
}

tf.random.set_seed(CFG["seed"])
np.random.seed(CFG["seed"])

print("=" * 65)
print("  Nahr Ibrahim — Transformer Model")
print("=" * 65)
print(f"  d_model       : {CFG['d_model']}")
print(f"  Attention heads: {CFG['n_heads']}")
print(f"  Encoder blocks : {CFG['n_encoder_blocks']}")
print(f"  FFN dim        : {CFG['ffn_dim']}")
print(f"  Log transform  : {CFG['log_transform']}")
print(f"  Peak weight    : {CFG['peak_weight']}x")

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

print(f"  X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

# =============================================================================
# 2. LOG-TRANSFORM TARGET
# =============================================================================
EPS = 1e-6

def log_transform(y):
    return np.log1p(y / (EPS + 1))

def inverse_log_transform(y_log):
    return np.expm1(y_log) * (EPS + 1)

if CFG["log_transform"]:
    y_train_model = log_transform(y_train).astype(np.float32)
    y_val_model   = log_transform(y_val).astype(np.float32)
    y_test_model  = log_transform(y_test).astype(np.float32)
    print("  Log-transform applied to targets")
else:
    y_train_model = y_train
    y_val_model   = y_val
    y_test_model  = y_test

# =============================================================================
# 3. LOSS & METRICS
# =============================================================================
def peak_weighted_mse(y_true, y_pred):
    base_loss = tf.square(y_true - y_pred)
    peak_mask = tf.cast(y_true > CFG["peak_threshold"], tf.float32)
    weights   = 1.0 + (CFG["peak_weight"] - 1.0) * peak_mask
    return tf.reduce_mean(weights * base_loss)

def nse_metric(y_true, y_pred):
    num = tf.reduce_sum(tf.square(y_true - y_pred))
    den = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - num / (den + tf.keras.backend.epsilon())

# =============================================================================
# 4. LEARNING RATE SCHEDULE
# =============================================================================
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps):
        super().__init__()
        self.peak_lr      = peak_lr
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)

    def __call__(self, step):
        step      = tf.cast(step, tf.float32)
        warmup_lr = self.peak_lr * (step / self.warmup_steps)
        cosine_lr = self.peak_lr * 0.5 * (
            1.0 + tf.cos(np.pi * (step - self.warmup_steps) /
                         (self.total_steps - self.warmup_steps))
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr"      : self.peak_lr,
            "warmup_steps" : self.warmup_steps,
            "total_steps"  : self.total_steps,
        }

steps_per_epoch = len(X_train) // CFG["batch_size"]
total_steps     = CFG["epochs"] * steps_per_epoch
warmup_steps    = CFG["warmup_epochs"] * steps_per_epoch

lr_schedule = WarmupCosineDecay(
    peak_lr      = CFG["learning_rate"],
    warmup_steps = warmup_steps,
    total_steps  = total_steps,
)

# =============================================================================
# 5. TRANSFORMER BUILDING BLOCKS
# =============================================================================
class PositionalEncoding(tf.keras.layers.Layer):
    """
    Injects position information into the sequence.
    Without this, the Transformer treats all timesteps as unordered.
    Uses sine/cosine functions at different frequencies (Vaswani et al. 2017).
    """
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

        # Precompute positional encoding matrix
        positions = np.arange(max_len)[:, np.newaxis]       # (max_len, 1)
        dims      = np.arange(d_model)[np.newaxis, :]       # (1, d_model)
        angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)

        # Apply sin to even indices, cos to odd indices
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        self.pos_encoding = tf.cast(
            angles[np.newaxis, :, :], dtype=tf.float32
        )  # (1, max_len, d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """
    Single Transformer encoder block:
      1. Multi-head self-attention (each head attends to different temporal patterns)
      2. Add & Layer Norm (residual connection)
      3. Feed-forward network (point-wise dense layers)
      4. Add & Layer Norm (residual connection)

    For hydrology: attention heads can specialize —
      one head for precipitation patterns, another for temperature trends, etc.
    """
    def __init__(self, d_model, n_heads, ffn_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.ffn_dim  = ffn_dim
        self.dropout  = dropout

        # Multi-head self-attention
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads   = n_heads,
            key_dim     = d_model // n_heads,
            dropout     = dropout,
        )

        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ffn_dim, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
        ])

        # Layer normalization (more stable than batch norm for sequences)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward with residual connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model" : self.d_model,
            "n_heads" : self.n_heads,
            "ffn_dim" : self.ffn_dim,
            "dropout" : self.dropout,
        })
        return config

# =============================================================================
# 6. BUILD TRANSFORMER MODEL
# =============================================================================
print("\n[2/6] Building Transformer model ...")

def build_transformer(cfg: dict) -> tf.keras.Model:
    """
    Temporal Transformer for rainfall-runoff modeling.

    Architecture:
      Input (30, 12)
        → Linear projection to d_model (30, 64)
        → Positional encoding (30, 64)
        → N × TransformerEncoderBlock (30, 64)
        → Global average pooling (64,)
        → Dense head → Output (1,)

    The global average pooling aggregates temporal context
    across all 30 timesteps after attention processing.
    """
    inputs = tf.keras.Input(
        shape=(cfg["lookback"], cfg["n_features"]),
        name="input_sequence"
    )

    # ── Project input features to d_model dimensions ──────────
    # The Transformer requires all layers to have the same width (d_model)
    x = tf.keras.layers.Dense(
        cfg["d_model"],
        name="input_projection"
    )(inputs)

    # ── Positional encoding ────────────────────────────────────
    x = PositionalEncoding(
        max_len = cfg["lookback"],
        d_model = cfg["d_model"],
        name    = "positional_encoding"
    )(x)

    x = tf.keras.layers.Dropout(cfg["dropout"], name="input_dropout")(x)

    # ── Stacked Transformer encoder blocks ────────────────────
    for i in range(cfg["n_encoder_blocks"]):
        x = TransformerEncoderBlock(
            d_model = cfg["d_model"],
            n_heads = cfg["n_heads"],
            ffn_dim = cfg["ffn_dim"],
            dropout = cfg["dropout"],
            name    = f"encoder_block_{i+1}"
        )(x)

    # ── Aggregate temporal dimension ──────────────────────────
    # Global average pooling across all 30 timesteps
    x = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # ── Dense prediction head ─────────────────────────────────
    for i, units in enumerate(cfg["dense_units"]):
        x = tf.keras.layers.Dense(
            units,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name=f"dense_{i+1}"
        )(x)
        x = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"ln_dense_{i+1}"
        )(x)
        x = tf.keras.layers.Activation("relu", name=f"relu_{i+1}")(x)
        x = tf.keras.layers.Dropout(
            cfg["dropout"] / 2, name=f"drop_dense_{i+1}"
        )(x)

    outputs = tf.keras.layers.Dense(
        1, activation="linear", name="output"
    )(x)

    model = tf.keras.Model(
        inputs  = inputs,
        outputs = outputs,
        name    = "Transformer_NahrIbrahim"
    )
    return model


model = build_transformer(CFG)
model.summary()

# =============================================================================
# 7. COMPILE & CALLBACKS
# =============================================================================
print("\n[3/6] Compiling ...")

optimizer = tf.keras.optimizers.Adam(
    learning_rate = lr_schedule,
    clipnorm      = 1.0
)

model.compile(
    optimizer = optimizer,
    loss      = peak_weighted_mse,
    metrics   = [nse_metric, "mae",
                 tf.keras.metrics.RootMeanSquaredError(name="rmse")]
)

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
        filepath        = str(MODEL_DIR / "checkpoints" / "transformer_best.keras"),
        monitor         = "val_loss",
        mode            = "min",
        save_best_only  = True,
        verbose         = 0
    ),
    tf.keras.callbacks.CSVLogger(
        str(MODEL_DIR / "configs" / "transformer_training_log.csv"),
        append=False
    ),
]

# =============================================================================
# 8. TRAIN
# =============================================================================
print("\n[4/6] Training Transformer ...")
print(f"  Max epochs : {CFG['epochs']} | Patience : {CFG['patience']}")
print(f"  Warmup     : {CFG['warmup_epochs']} epochs then cosine decay")
print()

start_time = datetime.now()

history = model.fit(
    X_train, y_train_model,
    validation_data = (X_val, y_val_model),
    epochs          = CFG["epochs"],
    batch_size      = CFG["batch_size"],
    callbacks       = callbacks,
    shuffle         = False,
    verbose         = 1
)

elapsed = datetime.now() - start_time
print(f"\n  Training complete in {elapsed}")

model.save(str(MODEL_DIR / "trained" / "transformer_final.keras"))
pd.DataFrame([CFG]).to_csv(
    MODEL_DIR / "configs" / "transformer_config.csv", index=False
)

# =============================================================================
# 9. EVALUATION
# =============================================================================
print("\n[5/6] Evaluating ...")

def inverse_transform_q(q_norm, q_min, q_max):
    return q_norm * (q_max - q_min) + q_min


def compute_metrics(y_true_norm, y_pred_raw,
                    q_min, q_max, split_name,
                    log_transform=False):
    if log_transform:
        y_pred_norm = inverse_log_transform(y_pred_raw.flatten())
    else:
        y_pred_norm = y_pred_raw.flatten()

    y_true = inverse_transform_q(y_true_norm, q_min, q_max)
    y_pred = np.clip(
        inverse_transform_q(y_pred_norm, q_min, q_max), 0, None
    )

    # NSE
    nse_val = 1 - np.sum((y_true - y_pred)**2) / \
                  np.sum((y_true - np.mean(y_true))**2)

    # KGE
    r       = np.corrcoef(y_true, y_pred)[0, 1]
    alpha   = np.std(y_pred)  / np.std(y_true)
    beta    = np.mean(y_pred) / np.mean(y_true)
    kge_val = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

    rmse_val  = np.sqrt(np.mean((y_true - y_pred)**2))
    mae_val   = np.mean(np.abs(y_true - y_pred))
    pbias_val = 100 * np.sum(y_pred - y_true) / np.sum(y_true)

    threshold = np.percentile(y_true, 95)
    peak_mask = y_true >= threshold
    peak_bias = 100 * (np.mean(y_pred[peak_mask]) -
                        np.mean(y_true[peak_mask])) / \
                       np.mean(y_true[peak_mask])

    eps         = 0.001
    log_nse_val = 1 - np.sum(
        (np.log(y_true+eps) - np.log(y_pred+eps))**2
    ) / np.sum(
        (np.log(y_true+eps) - np.mean(np.log(y_true+eps)))**2
    )

    return {
        "split"        : split_name,
        "NSE"          : round(nse_val,   4),
        "KGE"          : round(kge_val,   4),
        "RMSE"         : round(rmse_val,  4),
        "MAE"          : round(mae_val,   4),
        "PBIAS_%"      : round(pbias_val, 2),
        "Peak_Bias_%"  : round(peak_bias, 2),
        "Log_NSE"      : round(log_nse_val, 4),
        "r"            : round(r,     4),
        "alpha"        : round(alpha, 4),
        "beta"         : round(beta,  4),
    }, y_true, y_pred


y_pred_train_raw = model.predict(X_train, batch_size=64, verbose=0)
y_pred_val_raw   = model.predict(X_val,   batch_size=64, verbose=0)
y_pred_test_raw  = model.predict(X_test,  batch_size=64, verbose=0)

m_train, qt_true, qt_pred = compute_metrics(
    y_train, y_pred_train_raw, q_min, q_max,
    "Train", CFG["log_transform"])
m_val, qv_true, qv_pred = compute_metrics(
    y_val, y_pred_val_raw, q_min, q_max,
    "Validation", CFG["log_transform"])
m_test, qtest_true, qtest_pred = compute_metrics(
    y_test, y_pred_test_raw, q_min, q_max,
    "Test", CFG["log_transform"])

all_metrics = [m_train, m_val, m_test]
metrics_df  = pd.DataFrame(all_metrics)

# Print results vs LSTM v
print(f"\n  {'Metric':<14} {'LSTM':>10} {'Transformer':>12}")
print(f"  {'-'*40}")
lstm = {"NSE": 0.5759, "KGE": 0.6548, "RMSE": 0.2920,
           "MAE": 0.1683, "PBIAS_%": 3.80,
           "Peak_Bias_%": -38.76, "Log_NSE": 0.6329}
for metric in ["NSE", "KGE", "RMSE", "MAE",
               "PBIAS_%", "Peak_Bias_%", "Log_NSE"]:
    v  = m_test[metric]
    v1 = lstm[metric]
    arrow = "↑" if v > v1 else "↓"
    print(f"  {metric:<14} {v1:>10} {v:>12}  {arrow}")

metrics_df.to_csv(MET_DIR / "transformer_metrics.csv", index=False)

for sname, dates, yt, yp in [
    ("train", dates_train, qt_true,    qt_pred),
    ("val",   dates_val,   qv_true,    qv_pred),
    ("test",  dates_test,  qtest_true, qtest_pred),
]:
    pd.DataFrame({
        "date"         : pd.to_datetime(dates),
        "observed_m3s" : yt,
        "predicted_m3s": yp,
        "residual_m3s" : yt - yp,
    }).to_csv(PRED_DIR / f"transformer_predictions_{sname}.csv", index=False)

print(f"\n  Metrics → results/metrics/transformer_metrics.csv")
print(f"  Predictions → results/predictions/")

# =============================================================================
# 10. VISUALIZATION
# =============================================================================
print("\n[6/6] Generating plots ...")

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#080f1a")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

log        = pd.read_csv(MODEL_DIR / "configs" / "transformer_training_log.csv")
best_epoch = log["val_loss"].idxmin()

# ── Training history ──
ax0 = fig.add_subplot(gs[0, :2])
ax0.set_facecolor("#0d1825")
ax0.plot(log["epoch"], log["nse_metric"],
         color="#3b9eff", linewidth=1.8, label="Train NSE")
ax0.plot(log["epoch"], log["val_nse_metric"],
         color="#f4a261", linewidth=1.8, linestyle="--", label="Val NSE")
ax0.axvline(best_epoch, color="#00b4a0", linestyle=":",
            linewidth=1.5, label=f"Best epoch ({best_epoch})")
ax0.axhline(0, color="#4a6a82", linewidth=0.8, linestyle="--")
ax0.set_title("Training History — NSE", color="#e8f4f8", fontsize=11)
ax0.set_xlabel("Epoch", color="#8aafc4")
ax0.set_ylabel("NSE", color="#8aafc4")
ax0.tick_params(colors="#4a6a82")
ax0.spines[:].set_color("#1e3448")
ax0.legend(facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4")
ax0.set_facecolor("#0d1825")

# ── Loss ──
ax0b = fig.add_subplot(gs[0, 2])
ax0b.set_facecolor("#0d1825")
ax0b.plot(log["epoch"], log["loss"],
          color="#3b9eff", linewidth=1.5, label="Train")
ax0b.plot(log["epoch"], log["val_loss"],
          color="#f4a261", linewidth=1.5, linestyle="--", label="Val")
ax0b.set_title("Training Loss", color="#e8f4f8", fontsize=11)
ax0b.set_xlabel("Epoch", color="#8aafc4")
ax0b.set_ylabel("Peak-Weighted MSE", color="#8aafc4")
ax0b.tick_params(colors="#4a6a82")
ax0b.spines[:].set_color("#1e3448")
ax0b.legend(facecolor="#0d1825", edgecolor="#1e3448",
            labelcolor="#8aafc4", fontsize=8)
ax0b.set_facecolor("#0d1825")

# ── Test hydrograph ──
ax1 = fig.add_subplot(gs[1, :])
ax1.set_facecolor("#0d1825")
dates_t = pd.to_datetime(dates_test)
ax1.plot(dates_t, qtest_true, color="#8aafc4",
         linewidth=1.2, label="Observed", alpha=0.9)
ax1.plot(dates_t, qtest_pred, color="#00d4ff",
         linewidth=1.2, linestyle="--",
         label="Transformer Predicted", alpha=0.9)
ax1.fill_between(dates_t, qtest_true, qtest_pred,
                 alpha=0.15, color="#00d4ff", label="Error")
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
ax2.scatter(qtest_true, qtest_pred, alpha=0.4, s=8,
            color="#00d4ff", edgecolors="none")
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
ax3.hist(residuals, bins=50, color="#00d4ff", alpha=0.75, density=True)
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

# ── LSTM vs Transformer comparison ──
ax4 = fig.add_subplot(gs[2, 2])
ax4.set_facecolor("#0d1825")
metrics_compare = ["NSE", "KGE", "Log_NSE"]
lstm_vals  = [lstm[m]  for m in metrics_compare]
trans_vals = [m_test[m]   for m in metrics_compare]
x = np.arange(len(metrics_compare))
w = 0.35
ax4.bar(x - w/2, lstm_vals,  w, label="LSTM",
        color="#4a6a82", alpha=0.8)
ax4.bar(x + w/2, trans_vals, w, label="Transformer",
        color="#00d4ff", alpha=0.8)
ax4.axhline(0.75, color="#00b4a0", linewidth=1,
            linestyle=":", alpha=0.7, label="Good (0.75)")
ax4.set_xticks(x)
ax4.set_xticklabels(metrics_compare, color="#8aafc4")
ax4.set_ylabel("Score", color="#8aafc4")
ax4.set_title("LSTM vs Transformer", color="#e8f4f8", fontsize=11)
ax4.tick_params(colors="#4a6a82")
ax4.spines[:].set_color("#1e3448")
ax4.legend(facecolor="#0d1825", edgecolor="#1e3448",
           labelcolor="#8aafc4", fontsize=8)
ax4.set_ylim(-0.1, 1.1)
ax4.set_facecolor("#0d1825")

fig.suptitle("Transformer — Nahr Ibrahim Rainfall–Runoff Model",
             color="#e8f4f8", fontsize=14, y=0.98,
             fontfamily="monospace")

plt.savefig(FIG_DIR / "transformer_results.png",
            dpi=150, bbox_inches="tight", facecolor="#080f1a")
plt.show()

# =============================================================================
# 11. SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  TRANSFORMER SUMMARY")
print("=" * 65)
print(f"\n  Best epoch    : {best_epoch}")
print(f"  Training time : {elapsed}")
print(f"\n  Test Performance:")
print(f"    NSE         : {m_test['NSE']:.4f}")
print(f"    KGE         : {m_test['KGE']:.4f}")
print(f"    RMSE        : {m_test['RMSE']:.4f} m³/s")
print(f"    MAE         : {m_test['MAE']:.4f} m³/s")
print(f"    PBIAS       : {m_test['PBIAS_%']:.2f}%")
print(f"    Peak Bias   : {m_test['Peak_Bias_%']:.2f}%")
print(f"    Log-NSE     : {m_test['Log_NSE']:.4f}")
print(f"\n  Files saved:")
print(f"    models/trained/transformer_final.keras")
print(f"    models/checkpoints/transformer_best.keras")
print(f"    results/metrics/transformer_metrics.csv")
print(f"    results/predictions/transformer_predictions_*.csv")
print(f"    results/figures/transformer_results.png")
print(f"\n  ✅ Transformer complete. Next: CNN-LSTM model.")
print("=" * 65)