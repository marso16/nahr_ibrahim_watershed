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
    "model_name"      : "TFT",
    "lookback"        : 30,
    "n_features"      : 15,
    "d_model"         : 64,       # hidden state dimension throughout
    "n_heads"         : 4,        # attention heads
    "lstm_units"      : 64,       # LSTM encoder hidden size
    "dropout"         : 0.2,
    "dense_units"     : [32],
    # Training
    "learning_rate"   : 1e-3,
    "batch_size"      : 32,
    "epochs"          : 150,
    "patience"        : 20,
    "min_delta"       : 1e-5,
    "seed"            : 42,
}

tf.random.set_seed(CFG["seed"])
np.random.seed(CFG["seed"])

print("=" * 65)
print("  Nahr Ibrahim — Temporal Fusion Transformer (TFT)")
print("=" * 65)
print(f"  d_model   : {CFG['d_model']}")
print(f"  Heads     : {CFG['n_heads']}")
print(f"  LSTM units: {CFG['lstm_units']}")
print(f"  Dropout   : {CFG['dropout']}")

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

FEATURE_NAMES = [
    "precip_mm_day", "precip_3day", "precip_7day",
    "temp_mean_c", "temp_max_c", "temp_min_c", "temp_range_c",
    "swe_mm", "swe_delta", "snow_cover_pct",
    "month_sin", "month_cos",
]

print(f"  X_train : {X_train.shape}")
print(f"  X_val   : {X_val.shape}")
print(f"  X_test  : {X_test.shape}")

# =============================================================================
# 2. METRICS
# =============================================================================
def nse_metric(y_true, y_pred):
    num = tf.reduce_sum(tf.square(y_true - y_pred))
    den = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - num / (den + tf.keras.backend.epsilon())

# =============================================================================
# 3. TFT BUILDING BLOCKS
# =============================================================================
class GatedResidualNetwork(tf.keras.layers.Layer):
    """
    Gated Residual Network (GRN) — core TFT component.

    Allows the model to suppress irrelevant inputs using a gating
    mechanism. For example, in summer months the model learns to
    gate out snow-related features automatically.

    Architecture:
        x → Dense → ELU → Dense → GLU gate → LayerNorm(x + gated)

    The GLU (Gated Linear Unit) splits the output in half:
        one half becomes the signal, other half becomes the gate (sigmoid)
        output = signal * sigmoid(gate)
    """
    def __init__(self, units, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units   = units
        self.dropout = dropout

        self.dense1    = tf.keras.layers.Dense(units, activation="elu")
        self.dense2    = tf.keras.layers.Dense(units * 2)  # *2 for GLU
        self.dropout_l = tf.keras.layers.Dropout(dropout)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Project residual to same dimension if needed
        self.proj = tf.keras.layers.Dense(units, use_bias=False)

    def call(self, x, training=False):
        residual = self.proj(x)

        h = self.dense1(x)
        h = self.dropout_l(h, training=training)
        h = self.dense2(h)

        # GLU gating: split into two halves
        h1, h2 = tf.split(h, 2, axis=-1)
        gated   = h1 * tf.sigmoid(h2)

        return self.layernorm(residual + gated)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "dropout": self.dropout})
        return config


class VariableSelectionNetwork(tf.keras.layers.Layer):
    """
    Variable Selection Network (VSN) — learns feature importance.

    For each of the 12 input features, computes a weight (0-1) indicating
    how important that feature is for predicting discharge.

    After training, these weights reveal:
    - Which features drive discharge prediction most
    - How feature importance varies across seasons
    → Directly answers your Objective 4 (sensitivity analysis)

    Architecture:
        Each feature → individual GRN → feature-specific representation
        All features → combined GRN → importance weights (softmax)
        Output = weighted sum of feature representations
    """
    def __init__(self, n_features, units, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.units      = units
        self.dropout    = dropout

        # Individual GRN for each feature
        self.feature_grns = [
            GatedResidualNetwork(units, dropout, name=f"feat_grn_{i}")
            for i in range(n_features)
        ]
        # Feature projection layers — must be created in __init__, not call()
        self.feature_projs = [
            tf.keras.layers.Dense(units, name=f"feat_proj_{i}")
            for i in range(n_features)
        ]

        # Combined GRN for computing importance weights
        self.combined_grn = GatedResidualNetwork(
            n_features, dropout, name="combined_grn"
        )
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, x, training=False):
        # x shape: (batch, timesteps, n_features) or (batch, n_features)
        # Process each feature independently
        feature_outputs = []
        for i, (grn, proj) in enumerate(zip(self.feature_grns, self.feature_projs)):
            feat = x[..., i:i+1]  
            feat_proj = proj(feat)
            feature_outputs.append(grn(feat_proj, training=training))

        # Stack feature representations: (batch, timesteps, n_features, units)
        stacked = tf.stack(feature_outputs, axis=-2)

        # Compute importance weights from combined input
        combined = tf.reshape(x, (-1, tf.shape(x)[1], self.n_features))
        weights  = self.combined_grn(combined, training=training)
        weights  = self.softmax(weights)  # (batch, timesteps, n_features)

        # Weighted sum across features
        weights_expanded = tf.expand_dims(weights, axis=-1)
        output = tf.reduce_sum(stacked * weights_expanded, axis=-2)

        return output, weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_features": self.n_features,
            "units"     : self.units,
            "dropout"   : self.dropout,
        })
        return config


class InterpretableMultiHeadAttention(tf.keras.layers.Layer):
    """
    Interpretable Multi-Head Attention — TFT variant.

    Unlike standard multi-head attention, this version shares value
    projections across heads, making attention weights directly
    interpretable as 'importance of past day t for predicting today'.

    For your thesis: plot the average attention weights across test
    samples — you should see peaks at 3-10 days prior (karstic lag).
    """
    def __init__(self, n_heads, d_model, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head  = d_model // n_heads
        self.dropout = dropout  # ← add this line

        self.W_q = tf.keras.layers.Dense(d_model, use_bias=False)
        self.W_k = tf.keras.layers.Dense(d_model, use_bias=False)
        self.W_v = tf.keras.layers.Dense(self.d_head, use_bias=False)  # shared
        self.W_o = tf.keras.layers.Dense(d_model, use_bias=False)

        self.dropout_l  = tf.keras.layers.Dropout(dropout)
        self.layernorm  = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        seq_len    = tf.shape(x)[1]

        # Project to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)  # shared across heads: (batch, seq, d_head)

        # Split Q and K into heads
        Q = tf.reshape(Q, (batch_size, seq_len, self.n_heads, self.d_head))
        K = tf.reshape(K, (batch_size, seq_len, self.n_heads, self.d_head))
        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.transpose(K, [0, 2, 1, 3])

        # Expand V for all heads
        V_exp = tf.expand_dims(V, axis=1)
        V_exp = tf.tile(V_exp, [1, self.n_heads, 1, 1])

        # Scaled dot-product attention
        scale   = tf.math.sqrt(tf.cast(self.d_head, tf.float32))
        scores  = tf.matmul(Q, K, transpose_b=True) / scale
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.dropout_l(weights, training=training)

        # Context vector
        context = tf.matmul(weights, V_exp)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, seq_len,
                                       self.n_heads * self.d_head))

        output = self.W_o(context)

        # Average attention weights across heads for interpretability
        avg_weights = tf.reduce_mean(weights, axis=1)

        return self.layernorm(x + output), avg_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "dropout": self.dropout,
        })
        return config

# =============================================================================
# 4. BUILD TFT MODEL
# =============================================================================
print("\n[2/6] Building TFT model ...")

def build_tft(cfg: dict) -> tf.keras.Model:
    """
    Temporal Fusion Transformer for rainfall-runoff modeling.

    Pipeline:
      Input (30, 12)
        → Variable Selection Network    [learns feature importance]
        → GRN per timestep              [gating irrelevant inputs]
        → LSTM encoder                  [temporal state processing]
        → GRN                           [post-LSTM feature refinement]
        → Interpretable Attention       [focus on relevant past days]
        → GRN                           [post-attention refinement]
        → Dense head → Output (1,)
    """
    inputs = tf.keras.Input(
        shape=(cfg["lookback"], cfg["n_features"]),
        name="input_sequence"
    )

    # ── Variable Selection Network ────────────────────────────
    vsn = VariableSelectionNetwork(
        n_features = cfg["n_features"],
        units      = cfg["d_model"],
        dropout    = cfg["dropout"],
        name       = "vsn"
    )
    x, var_weights = vsn(inputs)
    # x shape: (batch, 30, d_model)

    # ── GRN on selected features ──────────────────────────────
    grn1 = GatedResidualNetwork(
        cfg["d_model"], cfg["dropout"], name="grn_input"
    )
    x = grn1(x)

    # ── LSTM encoder ──────────────────────────────────────────
    x = tf.keras.layers.LSTM(
        cfg["lstm_units"],
        return_sequences  = True,
        dropout           = cfg["dropout"],
        recurrent_dropout = 0.0,
        name              = "lstm_encoder"
    )(x)

    # Project LSTM output to d_model if different size
    if cfg["lstm_units"] != cfg["d_model"]:
        x = tf.keras.layers.Dense(cfg["d_model"], name="lstm_proj")(x)

    # ── Post-LSTM GRN ─────────────────────────────────────────
    grn2 = GatedResidualNetwork(
        cfg["d_model"], cfg["dropout"], name="grn_post_lstm"
    )
    x = grn2(x)

    # ── Interpretable Multi-Head Attention ────────────────────
    attn = InterpretableMultiHeadAttention(
        n_heads = cfg["n_heads"],
        d_model = cfg["d_model"],
        dropout = cfg["dropout"],
        name    = "interpretable_attention"
    )
    x, attn_weights = attn(x)

    # ── Post-attention GRN ────────────────────────────────────
    grn3 = GatedResidualNetwork(
        cfg["d_model"], cfg["dropout"], name="grn_post_attn"
    )
    x = grn3(x)

    # ── Aggregate temporal dimension ──────────────────────────
    x = tf.keras.layers.GlobalAveragePooling1D(name="global_avg")(x)

    # ── Dense prediction head ─────────────────────────────────
    for i, units in enumerate(cfg["dense_units"]):
        x = tf.keras.layers.Dense(units, name=f"dense_{i+1}")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu_{i+1}")(x)
        x = tf.keras.layers.Dropout(
            cfg["dropout"] / 2, name=f"drop_dense_{i+1}"
        )(x)

    outputs = tf.keras.layers.Dense(
        1, activation="linear", name="output"
    )(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs,
        name="TFT_NahrIbrahim"
    )
    return model, vsn


model, vsn_layer = build_tft(CFG)
model.summary()

# =============================================================================
# 5. COMPILE
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
# 6. CALLBACKS
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
        filepath       = str(MODEL_DIR / "checkpoints" / "tft_best.keras"),
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
        str(MODEL_DIR / "configs" / "tft_training_log.csv"),
        append=False
    ),
]

# =============================================================================
# 7. TRAIN
# =============================================================================
print("\n[4/6] Training TFT ...")
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

model.save(str(MODEL_DIR / "trained" / "tft_final.keras"))
pd.DataFrame([CFG]).to_csv(
    MODEL_DIR / "configs" / "tft_config.csv", index=False
)
print(f"  Model saved → models/trained/tft_final.keras")

# =============================================================================
# 8. EVALUATION
# =============================================================================
print("\n[5/6] Evaluating ...")

def inverse_transform_q(q_norm, q_min, q_max):
    return q_norm * (q_max - q_min) + q_min


def compute_metrics(y_true_norm, y_pred_norm, q_min, q_max, split_name):
    y_true = inverse_transform_q(y_true_norm, q_min, q_max)
    y_pred = np.clip(
        inverse_transform_q(y_pred_norm.flatten(), q_min, q_max), 0, None
    )

    nse_val  = 1 - np.sum((y_true-y_pred)**2) / \
                   np.sum((y_true-np.mean(y_true))**2)
    r        = np.corrcoef(y_true, y_pred)[0, 1]
    alpha    = np.std(y_pred)  / np.std(y_true)
    beta     = np.mean(y_pred) / np.mean(y_true)
    kge_val  = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
    rmse_val = np.sqrt(np.mean((y_true-y_pred)**2))
    mae_val  = np.mean(np.abs(y_true-y_pred))
    pbias    = 100 * np.sum(y_pred-y_true) / np.sum(y_true)

    threshold = np.percentile(y_true, 95)
    peak_mask = y_true >= threshold
    peak_bias = 100 * (np.mean(y_pred[peak_mask]) -
                        np.mean(y_true[peak_mask])) / \
                       np.mean(y_true[peak_mask])

    eps     = 0.001
    log_nse = 1 - np.sum(
        (np.log(y_true+eps) - np.log(y_pred+eps))**2
    ) / np.sum(
        (np.log(y_true+eps) - np.mean(np.log(y_true+eps)))**2
    )

    return {
        "split"       : split_name,
        "NSE"         : round(nse_val,  4),
        "KGE"         : round(kge_val,  4),
        "RMSE"        : round(rmse_val, 4),
        "MAE"         : round(mae_val,  4),
        "PBIAS_%"     : round(pbias,    2),
        "Peak_Bias_%" : round(peak_bias,2),
        "Log_NSE"     : round(log_nse,  4),
        "r"           : round(r,        4),
        "alpha"       : round(alpha,    4),
        "beta"        : round(beta,     4),
    }, y_true, y_pred


y_pred_train = model.predict(X_train, batch_size=64, verbose=0)
y_pred_val   = model.predict(X_val,   batch_size=64, verbose=0)
y_pred_test  = model.predict(X_test,  batch_size=64, verbose=0)

m_train, qt_true, qt_pred     = compute_metrics(y_train, y_pred_train, q_min, q_max, "Train")
m_val,   qv_true, qv_pred     = compute_metrics(y_val,   y_pred_val,   q_min, q_max, "Validation")
m_test,  qtest_true, qtest_pred = compute_metrics(y_test, y_pred_test, q_min, q_max, "Test")

metrics_df = pd.DataFrame([m_train, m_val, m_test])

print(f"\n  {'Metric':<14} {'Train':>10} {'Validation':>12} {'Test':>10}")
print(f"  {'-'*50}")
for metric in ["NSE","KGE","RMSE","MAE","PBIAS_%","Peak_Bias_%","Log_NSE"]:
    print(f"  {metric:<14} "
          f"{metrics_df[metrics_df.split=='Train'][metric].values[0]:>10} "
          f"{metrics_df[metrics_df.split=='Validation'][metric].values[0]:>12} "
          f"{metrics_df[metrics_df.split=='Test'][metric].values[0]:>10}")

# Full benchmark comparison
print(f"\n  {'Metric':<14} {'LSTM':>8} {'CNN-LSTM':>10} {'Transf':>8} {'TFT':>8}")
print(f"  {'-'*50}")
lstm = {"NSE":0.5759,"KGE":0.6548,"Peak_Bias_%":-38.76,"Log_NSE":0.6329}
transf  = {"NSE":0.6432,"KGE":0.6858,"Peak_Bias_%":-37.11,"Log_NSE":0.6710}
cnn_lstm= {"NSE": None, "KGE": None, "Peak_Bias_%": None,  "Log_NSE": None}
for metric in ["NSE","KGE","Peak_Bias_%"]:
    cnn_val = cnn_lstm[metric]
    cnn_str = f"{cnn_val:>8}" if cnn_val else "     N/A"
    print(f"  {metric:<14} {lstm[metric]:>8} {cnn_str:>10} "
          f"{transf[metric]:>8} {m_test[metric]:>8}")

metrics_df.to_csv(MET_DIR / "tft_metrics.csv", index=False)

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
    }).to_csv(PRED_DIR / f"tft_predictions_{sname}.csv", index=False)

print(f"\n  Metrics     → results/metrics/tft_metrics.csv")
print(f"  Predictions → results/predictions/")

# =============================================================================
# 9. VISUALIZATION
# =============================================================================
print("\n[6/6] Generating plots ...")

fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor("#080f1a")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

log        = pd.read_csv(MODEL_DIR / "configs" / "tft_training_log.csv")
best_epoch = log["val_loss"].idxmin()

# ── Training NSE ──
ax0 = fig.add_subplot(gs[0, :2])
ax0.set_facecolor("#0d1825")
ax0.plot(log["epoch"], log["nse_metric"],
         color="#a855f7", linewidth=1.8, label="Train NSE")
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

# ── Training Loss ──
ax0b = fig.add_subplot(gs[0, 2])
ax0b.set_facecolor("#0d1825")
ax0b.plot(log["epoch"], log["loss"],
          color="#a855f7", linewidth=1.5, label="Train")
ax0b.plot(log["epoch"], log["val_loss"],
          color="#f4a261", linewidth=1.5, linestyle="--", label="Val")
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
         color="#a855f7", linewidth=1.2, linestyle="--",
         label="TFT Predicted", alpha=0.9)
ax1.fill_between(dates_t, qtest_true, qtest_pred,
                 alpha=0.15, color="#a855f7", label="Error")
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
            alpha=0.4, s=8, color="#a855f7", edgecolors="none")
lim = max(qtest_true.max(), qtest_pred.max()) * 1.05
ax2.plot([0, lim], [0, lim], color="#e76f51",
         linewidth=1.5, linestyle="--", label="1:1 line")
ax2.set_xlabel("Observed (m³/s)", color="#8aafc4")
ax2.set_ylabel("Predicted (m³/s)", color="#8aafc4")
ax2.set_title("Observed vs Predicted", color="#e8f4f8", fontsize=11)
ax2.tick_params(colors="#4a6a82")
ax2.spines[:].set_color("#1e3448")
ax2.legend(facecolor="#0d1825", edgecolor="#1e3864",
           labelcolor="#8aafc4", fontsize=8)
ax2.set_facecolor("#0d1825")

# ── Residuals ──
ax3 = fig.add_subplot(gs[2, 1])
ax3.set_facecolor("#0d1825")
residuals = qtest_true - qtest_pred
ax3.hist(residuals, bins=50, color="#a855f7", alpha=0.75, density=True)
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

# ── Full 4-model benchmark comparison ──
ax4 = fig.add_subplot(gs[2, 2])
ax4.set_facecolor("#0d1825")
metric_names = ["NSE", "KGE", "Log_NSE"]
models_cmp = [
    ("LSTM",        [0.5759, 0.6548, 0.6329], "#4a6a82"),
    ("Transformer", [0.6432, 0.6858, 0.6710], "#00d4ff"),
    ("CNN-LSTM",    [None,   None,   None   ], "#00b4a0"),
    ("TFT",         [m_test["NSE"], m_test["KGE"], m_test["Log_NSE"]], "#a855f7"),
]
x = np.arange(len(metric_names))
w = 0.2
for i, (name, vals, color) in enumerate(models_cmp):
    clean_vals = [v if v is not None else 0 for v in vals]
    ax4.bar(x + i*w, clean_vals, w, label=name, color=color, alpha=0.8)

ax4.axhline(0.75, color="#f4a261", linewidth=1,
            linestyle=":", alpha=0.7, label="Good (0.75)")
ax4.set_xticks(x + 1.5*w)
ax4.set_xticklabels(metric_names, color="#8aafc4", fontsize=9)
ax4.set_ylabel("Score", color="#8aafc4")
ax4.set_title("4-Model Benchmark", color="#e8f4f8", fontsize=11)
ax4.tick_params(colors="#4a6a82")
ax4.spines[:].set_color("#1e3448")
ax4.legend(facecolor="#0d1825", edgecolor="#1e3448",
           labelcolor="#8aafc4", fontsize=7)
ax4.set_ylim(-0.1, 1.1)
ax4.set_facecolor("#0d1825")

fig.suptitle("TFT — Nahr Ibrahim Rainfall–Runoff Model",
             color="#e8f4f8", fontsize=14, y=0.98,
             fontfamily="monospace")

plt.savefig(FIG_DIR / "tft_results.png",
            dpi=150, bbox_inches="tight", facecolor="#080f1a")
plt.show()

# =============================================================================
# 10. VARIABLE IMPORTANCE — TFT exclusive figure
# =============================================================================
print("\n  Computing variable importance weights ...")

# Build a sub-model that outputs VSN weights
vsn_model = tf.keras.Model(
    inputs  = model.input,
    outputs = model.get_layer("vsn").output
)

# Get weights for test set (use a batch)
_, var_weights_test = vsn_model.predict(
    X_test[:500], batch_size=64, verbose=0
)
# var_weights_test shape: (500, 30, 12) — weights per timestep per feature
mean_importance = var_weights_test.mean(axis=(0, 1))  # average over samples and time

# Plot feature importance
fig_vi, ax_vi = plt.subplots(figsize=(12, 5))
fig_vi.patch.set_facecolor("#080f1a")
ax_vi.set_facecolor("#0d1825")

colors_vi = ["#3b9eff" if i < 3 else
             "#f4a261" if i < 7 else
             "#00b4a0" if i < 10 else
             "#a855f7"
             for i in range(len(FEATURE_NAMES))]

bars = ax_vi.barh(FEATURE_NAMES, mean_importance,
                  color=colors_vi, alpha=0.85)
ax_vi.set_xlabel("Mean Variable Importance Weight", color="#8aafc4")
ax_vi.set_title("TFT Variable Importance — Nahr Ibrahim Watershed\n"
                "(higher = more important for discharge prediction)",
                color="#e8f4f8", fontsize=12)
ax_vi.tick_params(colors="#4a6a82")
ax_vi.spines[:].set_color("#1e3448")

for bar, val in zip(bars, mean_importance):
    ax_vi.text(val + 0.001, bar.get_y() + bar.get_height()/2,
               f"{val:.3f}", va="center", color="#e8f4f8", fontsize=8)

fig_vi.tight_layout()
plt.savefig(FIG_DIR / "tft_variable_importance.png",
            dpi=150, bbox_inches="tight", facecolor="#080f1a")
plt.show()

importance_df = pd.DataFrame({
    "feature"   : FEATURE_NAMES,
    "importance": mean_importance
}).sort_values("importance", ascending=False)
importance_df.to_csv(MET_DIR / "tft_variable_importance.csv", index=False)
print(f"  Variable importance saved → results/metrics/tft_variable_importance.csv")
print(f"\n  Top 3 most important features:")
for _, row in importance_df.head(3).iterrows():
    print(f"    {row['feature']:<22} : {row['importance']:.4f}")

# =============================================================================
# 11. SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  TFT SUMMARY")
print("=" * 65)
print(f"\n  Best epoch    : {best_epoch}")
print(f"  Training time : {elapsed}")
print(f"\n  Test Performance:")
print(f"    NSE         : {m_test['NSE']:.4f}  (>0.75 = good)")
print(f"    KGE         : {m_test['KGE']:.4f}  (>0.75 = good)")
print(f"    RMSE        : {m_test['RMSE']:.4f} m³/s")
print(f"    MAE         : {m_test['MAE']:.4f} m³/s")
print(f"    PBIAS       : {m_test['PBIAS_%']:.2f}%")
print(f"    Peak Bias   : {m_test['Peak_Bias_%']:.2f}%")
print(f"    Log-NSE     : {m_test['Log_NSE']:.4f}")
print(f"\n  TFT-exclusive outputs:")
print(f"    results/metrics/tft_variable_importance.csv")
print(f"    results/figures/tft_variable_importance.png")
print(f"\n  Files saved:")
print(f"    models/trained/tft_final.keras")
print(f"    models/checkpoints/tft_best.keras")
print(f"    results/metrics/tft_metrics.csv")
print(f"    results/predictions/tft_predictions_*.csv")
print(f"    results/figures/tft_results.png")
print(f"\n  ✅ TFT complete.")
print("=" * 65)