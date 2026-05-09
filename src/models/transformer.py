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

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")
SEQ_DIR = ROOT / "data" / "sequences"
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "results" / "figures"
MET_DIR = ROOT / "results" / "metrics"
PRED_DIR = ROOT / "results" / "predictions"

for d in [
    MODEL_DIR / "trained",
    MODEL_DIR / "checkpoints",
    MODEL_DIR / "configs",
    FIG_DIR,
    MET_DIR,
    PRED_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────────────
CFG = {
    "lookback": 30,
    "n_features": 18,
    "d_model": 64,  # must be divisible by n_heads
    "n_heads": 4,
    "n_encoder_blocks": 3,
    "ffn_dim": 128,
    "dropout": 0.2,
    "dense_units": [64, 32],
    "lr": 1e-3,
    "warmup_epochs": 10,
    "batch_size": 32,
    "epochs": 300,
    "patience": 40,
    "min_delta": 1e-6,
    "peak_weight": 3.0,
    "peak_threshold": 0.75,
    "log_transform": True,
    "seed": 42,
}

tf.random.set_seed(CFG["seed"])
np.random.seed(CFG["seed"])

print("Nahr Ibrahim — Transformer")
print(
    f"  d_model={CFG['d_model']}  heads={CFG['n_heads']}  "
    f"blocks={CFG['n_encoder_blocks']}  ffn={CFG['ffn_dim']}  "
    f"peak_w={CFG['peak_weight']}x  log={CFG['log_transform']}\n"
)

# ── load data ──────────────────────────────────────────────────────────────────
X_train = np.load(SEQ_DIR / "X_train.npy")
y_train = np.load(SEQ_DIR / "y_train.npy")
X_val = np.load(SEQ_DIR / "X_val.npy")
y_val = np.load(SEQ_DIR / "y_val.npy")
X_test = np.load(SEQ_DIR / "X_test.npy")
y_test = np.load(SEQ_DIR / "y_test.npy")

dates_train = np.load(SEQ_DIR / "dates_train.npy", allow_pickle=True)
dates_val = np.load(SEQ_DIR / "dates_val.npy", allow_pickle=True)
dates_test = np.load(SEQ_DIR / "dates_test.npy", allow_pickle=True)

scaler = pd.read_csv(SPLIT_DIR / "scaler_params.csv", index_col=0)
q_min = scaler.loc["discharge_m3s", "min"]
q_max = scaler.loc["discharge_m3s", "max"]

print(f"X_train {X_train.shape} | X_val {X_val.shape} | X_test {X_test.shape}")

# ── log-transform target ───────────────────────────────────────────────────────
EPS = 1e-6


def log_transform(y):
    return np.log1p(y / (EPS + 1))


def inv_log_transform(y):
    return np.expm1(y) * (EPS + 1)


if CFG["log_transform"]:
    y_train_m = log_transform(y_train).astype(np.float32)
    y_val_m = log_transform(y_val).astype(np.float32)
    y_test_m = log_transform(y_test).astype(np.float32)
    print("  log-transform applied\n")
else:
    y_train_m, y_val_m, y_test_m = y_train, y_val, y_test


# ── loss & metrics ─────────────────────────────────────────────────────────────
def peak_weighted_mse(y_true, y_pred):
    base = tf.square(y_true - y_pred)
    mask = tf.cast(y_true > CFG["peak_threshold"], tf.float32)
    w = 1.0 + (CFG["peak_weight"] - 1.0) * mask
    return tf.reduce_mean(w * base)


def nse_metric(y_true, y_pred):
    num = tf.reduce_sum(tf.square(y_true - y_pred))
    den = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - num / (den + tf.keras.backend.epsilon())


# ── lr schedule: warmup then cosine decay ──────────────────────────────────────
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = float(warmup_steps)
        self.total_steps = float(total_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.peak_lr * (step / self.warmup_steps)
        cosine_lr = (
            self.peak_lr
            * 0.5
            * (
                1.0
                + tf.cos(
                    np.pi
                    * (step - self.warmup_steps)
                    / (self.total_steps - self.warmup_steps)
                )
            )
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


steps_per_epoch = len(X_train) // CFG["batch_size"]
total_steps = CFG["epochs"] * steps_per_epoch
warmup_steps = CFG["warmup_epochs"] * steps_per_epoch

lr_schedule = WarmupCosineDecay(CFG["lr"], warmup_steps, total_steps)


# ── transformer building blocks ────────────────────────────────────────────────
class PositionalEncoding(tf.keras.layers.Layer):
    # Sine/cosine positional encoding (Vaswani et al. 2017).
    # Without this, attention treats all timesteps as unordered.
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

        pos = np.arange(max_len)[:, None]
        dims = np.arange(d_model)[None, :]
        angles = pos / np.power(10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.cast(angles[None, :, :], tf.float32)  # (1, max_len, d_model)

    def call(self, x):
        return x + self.pe[:, : tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len, "d_model": self.d_model})
        return cfg


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, ffn_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ffn_dim, activation="relu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(d_model),
            ]
        )
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        # self-attention + residual
        attn = self.attention(x, x, training=training)
        out1 = self.ln1(x + self.drop1(attn, training=training))
        # ffn + residual
        out2 = self.ln2(
            out1 + self.drop2(self.ffn(out1, training=training), training=training)
        )
        return out2

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "ffn_dim": self.ffn_dim,
                "dropout": self.dropout,
            }
        )
        return cfg


# ── model ──────────────────────────────────────────────────────────────────────
def build_transformer(cfg):
    inp = tf.keras.Input(shape=(cfg["lookback"], cfg["n_features"]), name="input")

    # project features → d_model so all layers share the same width
    x = tf.keras.layers.Dense(cfg["d_model"])(inp)
    x = PositionalEncoding(cfg["lookback"], cfg["d_model"])(x)
    x = tf.keras.layers.Dropout(cfg["dropout"])(x)

    for _ in range(cfg["n_encoder_blocks"]):
        x = TransformerEncoderBlock(
            cfg["d_model"], cfg["n_heads"], cfg["ffn_dim"], cfg["dropout"]
        )(x)

    # collapse temporal axis
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    for units in cfg["dense_units"]:
        x = tf.keras.layers.Dense(
            units, kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(cfg["dropout"] / 2)(x)

    out = tf.keras.layers.Dense(1, activation="linear", name="output")(x)
    return tf.keras.Model(inp, out, name="Transformer_NahrIbrahim")


model = build_transformer(CFG)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr_schedule, clipnorm=1.0),
    loss=peak_weighted_mse,
    metrics=[nse_metric, "mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
)

# ── callbacks ──────────────────────────────────────────────────────────────────
ckpt_path = str(MODEL_DIR / "checkpoints" / "transformer_best.keras")
log_path = str(MODEL_DIR / "configs" / "transformer_training_log.csv")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=CFG["patience"],
        min_delta=CFG["min_delta"],
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=0,
    ),
    tf.keras.callbacks.CSVLogger(log_path, append=False),
]

# ── train ──────────────────────────────────────────────────────────────────────
print(
    f"Training — max {CFG['epochs']} epochs, patience {CFG['patience']}, "
    f"warmup {CFG['warmup_epochs']} then cosine\n"
)

t0 = datetime.now()
history = model.fit(
    X_train,
    y_train_m,
    validation_data=(X_val, y_val_m),
    epochs=CFG["epochs"],
    batch_size=CFG["batch_size"],
    callbacks=callbacks,
    shuffle=False,
    verbose=1,
)
elapsed = datetime.now() - t0
print(f"\nDone in {elapsed}")

model.save(str(MODEL_DIR / "trained" / "transformer_final.keras"))
pd.DataFrame([CFG]).to_csv(
    MODEL_DIR / "configs" / "transformer_config.csv", index=False
)


# ── evaluation ─────────────────────────────────────────────────────────────────
def inv_q(q_norm):
    return q_norm * (q_max - q_min) + q_min


def metrics(y_norm, yhat_raw, label):
    yhat = (
        inv_log_transform(yhat_raw.flatten())
        if CFG["log_transform"]
        else yhat_raw.flatten()
    )
    yt = inv_q(y_norm)
    yp = np.clip(inv_q(yhat), 0, None)

    nse = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2)
    r = np.corrcoef(yt, yp)[0, 1]
    alpha = np.std(yp) / np.std(yt)
    beta = np.mean(yp) / np.mean(yt)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mae = np.mean(np.abs(yt - yp))
    pbias = 100 * np.sum(yp - yt) / np.sum(yt)

    peak_mask = yt >= np.percentile(yt, 95)
    peak_bias = (
        100 * (yp[peak_mask].mean() - yt[peak_mask].mean()) / yt[peak_mask].mean()
    )

    eps = 0.001
    log_nse = 1 - np.sum((np.log(yt + eps) - np.log(yp + eps)) ** 2) / np.sum(
        (np.log(yt + eps) - np.log(yt + eps).mean()) ** 2
    )

    return (
        {
            "split": label,
            "NSE": round(nse, 4),
            "KGE": round(kge, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "PBIAS_%": round(pbias, 2),
            "Peak_Bias_%": round(peak_bias, 2),
            "Log_NSE": round(log_nse, 4),
            "r": round(r, 4),
            "alpha": round(alpha, 4),
            "beta": round(beta, 4),
        },
        yt,
        yp,
    )


m_train, qt_true, qt_pred = metrics(
    y_train, model.predict(X_train, batch_size=64, verbose=0), "Train"
)
m_val, qv_true, qv_pred = metrics(
    y_val, model.predict(X_val, batch_size=64, verbose=0), "Validation"
)
m_test, qtest_true, qtest_pred = metrics(
    y_test, model.predict(X_test, batch_size=64, verbose=0), "Test"
)

df_met = pd.DataFrame([m_train, m_val, m_test])
df_met.to_csv(MET_DIR / "transformer_metrics.csv", index=False)

# compare vs LSTM baseline
lstm = {
    "NSE": 0.5759,
    "KGE": 0.6548,
    "RMSE": 0.2920,
    "MAE": 0.1683,
    "PBIAS_%": 3.80,
    "Peak_Bias_%": -38.76,
    "Log_NSE": 0.6329,
}

print(f"\n{'Metric':<14} {'LSTM':>10} {'Transformer':>12}")
print("-" * 40)
for m in ["NSE", "KGE", "RMSE", "MAE", "PBIAS_%", "Peak_Bias_%", "Log_NSE"]:
    arrow = "↑" if m_test[m] > lstm[m] else "↓"
    print(f"{m:<14} {lstm[m]:>10} {m_test[m]:>12}  {arrow}")

for label, dates, yt, yp in [
    ("train", dates_train, qt_true, qt_pred),
    ("val", dates_val, qv_true, qv_pred),
    ("test", dates_test, qtest_true, qtest_pred),
]:
    pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "observed_m3s": yt,
            "predicted_m3s": yp,
            "residual_m3s": yt - yp,
        }
    ).to_csv(PRED_DIR / f"transformer_predictions_{label}.csv", index=False)

# ── plots ──────────────────────────────────────────────────────────────────────
log = pd.read_csv(log_path)
best_epoch = log["val_loss"].idxmin()
BG, PANEL = "#080f1a", "#0d1825"
CYAN, BLUE, ORG, RED = "#00d4ff", "#3b9eff", "#f4a261", "#e76f51"
TICK, SPINE, TEXT = "#4a6a82", "#1e3448", "#8aafc4"


def _style(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TICK)
    ax.spines[:].set_color(SPINE)


fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# training NSE
ax = fig.add_subplot(gs[0, :2])
_style(ax)
ax.plot(log["epoch"], log["nse_metric"], color=BLUE, lw=1.8, label="Train NSE")
ax.plot(
    log["epoch"], log["val_nse_metric"], color=ORG, lw=1.8, ls="--", label="Val NSE"
)
ax.axvline(best_epoch, color="#00b4a0", ls=":", lw=1.5, label=f"Best ({best_epoch})")
ax.axhline(0, color=SPINE, lw=0.8, ls="--")
ax.set_title("Training History — NSE", color="#e8f4f8", fontsize=11)
ax.set_xlabel("Epoch", color=TEXT)
ax.set_ylabel("NSE", color=TEXT)
ax.legend(facecolor=PANEL, edgecolor=SPINE, labelcolor=TEXT)

# training loss
ax = fig.add_subplot(gs[0, 2])
_style(ax)
ax.plot(log["epoch"], log["loss"], color=BLUE, lw=1.5, label="Train")
ax.plot(log["epoch"], log["val_loss"], color=ORG, lw=1.5, ls="--", label="Val")
ax.set_title("Training Loss", color="#e8f4f8", fontsize=11)
ax.set_xlabel("Epoch", color=TEXT)
ax.set_ylabel("Peak-Weighted MSE", color=TEXT)
ax.legend(facecolor=PANEL, edgecolor=SPINE, labelcolor=TEXT, fontsize=8)

# hydrograph
ax = fig.add_subplot(gs[1, :])
_style(ax)
dates_t = pd.to_datetime(dates_test)
ax.plot(dates_t, qtest_true, color=TEXT, lw=1.2, label="Observed", alpha=0.9)
ax.plot(
    dates_t, qtest_pred, color=CYAN, lw=1.2, ls="--", label="Transformer", alpha=0.9
)
ax.fill_between(dates_t, qtest_true, qtest_pred, alpha=0.15, color=CYAN)
ax.set_title(
    f"Test Hydrograph (2021–2025)  NSE={m_test['NSE']:.3f}  "
    f"KGE={m_test['KGE']:.3f}  Peak Bias={m_test['Peak_Bias_%']:.1f}%",
    color="#e8f4f8",
    fontsize=11,
)
ax.set_ylabel("Discharge (m³/s)", color=TEXT)
ax.legend(facecolor=PANEL, edgecolor=SPINE, labelcolor=TEXT)

# scatter
ax = fig.add_subplot(gs[2, 0])
_style(ax)
ax.scatter(qtest_true, qtest_pred, alpha=0.4, s=8, color=CYAN, edgecolors="none")
lim = max(qtest_true.max(), qtest_pred.max()) * 1.05
ax.plot([0, lim], [0, lim], color=RED, lw=1.5, ls="--", label="1:1")
ax.set_xlabel("Observed (m³/s)", color=TEXT)
ax.set_ylabel("Predicted (m³/s)", color=TEXT)
ax.set_title("Observed vs Predicted", color="#e8f4f8", fontsize=11)
ax.legend(facecolor=PANEL, edgecolor=SPINE, labelcolor=TEXT, fontsize=8)

# residuals
ax = fig.add_subplot(gs[2, 1])
_style(ax)
res = qtest_true - qtest_pred
ax.hist(res, bins=50, color=CYAN, alpha=0.75, density=True)
ax.axvline(0, color=RED, lw=1.5, ls="--")
ax.axvline(res.mean(), color=ORG, lw=1.2, ls=":", label=f"Mean={res.mean():.3f}")
ax.set_xlabel("Residual (m³/s)", color=TEXT)
ax.set_ylabel("Density", color=TEXT)
ax.set_title("Residual Distribution", color="#e8f4f8", fontsize=11)
ax.legend(facecolor=PANEL, edgecolor=SPINE, labelcolor=TEXT, fontsize=8)

# LSTM vs Transformer
ax = fig.add_subplot(gs[2, 2])
_style(ax)
met_names = ["NSE", "KGE", "Log_NSE"]
x = np.arange(len(met_names))
w = 0.35
ax.bar(
    x - w / 2, [lstm[m] for m in met_names], w, label="LSTM", color="#4a6a82", alpha=0.8
)
ax.bar(
    x + w / 2,
    [m_test[m] for m in met_names],
    w,
    label="Transformer",
    color=CYAN,
    alpha=0.8,
)
ax.axhline(0.75, color="#00b4a0", lw=1, ls=":", alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(met_names, color=TEXT)
ax.set_ylabel("Score", color=TEXT)
ax.set_title("LSTM vs Transformer", color="#e8f4f8", fontsize=11)
ax.legend(facecolor=PANEL, edgecolor=SPINE, labelcolor=TEXT, fontsize=8)
ax.set_ylim(-0.1, 1.1)

fig.suptitle(
    "Transformer — Nahr Ibrahim Rainfall–Runoff Model",
    color="#e8f4f8",
    fontsize=14,
    y=0.98,
    fontfamily="monospace",
)

plt.savefig(
    FIG_DIR / "transformer_results.png", dpi=150, bbox_inches="tight", facecolor=BG
)
plt.show()

# ── summary ────────────────────────────────────────────────────────────────────
print(f"\nbest epoch : {best_epoch}  |  training time : {elapsed}")
print(
    f"\nTest  NSE={m_test['NSE']:.4f}  KGE={m_test['KGE']:.4f}  "
    f"RMSE={m_test['RMSE']:.4f}  MAE={m_test['MAE']:.4f}"
)
print(
    f"      PBIAS={m_test['PBIAS_%']:.2f}%  "
    f"Peak bias={m_test['Peak_Bias_%']:.2f}%  Log-NSE={m_test['Log_NSE']:.4f}"
)
