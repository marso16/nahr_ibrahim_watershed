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
    "lstm_units": [128, 64],
    "dense_units": [32],
    "dropout": 0.3,
    "recurrent_drop": 0.2,
    "lr": 1e-3,
    "batch_size": 32,
    "epochs": 150,
    "patience": 20,
    "min_delta": 1e-5,
    "seed": 42,
}

tf.random.set_seed(CFG["seed"])
np.random.seed(CFG["seed"])

print("Nahr Ibrahim — LSTM rainfall-runoff model")
print(f"  units={CFG['lstm_units']}  dropout={CFG['dropout']}  lr={CFG['lr']}\n")

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
print(f"Q range [{q_min:.3f}, {q_max:.3f}] m³/s\n")


# ── custom metric ──────────────────────────────────────────────────────────────
def nse_metric(y_true, y_pred):
    num = tf.reduce_sum(tf.square(y_true - y_pred))
    den = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - num / (den + tf.keras.backend.epsilon())


# ── model ──────────────────────────────────────────────────────────────────────
def build_lstm(cfg):
    inp = tf.keras.Input(shape=(cfg["lookback"], cfg["n_features"]), name="input")
    x = inp
    for i, units in enumerate(cfg["lstm_units"]):
        x = tf.keras.layers.LSTM(
            units,
            return_sequences=i < len(cfg["lstm_units"]) - 1,
            dropout=cfg["dropout"],
            recurrent_dropout=cfg["recurrent_drop"],
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(x)
        x = tf.keras.layers.Dropout(cfg["dropout"])(x)
    for units in cfg["dense_units"]:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(cfg["dropout"] / 2)(x)
    out = tf.keras.layers.Dense(1, activation="linear", name="output")(x)
    return tf.keras.Model(inp, out, name="LSTM_NahrIbrahim")


model = build_lstm(CFG)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(CFG["lr"], clipnorm=1.0),
    loss="mse",
    metrics=[nse_metric, "mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
)

# ── callbacks ──────────────────────────────────────────────────────────────────
ckpt_path = str(MODEL_DIR / "checkpoints" / "lstm_best.keras")
log_path = str(MODEL_DIR / "configs" / "lstm_training_log.csv")

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
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(log_path, append=False),
]

# ── train ──────────────────────────────────────────────────────────────────────
print(
    f"Training — max {CFG['epochs']} epochs, patience {CFG['patience']}, "
    f"batch {CFG['batch_size']}\n"
)

t0 = datetime.now()
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=CFG["epochs"],
    batch_size=CFG["batch_size"],
    callbacks=callbacks,
    shuffle=False,
    verbose=1,
)
elapsed = datetime.now() - t0
print(f"\nDone in {elapsed}")

model.save(str(MODEL_DIR / "trained" / "lstm_final.keras"))
pd.DataFrame([CFG]).to_csv(MODEL_DIR / "configs" / "lstm_config.csv", index=False)


# ── evaluation ─────────────────────────────────────────────────────────────────
def inv_q(q_norm):
    return q_norm * (q_max - q_min) + q_min


def metrics(y_norm, yhat_norm, label):
    yt = inv_q(y_norm)
    yp = np.clip(inv_q(yhat_norm.flatten()), 0, None)

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
df_met.to_csv(MET_DIR / "lstm_metrics.csv", index=False)

print(f"\n{'Metric':<14} {'Train':>10} {'Val':>10} {'Test':>10}")
print("-" * 48)
for m in ["NSE", "KGE", "RMSE", "MAE", "PBIAS_%", "Peak_Bias_%", "Log_NSE"]:
    tr = df_met[df_met.split == "Train"][m].values[0]
    vl = df_met[df_met.split == "Validation"][m].values[0]
    te = df_met[df_met.split == "Test"][m].values[0]
    print(f"{m:<14} {tr:>10} {vl:>10} {te:>10}")

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
    ).to_csv(PRED_DIR / f"lstm_predictions_{label}.csv", index=False)

# ── plots ──────────────────────────────────────────────────────────────────────
log = pd.read_csv(log_path)
best_epoch = log["val_loss"].idxmin()
BG, PANEL = "#080f1a", "#0d1825"
BLUE, ORG, RED = "#3b9eff", "#f4a261", "#e76f51"
TICK, SPINE = "#4a6a82", "#1e3448"
TEXT = "#8aafc4"


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
ax.axhline(0.75, color="#00b4a0", lw=0.8, ls="--", alpha=0.4)
ax.axhline(0, color=SPINE, lw=0.8, ls="--")
ax.set_title("Training History — NSE", color="#e8f4f8", fontsize=11)
ax.set_xlabel("Epoch", color=TEXT)
ax.set_ylabel("NSE", color=TEXT)
ax.legend(facecolor=PANEL, edgecolor=SPINE, labelcolor=TEXT)

# training loss
ax = fig.add_subplot(gs[0, 2])
_style(ax)
ax.plot(log["epoch"], log["loss"], color=BLUE, lw=1.5, label="Train MSE")
ax.plot(log["epoch"], log["val_loss"], color=ORG, lw=1.5, ls="--", label="Val MSE")
ax.set_title("Training Loss (MSE)", color="#e8f4f8", fontsize=11)
ax.set_xlabel("Epoch", color=TEXT)
ax.set_ylabel("MSE", color=TEXT)
ax.legend(facecolor=PANEL, edgecolor=SPINE, labelcolor=TEXT, fontsize=8)

# hydrograph
ax = fig.add_subplot(gs[1, :])
_style(ax)
dates_t = pd.to_datetime(dates_test)
ax.plot(dates_t, qtest_true, color="#8aafc4", lw=1.2, label="Observed", alpha=0.9)
ax.plot(dates_t, qtest_pred, color=BLUE, lw=1.2, ls="--", label="Predicted", alpha=0.9)
ax.fill_between(dates_t, qtest_true, qtest_pred, alpha=0.15, color=BLUE)
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
ax.scatter(qtest_true, qtest_pred, alpha=0.4, s=8, color=BLUE, edgecolors="none")
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
ax.hist(res, bins=50, color=BLUE, alpha=0.75, density=True)
ax.axvline(0, color=RED, lw=1.5, ls="--")
ax.axvline(res.mean(), color=ORG, lw=1.2, ls=":", label=f"Mean={res.mean():.3f}")
ax.set_xlabel("Residual (m³/s)", color=TEXT)
ax.set_ylabel("Density", color=TEXT)
ax.set_title("Residual Distribution", color="#e8f4f8", fontsize=11)
ax.legend(facecolor=PANEL, edgecolor=SPINE, labelcolor=TEXT, fontsize=8)

# metrics bar
ax = fig.add_subplot(gs[2, 2])
_style(ax)
met_names = ["NSE", "KGE", "Log_NSE"]
x = np.arange(len(met_names))
w = 0.25
for i, (split, c) in enumerate(zip(["Train", "Validation", "Test"], [BLUE, ORG, RED])):
    row = df_met[df_met.split == split]
    ax.bar(
        x + i * w,
        [row[m].values[0] for m in met_names],
        w,
        label=split,
        color=c,
        alpha=0.8,
    )
ax.axhline(0.75, color="#00b4a0", lw=1, ls=":", alpha=0.7)
ax.set_xticks(x + w)
ax.set_xticklabels(met_names, color=TEXT, fontsize=9)
ax.set_ylabel("Score", color=TEXT)
ax.set_title("Performance Metrics", color="#e8f4f8", fontsize=11)
ax.legend(facecolor=PANEL, edgecolor=SPINE, labelcolor=TEXT, fontsize=7)
ax.set_ylim(-0.1, 1.1)

fig.suptitle(
    "LSTM — Nahr Ibrahim Rainfall-Runoff Model",
    color="#e8f4f8",
    fontsize=14,
    y=0.98,
    fontfamily="monospace",
)

plt.savefig(FIG_DIR / "lstm_results.png", dpi=150, bbox_inches="tight", facecolor=BG)
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
