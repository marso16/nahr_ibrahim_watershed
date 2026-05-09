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
MASTER_DIR = ROOT / "data" / "master"
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

# λ controls weight of water balance penalty vs MSE
# 0.05 = 5% penalty — conservative and stable
LAMBDA_WB = 0.05

# feature indices used in water balance computation
IDX_PRECIP = 0  # precip_mm_day
IDX_SWE = 7  # swe_mm

tf.random.set_seed(42)
np.random.seed(42)

print("Nahr Ibrahim — Physics-Informed Models")
print(f"  λ={LAMBDA_WB}  constraint: P - ET - Q - ΔS ≈ 0  PET: Hamon (1961)\n")

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

master = pd.read_csv(MASTER_DIR / "nahr_ibrahim_master_model.csv", parse_dates=["date"])

print(f"X_train {X_train.shape} | X_val {X_val.shape} | X_test {X_test.shape}")
print(f"Q range [{q_min:.3f}, {q_max:.3f}] m³/s\n")


# ── Hamon PET ──────────────────────────────────────────────────────────────────
def hamon_pet(temp_c, doy):
    # Hamon (1961): PET = 0.1651 × Ld × ρsat
    # Daylight hours — sinusoidal approx for Lebanon (~34°N)
    Ld = 12 + 4 * np.sin(2 * np.pi * (doy - 80) / 365)
    rho_sat = (
        216.7 * 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3)) / (temp_c + 273.3)
    )
    return np.clip(0.1651 * Ld * rho_sat, 0, None)


def get_pet(dates_array):
    dates_pd = pd.to_datetime(dates_array)
    merged = pd.DataFrame({"date": dates_pd}).merge(
        master[["date", "temp_mean_c"]], on="date", how="left"
    )
    temp = merged["temp_mean_c"].fillna(merged["temp_mean_c"].mean()).values
    return hamon_pet(temp, dates_pd.dayofyear.values).astype(np.float32)


pet_train = get_pet(dates_train)
pet_val = get_pet(dates_val)
pet_test = get_pet(dates_test)

print(
    f"PET train: {pet_train.min():.2f}–{pet_train.max():.2f} mm/day  "
    f"(mean {pet_train.mean():.2f})"
)

# normalize PET on the same scale as precip
pet_scale = scaler.loc["precip_mm_day", "max"] - scaler.loc["precip_mm_day", "min"]
pet_train_n = (pet_train / pet_scale).astype(np.float32)
pet_val_n = (pet_val / pet_scale).astype(np.float32)
pet_test_n = (pet_test / pet_scale).astype(np.float32)


# ── augment input: append PET as extra feature ─────────────────────────────────
def augment(X, pet_norm):
    # broadcast daily PET scalar across all 30 timesteps
    pet_exp = np.tile(pet_norm[:, None, None], (1, X.shape[1], 1)).astype(np.float32)
    return np.concatenate([X, pet_exp], axis=2)


X_train_aug = augment(X_train, pet_train_n)
X_val_aug = augment(X_val, pet_val_n)
X_test_aug = augment(X_test, pet_test_n)

N_FEAT = X_train_aug.shape[2] - 1  # exclude PET from model input
print(f"augmented shape: {X_train_aug.shape}  (PET appended as last feature)\n")


# ── metrics helpers ────────────────────────────────────────────────────────────
def nse_metric(y_true, y_pred):
    num = tf.reduce_sum(tf.square(y_true - y_pred))
    den = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - num / (den + tf.keras.backend.epsilon())


def inv_q(q_norm):
    return np.clip(q_norm * (q_max - q_min) + q_min, 0, None)


def metrics(y_norm, yhat_norm, label):
    yt = inv_q(y_norm)
    yp = inv_q(yhat_norm.flatten())

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
        },
        yt,
        yp,
    )


# ── custom training loop with physics-informed loss ───────────────────────────
def train_pi(name, model, X_tr, y_tr, X_v, y_v, epochs=150, patience=20):
    print(f"\nTraining {name} ...")
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

    @tf.function
    def train_step(xb, yb, pet_b):
        with tf.GradientTape() as tape:
            yhat = model(xb, training=True)
            mse = tf.reduce_mean(tf.square(yb - yhat))
            P = tf.reduce_mean(xb[:, :, IDX_PRECIP], axis=1)
            dS = tf.reduce_mean(xb[:, :, IDX_SWE], axis=1)
            Q = tf.squeeze(yhat, axis=-1)
            wb = tf.reduce_mean(tf.square(P - pet_b - Q - dS))
            loss = mse + LAMBDA_WB * wb
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, mse, wb

    @tf.function
    def val_step(xb, yb, pet_b):
        yhat = model(xb, training=False)
        mse = tf.reduce_mean(tf.square(yb - yhat))
        P = tf.reduce_mean(xb[:, :, IDX_PRECIP], axis=1)
        dS = tf.reduce_mean(xb[:, :, IDX_SWE], axis=1)
        Q = tf.squeeze(yhat, axis=-1)
        wb = tf.reduce_mean(tf.square(P - pet_b - Q - dS))
        return mse + LAMBDA_WB * wb

    bs = 32
    best_val_loss = np.inf
    patience_count = 0
    best_weights = None
    log = []
    t0 = datetime.now()

    for epoch in range(epochs):
        idx = np.random.permutation(len(X_tr))
        Xs, ys = X_tr[idx], y_tr[idx]
        pet_s = Xs[:, 0, -1]

        train_losses = []
        for i in range(0, len(Xs), bs):
            xb = tf.constant(Xs[i : i + bs, :, :-1], dtype=tf.float32)
            yb = tf.constant(ys[i : i + bs, None], dtype=tf.float32)
            pb = tf.constant(pet_s[i : i + bs], dtype=tf.float32)
            tl, _, _ = train_step(xb, yb, pb)
            train_losses.append(float(tl))

        pet_v = X_v[:, 0, -1]
        val_losses = []
        for i in range(0, len(X_v), bs):
            xb = tf.constant(X_v[i : i + bs, :, :-1], dtype=tf.float32)
            yb = tf.constant(y_v[i : i + bs, None], dtype=tf.float32)
            pb = tf.constant(pet_v[i : i + bs], dtype=tf.float32)
            val_losses.append(float(val_step(xb, yb, pb)))

        tl = np.mean(train_losses)
        vl = np.mean(val_losses)
        yp_v = model.predict(X_v[:, :, :-1], batch_size=256, verbose=0).flatten()
        val_nse = 1 - np.sum((y_v - yp_v) ** 2) / np.sum((y_v - np.mean(y_v)) ** 2)

        log.append({"epoch": epoch, "loss": tl, "val_loss": vl, "val_nse": val_nse})

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  {epoch + 1:>4}  train={tl:.5f}  val={vl:.5f}  val_nse={val_nse:.4f}"
            )

        if vl < best_val_loss - 1e-5:
            best_val_loss = vl
            patience_count = 0
            best_weights = model.get_weights()
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  early stop at epoch {epoch + 1}")
                break
            if patience_count % 10 == 0:
                lr = max(float(opt.learning_rate) * 0.5, 1e-6)
                opt.learning_rate.assign(lr)
                print(f"  LR → {lr:.2e}")

    if best_weights is not None:
        model.set_weights(best_weights)

    elapsed = datetime.now() - t0
    log_path = MODEL_DIR / "configs" / f"{name}_training_log.csv"
    pd.DataFrame(log).to_csv(log_path, index=False)
    print(f"  done in {elapsed}  best_val={best_val_loss:.5f}")
    return model, log, elapsed


# ── PI-LSTM ────────────────────────────────────────────────────────────────────
def build_pi_lstm(n_feat):
    inp = tf.keras.Input(shape=(30, n_feat), name="input")
    x = inp
    for i, units in enumerate([128, 64]):
        x = tf.keras.layers.LSTM(
            units,
            return_sequences=i < 1,
            dropout=0.3,
            recurrent_dropout=0.2,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(x)
        x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    out = tf.keras.layers.Dense(1, activation="linear", name="output")(x)
    return tf.keras.Model(inp, out, name="PI_LSTM")


# ── PI-Transformer ─────────────────────────────────────────────────────────────
def build_pi_transformer(
    n_feat, d_model=64, n_heads=4, ffn_dim=128, n_blocks=3, dropout=0.2
):
    inp = tf.keras.Input(shape=(30, n_feat), name="input")

    # inline positional encoding — avoids custom class dependency
    pos = np.arange(30)[:, None]
    dims = np.arange(d_model)[None, :]
    angles = pos / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    pe = tf.cast(angles[None, :, :], tf.float32)

    x = tf.keras.layers.Dense(d_model)(inp) + pe
    x = tf.keras.layers.Dropout(dropout)(x)

    for _ in range(n_blocks):
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout
        )(x, x)
        attn = tf.keras.layers.Dropout(dropout)(attn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)

        ffn = tf.keras.layers.Dense(ffn_dim, activation="relu")(x)
        ffn = tf.keras.layers.Dropout(dropout)(ffn)
        ffn = tf.keras.layers.Dense(d_model)(ffn)
        ffn = tf.keras.layers.Dropout(dropout)(ffn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    for units in [64, 32]:
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(dropout / 2)(x)

    out = tf.keras.layers.Dense(1, activation="linear")(x)
    return tf.keras.Model(inp, out, name="PI_Transformer")


pi_lstm = build_pi_lstm(N_FEAT)
pi_transformer = build_pi_transformer(N_FEAT)
pi_lstm.summary()
pi_transformer.summary()

# ── train ──────────────────────────────────────────────────────────────────────
pi_lstm, lstm_log, lstm_time = train_pi(
    "pi_lstm", pi_lstm, X_train_aug, y_train, X_val_aug, y_val, epochs=150, patience=20
)
pi_lstm.save(str(MODEL_DIR / "trained" / "pi_lstm_final.keras"))

pi_transformer, trans_log, trans_time = train_pi(
    "pi_transformer",
    pi_transformer,
    X_train_aug,
    y_train,
    X_val_aug,
    y_val,
    epochs=150,
    patience=20,
)
pi_transformer.save(str(MODEL_DIR / "trained" / "pi_transformer_final.keras"))

# ── evaluate ───────────────────────────────────────────────────────────────────
PURE_LSTM = {
    "NSE": 0.515,
    "KGE": 0.423,
    "Log_NSE": 0.633,
    "Peak_Bias_%": -51.7,
    "PBIAS_%": 0.0,
}
PURE_TRANSFORMER = {
    "NSE": 0.680,
    "KGE": 0.671,
    "Log_NSE": 0.671,
    "Peak_Bias_%": -37.0,
    "PBIAS_%": 2.3,
}

all_metrics = []

for mname, mdl in [("PI-LSTM", pi_lstm), ("PI-Transformer", pi_transformer)]:
    yp_tr = mdl.predict(X_train_aug[:, :, :-1], batch_size=256, verbose=0)
    yp_v = mdl.predict(X_val_aug[:, :, :-1], batch_size=256, verbose=0)
    yp_te = mdl.predict(X_test_aug[:, :, :-1], batch_size=256, verbose=0)

    m_tr, _, _ = metrics(y_train, yp_tr, "Train")
    m_v, _, _ = metrics(y_val, yp_v, "Validation")
    m_te, qt_true, qt_pred = metrics(y_test, yp_te, "Test")
    all_metrics.append({**m_te, "model": mname})

    for label, dates, yp, yn in [
        ("train", dates_train, yp_tr, y_train),
        ("val", dates_val, yp_v, y_val),
        ("test", dates_test, yp_te, y_test),
    ]:
        yt = inv_q(yn)
        yp_m3s = inv_q(yp.flatten())
        pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "observed_m3s": yt,
                "predicted_m3s": yp_m3s,
                "residual_m3s": yt - yp_m3s,
            }
        ).to_csv(
            PRED_DIR / f"{mname.lower().replace('-', '_')}_predictions_{label}.csv",
            index=False,
        )

pd.DataFrame(all_metrics).to_csv(MET_DIR / "pi_models_metrics.csv", index=False)

print(f"\n{'Model':<20} {'NSE':>8} {'KGE':>8} {'Peak Bias':>11} {'PBIAS':>8}")
print("-" * 60)
for n, b in [("LSTM", PURE_LSTM), ("Transformer", PURE_TRANSFORMER)]:
    print(
        f"{n:<20} {b['NSE']:>8.4f} {b['KGE']:>8.4f} "
        f"{b['Peak_Bias_%']:>11.2f}% {b['PBIAS_%']:>7.2f}%"
    )
print("-" * 60)
for row in all_metrics:
    print(
        f"{row['model']:<20} {row['NSE']:>8.4f} {row['KGE']:>8.4f} "
        f"{row['Peak_Bias_%']:>11.2f}% {row['PBIAS_%']:>7.2f}%"
    )

# ── plots ──────────────────────────────────────────────────────────────────────
BG, PANEL = "#080f1a", "#0d1825"
BLUE, PURP, ORG, RED = "#3b9eff", "#a855f7", "#f4a261", "#e76f51"
TICK, SPINE, TEXT = "#4a6a82", "#1e3448", "#8aafc4"


def _style(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TICK)
    ax.spines[:].set_color(SPINE)


fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# validation NSE history
ax = fig.add_subplot(gs[0, :2])
_style(ax)
for log_data, c, lbl in [
    (lstm_log, BLUE, "PI-LSTM"),
    (trans_log, PURP, "PI-Transformer"),
]:
    df = pd.DataFrame(log_data)
    ax.plot(df["epoch"], df["val_nse"], color=c, lw=1.8, label=lbl)
ax.axhline(0.75, color="#00b4a0", lw=0.8, ls="--", alpha=0.4)
ax.axhline(0, color=SPINE, lw=0.8, ls="--")
ax.set_title("PI Models — Validation NSE", color="#e8f4f8", fontsize=11)
ax.set_xlabel("Epoch", color=TEXT)
ax.set_ylabel("Val NSE", color=TEXT)
ax.legend(facecolor=PANEL, edgecolor=SPINE, labelcolor=TEXT)

# val loss history
ax = fig.add_subplot(gs[0, 2])
_style(ax)
for log_data, c, lbl in [
    (lstm_log, BLUE, "PI-LSTM"),
    (trans_log, PURP, "PI-Transformer"),
]:
    df = pd.DataFrame(log_data)
    ax.plot(df["epoch"], df["val_loss"], color=c, lw=1.5, label=lbl)
ax.set_title("PI Loss (MSE + λ·WB)", color="#e8f4f8", fontsize=11)
ax.set_xlabel("Epoch", color=TEXT)
ax.set_ylabel("Loss", color=TEXT)
ax.legend(facecolor=PANEL, edgecolor=SPINE, labelcolor=TEXT, fontsize=8)

# test hydrograph — PI-Transformer
ax = fig.add_subplot(gs[1, :])
_style(ax)
dates_t = pd.to_datetime(dates_test)
yp_trans = pi_transformer.predict(
    X_test_aug[:, :, :-1], batch_size=256, verbose=0
).flatten()
qt_true = inv_q(y_test)
qt_pred = inv_q(yp_trans)
pi_tm = next(r for r in all_metrics if r["model"] == "PI-Transformer")

ax.plot(dates_t, qt_true, color=TEXT, lw=1.2, label="Observed", alpha=0.9)
ax.plot(
    dates_t, qt_pred, color=PURP, lw=1.2, ls="--", label="PI-Transformer", alpha=0.9
)
ax.fill_between(dates_t, qt_true, qt_pred, alpha=0.15, color=PURP)
ax.set_title(
    f"PI-Transformer Test Hydrograph (2021–2025)  "
    f"NSE={pi_tm['NSE']:.3f}  KGE={pi_tm['KGE']:.3f}  "
    f"Peak Bias={pi_tm['Peak_Bias_%']:.1f}%",
    color="#e8f4f8",
    fontsize=11,
)
ax.set_ylabel("Discharge (m³/s)", color=TEXT)
ax.legend(facecolor=PANEL, edgecolor=SPINE, labelcolor=TEXT)

# pure vs PI comparison bars
ax = fig.add_subplot(gs[2, :2])
_style(ax)
met_names = ["NSE", "KGE", "Log_NSE"]
pi_lstm_m = next(r for r in all_metrics if r["model"] == "PI-LSTM")
pi_trans_m = next(r for r in all_metrics if r["model"] == "PI-Transformer")
model_bars = [
    ("LSTM", [PURE_LSTM[m] for m in met_names], "#4a6a82"),
    ("PI-LSTM", [pi_lstm_m[m] for m in met_names], BLUE),
    ("Transformer", [PURE_TRANSFORMER[m] for m in met_names], "#555555"),
    ("PI-Transformer", [pi_trans_m[m] for m in met_names], PURP),
]
x = np.arange(len(met_names))
w = 0.2
for i, (n, vals, c) in enumerate(model_bars):
    ax.bar(x + i * w, vals, w, label=n, color=c, alpha=0.85)
ax.axhline(0.75, color="#00b4a0", lw=1, ls=":", alpha=0.7)
ax.set_xticks(x + 1.5 * w)
ax.set_xticklabels(met_names, color=TEXT)
ax.set_ylabel("Score", color=TEXT)
ax.set_title("Pure AI vs Physics-Informed", color="#e8f4f8", fontsize=11)
ax.legend(facecolor=PANEL, edgecolor=SPINE, labelcolor=TEXT, fontsize=7)
ax.set_ylim(-0.1, 1.1)

# water balance residuals
ax = fig.add_subplot(gs[2, 2])
_style(ax)
master_test = master[master.date.dt.year >= 2021].copy()
if len(master_test) > 0:
    pet_te = hamon_pet(
        master_test["temp_mean_c"].values, master_test["date"].dt.dayofyear.values
    )
    n = len(qt_pred)
    P = master_test["precip_mm_day"].values[:n]
    ET = pet_te[:n]
    dS = master_test["swe_mm"].diff().clip(upper=0).abs().fillna(0).values[:n]
    WB_obs = P - ET - qt_true - dS
    WB_pred = P - ET - qt_pred - dS
    ax.hist(
        WB_obs,
        bins=40,
        color=TEXT,
        alpha=0.6,
        density=True,
        label=f"Observed (mean={WB_obs.mean():.3f})",
    )
    ax.hist(
        WB_pred,
        bins=40,
        color=PURP,
        alpha=0.6,
        density=True,
        label=f"PI-Transformer (mean={WB_pred.mean():.3f})",
    )
    ax.axvline(0, color=RED, lw=1.5, ls="--")
ax.set_xlabel("WB Residual (mm/day)", color=TEXT)
ax.set_ylabel("Density", color=TEXT)
ax.set_title("Water Balance Residuals", color="#e8f4f8", fontsize=11)
ax.legend(facecolor=PANEL, edgecolor=SPINE, labelcolor=TEXT, fontsize=7)

fig.suptitle(
    f"Physics-Informed Models — Nahr Ibrahim Watershed\n"
    f"λ={LAMBDA_WB}  |  P − ET − Q − ΔS ≈ 0",
    color="#e8f4f8",
    fontsize=13,
    y=0.98,
    fontfamily="monospace",
)

plt.savefig(
    FIG_DIR / "pi_models_results.png", dpi=150, bbox_inches="tight", facecolor=BG
)
plt.show()

# ── summary ────────────────────────────────────────────────────────────────────
print(f"\nλ={LAMBDA_WB}  |  P − ET − Q − ΔS ≈ 0  |  Hamon PET")
print(
    f"\n{'Model':<20} {'NSE':>8} {'KGE':>8} {'RMSE':>8} {'Peak Bias':>11} {'PBIAS':>8}"
)
print("-" * 67)
for n, nse, kge, rmse, pb, pbias in [
    ("LSTM", 0.518, 0.503, 0.292, -43.3, -5.0),
    ("Transformer", 0.603, 0.632, 0.268, -41.2, -3.6),
]:
    print(f"{n:<20} {nse:>8.4f} {kge:>8.4f} {rmse:>8.4f} {pb:>10.2f}% {pbias:>7.2f}%")
print("-" * 67)
for row in all_metrics:
    print(
        f"{row['model']:<20} {row['NSE']:>8.4f} {row['KGE']:>8.4f} "
        f"{row['RMSE']:>8.4f} {row['Peak_Bias_%']:>10.2f}% {row['PBIAS_%']:>7.2f}%"
    )
