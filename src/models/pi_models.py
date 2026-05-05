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
SEQ_DIR   = ROOT / "data"    / "sequences"
SPLIT_DIR = ROOT / "data"    / "splits"
MASTER_DIR= ROOT / "data"    / "master"
MODEL_DIR = ROOT / "models"
FIG_DIR   = ROOT / "results" / "figures"
MET_DIR   = ROOT / "results" / "metrics"
PRED_DIR  = ROOT / "results" / "predictions"

for d in [MODEL_DIR/"trained", MODEL_DIR/"checkpoints",
          MODEL_DIR/"configs",  FIG_DIR, MET_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Physics-informed hyperparameter
# λ controls weight of water balance penalty vs MSE
# 0.05 = 5% penalty — conservative and stable
LAMBDA_WB = 0.05

# Feature column indices in the sequence array
# Must match FEATURE_COLS in split.py exactly
FEATURE_COLS = [
    "precip_mm_day", "precip_3day",    "precip_7day",     # indices 0,1,2
    "temp_mean_c",   "temp_max_c",     "temp_min_c",      # indices 3,4,5
    "temp_range_c",  "swe_mm",         "swe_delta",       # indices 6,7,8
    "snow_cover_pct","month_sin",      "month_cos",       # indices 9,10,11
    "soil_moisture_mm","sm_7day_mean", "sm_anomaly",      # indices 12,13,14
    "pet_mm_day",                                         # index 15
]

# Indices used in water balance computation
IDX_PRECIP = 0   # precip_mm_day
IDX_TEMP   = 3   # temp_mean_c
IDX_SWE    = 7   # swe_mm
IDX_MONTH  = 10  # month_sin (for daylight hours)

tf.random.set_seed(42)
np.random.seed(42)

print("=" * 65)
print("  Nahr Ibrahim — Physics-Informed Models")
print("=" * 65)
print(f"  Lambda (λ) : {LAMBDA_WB}")
print(f"  Constraint : Water balance (P - ET - Q - ΔS ≈ 0)")
print(f"  PET method : Hamon (1961) temperature-based")

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1/7] Loading data ...")

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

# Load master dataset for PET computation on training set
master = pd.read_csv(
    MASTER_DIR / "nahr_ibrahim_master_model.csv",
    parse_dates=["date"]
)

print(f"  X_train : {X_train.shape}")
print(f"  X_val   : {X_val.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  Q range : [{q_min:.3f}, {q_max:.3f}] m³/s")

# =============================================================================
# 2. HAMON PET COMPUTATION
# =============================================================================
print("\n[2/7] Computing Hamon PET ...")

def hamon_pet(temp_c: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """
    Hamon (1961) potential evapotranspiration.

    PET = 0.1651 × Ld × ρsat
    where:
        Ld    = daylight hours
        ρsat  = saturated vapour density at temp_c

    Reference: Hamon, W.R. (1961). Estimating potential evapotranspiration.
               Journal of the Hydraulics Division, 87(3), 107–120.
    """
    # Daylight hours — sinusoidal approximation for Lebanon (~34°N)
    Ld = 12 + 4 * np.sin(2 * np.pi * (doy - 80) / 365)

    # Saturated vapour density (g/m³)
    rho_sat = (216.7 * 0.6108 *
               np.exp(17.27 * temp_c / (temp_c + 237.3)) /
               (temp_c + 273.3))

    pet = 0.1651 * Ld * rho_sat  # mm/day
    return np.clip(pet, 0, None)


# Compute PET for all splits using master dataset dates
def get_pet_for_split(dates_array: np.ndarray) -> np.ndarray:
    """
    Match dates to master dataset and compute daily PET.
    Returns array of PET values aligned with sequence dates.
    """
    dates_pd = pd.to_datetime(dates_array)
    merged   = pd.DataFrame({"date": dates_pd}).merge(
        master[["date", "temp_mean_c"]], on="date", how="left"
    )
    temp = merged["temp_mean_c"].fillna(merged["temp_mean_c"].mean()).values
    doy  = dates_pd.dayofyear.values
    return hamon_pet(temp, doy).astype(np.float32)


pet_train = get_pet_for_split(dates_train)
pet_val   = get_pet_for_split(dates_val)
pet_test  = get_pet_for_split(dates_test)

print(f"  PET range (train): {pet_train.min():.2f} – "
      f"{pet_train.max():.2f} mm/day")
print(f"  PET mean  (train): {pet_train.mean():.2f} mm/day")

# Normalize PET using training scaler range for precip (same order of magnitude)
pet_scale = scaler.loc["precip_mm_day", "max"] - \
            scaler.loc["precip_mm_day", "min"]
pet_train_norm = (pet_train / pet_scale).astype(np.float32)
pet_val_norm   = (pet_val   / pet_scale).astype(np.float32)
pet_test_norm  = (pet_test  / pet_scale).astype(np.float32)

# =============================================================================
# 3. PHYSICS-INFORMED LOSS FUNCTION
# =============================================================================
def make_pi_loss(X_batch_pet: tf.Tensor, lambda_wb: float = 0.05):
    """
    Factory function returning physics-informed loss for a given batch.

    Water balance residual:
        R = P - ET - Q_pred - ΔS
        penalty = mean(R²)

    All terms are in normalized units (0–1 scale).

    P     = mean precip over lookback window (index 0 in features)
    ET    = PET passed as external tensor
    Q     = model prediction (what we penalize)
    ΔS    = mean swe_delta over lookback window (index 8 in features)
    """
    def loss(y_true, y_pred):
        # Standard MSE — same as pure AI models
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # Extract mean precip over 30-day window
        # X shape: (batch, lookback, features) — averaged over time
        P  = tf.reduce_mean(X_batch_pet[:, :, IDX_PRECIP], axis=1)

        # ΔS: mean SWE delta (snowmelt/accumulation proxy)
        dS = tf.reduce_mean(X_batch_pet[:, :, IDX_SWE], axis=1)

        # ET: Hamon PET passed externally (normalized)
        ET = X_batch_pet[:, 0, -1]  # appended as last feature

        # Q: model prediction (normalized)
        Q  = tf.squeeze(y_pred, axis=-1)

        # Water balance residual: P - ET - Q - ΔS ≈ 0
        residual = P - ET - Q - dS
        wb_penalty = tf.reduce_mean(tf.square(residual))

        return mse + lambda_wb * wb_penalty

    return loss


# =============================================================================
# 4. PREPARE AUGMENTED INPUTS (append PET as extra feature)
# =============================================================================
def augment_with_pet(X: np.ndarray, pet: np.ndarray) -> np.ndarray:
    """
    Append PET as an additional feature to the sequence array.
    PET is the same value for all 30 timesteps in a window
    (it's a daily scalar broadcast across the window).
    Shape: (samples, 30, 12) → (samples, 30, 13)
    """
    # pet shape: (samples,) → (samples, 30, 1)
    pet_expanded = np.tile(
        pet[:, np.newaxis, np.newaxis], (1, X.shape[1], 1)
    ).astype(np.float32)
    return np.concatenate([X, pet_expanded], axis=2)


X_train_aug = augment_with_pet(X_train, pet_train_norm)
X_val_aug   = augment_with_pet(X_val,   pet_val_norm)
X_test_aug  = augment_with_pet(X_test,  pet_test_norm)

print(f"\n  Augmented X shape: {X_train_aug.shape} "
      f"(added PET as feature 13)")

N_FEATURES_AUG = X_train_aug.shape[2]  # 13

# =============================================================================
# 5. METRIC & HELPERS
# =============================================================================
def nse_metric(y_true, y_pred):
    num = tf.reduce_sum(tf.square(y_true - y_pred))
    den = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - num / (den + tf.keras.backend.epsilon())


def inverse_transform_q(q_norm, q_min, q_max):
    return np.clip(q_norm * (q_max - q_min) + q_min, 0, None)


def compute_metrics(y_true_norm, y_pred_norm, q_min, q_max, split_name):
    y_true = inverse_transform_q(y_true_norm, q_min, q_max)
    y_pred = inverse_transform_q(y_pred_norm.flatten(), q_min, q_max)

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
        "split"        : split_name,
        "NSE"          : round(nse_val,  4),
        "KGE"          : round(kge_val,  4),
        "RMSE"         : round(rmse_val, 4),
        "MAE"          : round(mae_val,  4),
        "PBIAS_%"      : round(pbias,    2),
        "Peak_Bias_%"  : round(peak_bias,2),
        "Log_NSE"      : round(log_nse,  4),
    }, y_true, y_pred


# =============================================================================
# 6. TRAIN FUNCTION
# =============================================================================
def train_pi_model(model_name: str,
                   model: tf.keras.Model,
                   X_tr: np.ndarray, y_tr: np.ndarray,
                   X_v:  np.ndarray, y_v:  np.ndarray,
                   epochs: int = 150,
                   patience: int = 20) -> tuple:

    print(f"\n  Training {model_name} ...")

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3, clipnorm=1.0
    )

    # Use standard MSE for compilation — physics penalty
    # is implemented via custom training loop below
    model.compile(
        optimizer = optimizer,
        loss      = make_pi_loss(
            tf.constant(X_tr[:32], dtype=tf.float32),
            LAMBDA_WB
        ),
        metrics   = [nse_metric, "mae",
                     tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )

    # Re-compile with proper loss that uses input X
    # Using standard MSE + manual water balance via sample_weight approach
    # Simplest stable approach: MSE loss with water balance as extra metric

    model.compile(
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-3, clipnorm=1.0
        ),
        loss      = "mse",
        metrics   = [nse_metric, "mae",
                     tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )

    log_path = MODEL_DIR / "configs" / f"{model_name}_training_log.csv"
    ckpt_path= MODEL_DIR / "checkpoints" / f"{model_name}_best.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=patience, min_delta=1e-5,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path), monitor="val_loss",
            mode="min", save_best_only=True, verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=0.5, patience=10, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(log_path), append=False),
    ]

    # Custom training loop implementing physics-informed loss
    # This gives us full control over the gradient computation
    @tf.function
    def train_step(x_batch, y_batch, pet_batch):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)

            # MSE component
            mse = tf.reduce_mean(tf.square(y_batch - y_pred))

            # Water balance components (normalized units)
            P  = tf.reduce_mean(x_batch[:, :, IDX_PRECIP], axis=1)
            dS = tf.reduce_mean(x_batch[:, :, IDX_SWE],    axis=1)
            ET = pet_batch
            Q  = tf.squeeze(y_pred, axis=-1)

            residual   = P - ET - Q - dS
            wb_penalty = tf.reduce_mean(tf.square(residual))

            total_loss = mse + LAMBDA_WB * wb_penalty

        grads = tape.gradient(total_loss, model.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        optimizer.apply_gradients(
            zip(grads, model.trainable_variables)
        )
        return total_loss, mse, wb_penalty

    @tf.function
    def val_step(x_batch, y_batch, pet_batch):
        y_pred = model(x_batch, training=False)
        mse    = tf.reduce_mean(tf.square(y_batch - y_pred))
        P      = tf.reduce_mean(x_batch[:, :, IDX_PRECIP], axis=1)
        dS     = tf.reduce_mean(x_batch[:, :, IDX_SWE],    axis=1)
        ET     = pet_batch
        Q      = tf.squeeze(y_pred, axis=-1)
        residual   = P - ET - Q - dS
        wb_penalty = tf.reduce_mean(tf.square(residual))
        return mse + LAMBDA_WB * wb_penalty

    batch_size = 32
    best_val_loss  = np.inf
    patience_count = 0
    best_weights   = None
    history_log    = []

    start_time = datetime.now()

    for epoch in range(epochs):
        # Shuffle training data
        idx = np.random.permutation(len(X_tr))
        X_s = X_tr[idx];  y_s = y_tr[idx]

        # Extract PET from augmented features (last feature)
        pet_s = X_s[:, 0, -1]

        # Training batches
        train_losses = []
        for i in range(0, len(X_s), batch_size):
            xb = tf.constant(X_s[i:i+batch_size, :, :-1], dtype=tf.float32)
            yb = tf.constant(y_s[i:i+batch_size, np.newaxis], dtype=tf.float32)
            pb = tf.constant(pet_s[i:i+batch_size], dtype=tf.float32)
            tl, mse, wb = train_step(xb, yb, pb)
            train_losses.append(float(tl))

        # Validation
        val_losses = []
        pet_v = X_v[:, 0, -1]
        for i in range(0, len(X_v), batch_size):
            xb = tf.constant(X_v[i:i+batch_size, :, :-1], dtype=tf.float32)
            yb = tf.constant(y_v[i:i+batch_size, np.newaxis], dtype=tf.float32)
            pb = tf.constant(pet_v[i:i+batch_size], dtype=tf.float32)
            vl = val_step(xb, yb, pb)
            val_losses.append(float(vl))

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)

        # NSE on validation
        y_pred_v = model.predict(
            X_v[:, :, :-1], batch_size=256, verbose=0
        ).flatten()
        val_nse = 1 - np.sum((y_v - y_pred_v)**2) / \
                      np.sum((y_v - np.mean(y_v))**2)

        history_log.append({
            "epoch": epoch, "loss": train_loss,
            "val_loss": val_loss, "val_nse": val_nse
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>4} | "
                  f"train={train_loss:.5f} | "
                  f"val={val_loss:.5f} | "
                  f"val_nse={val_nse:.4f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss  = val_loss
            patience_count = 0
            best_weights   = model.get_weights()
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

        # Learning rate reduction
        if patience_count > 0 and patience_count % 10 == 0:
            current_lr = float(optimizer.learning_rate)
            new_lr     = max(current_lr * 0.5, 1e-6)
            optimizer.learning_rate.assign(new_lr)
            print(f"  LR reduced to {new_lr:.2e}")

    # Restore best weights
    if best_weights is not None:
        model.set_weights(best_weights)

    elapsed = datetime.now() - start_time

    # Save log
    pd.DataFrame(history_log).to_csv(log_path, index=False)

    print(f"\n  Training complete in {elapsed}")
    print(f"  Best val loss: {best_val_loss:.5f}")

    return model, history_log, elapsed


# =============================================================================
# 7. BUILD PI-LSTM
# =============================================================================
print("\n[3/7] Building PI-LSTM ...")

# Same architecture as LSTM v1 — only loss function changes
def build_pi_lstm(n_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(30, n_features), name="input")
    x = inputs
    for i, units in enumerate([128, 64]):
        return_seq = (i < 1)
        x = tf.keras.layers.LSTM(
            units, return_sequences=return_seq,
            dropout=0.3, recurrent_dropout=0.2,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name=f"lstm_{i+1}"
        )(x)
        x = tf.keras.layers.Dropout(0.3, name=f"drop_{i+1}")(x)
    x = tf.keras.layers.Dense(32, name="dense_1")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    outputs = tf.keras.layers.Dense(1, activation="linear", name="output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs,
                          name="PI_LSTM_NahrIbrahim")


pi_lstm = build_pi_lstm(N_FEATURES_AUG - 1)  # exclude PET from input
pi_lstm.summary()

# =============================================================================
# 8. BUILD PI-TRANSFORMER
# =============================================================================
print("\n[4/7] Building PI-Transformer ...")


def build_pi_transformer(n_features: int,
                         d_model: int = 64,
                         n_heads: int = 4,
                         ffn_dim: int = 128,
                         n_blocks: int = 3,
                         dropout: float = 0.2) -> tf.keras.Model:
    # Same architecture as pure Transformer
    inputs = tf.keras.Input(shape=(30, n_features), name="input")

    # Positional encoding
    positions = np.arange(30)[:, np.newaxis]
    dims      = np.arange(d_model)[np.newaxis, :]
    angles    = positions / np.power(10000, (2*(dims//2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    pos_enc = tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)

    x = tf.keras.layers.Dense(d_model, name="input_proj")(inputs)
    x = x + pos_enc
    x = tf.keras.layers.Dropout(dropout)(x)

    for i in range(n_blocks):
        # Multi-head attention
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model//n_heads,
            dropout=dropout, name=f"mha_{i+1}"
        )(x, x)
        attn = tf.keras.layers.Dropout(dropout)(attn)
        x    = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)

        # FFN
        ffn  = tf.keras.layers.Dense(ffn_dim, activation="relu")(x)
        ffn  = tf.keras.layers.Dropout(dropout)(ffn)
        ffn  = tf.keras.layers.Dense(d_model)(ffn)
        ffn  = tf.keras.layers.Dropout(dropout)(ffn)
        x    = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout/2)(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout/2)(x)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs,
                          name="PI_Transformer_NahrIbrahim")


pi_transformer = build_pi_transformer(N_FEATURES_AUG - 1)
pi_transformer.summary()

# =============================================================================
# 9. TRAIN BOTH PI MODELS
# =============================================================================
print("\n[5/7] Training physics-informed models ...")

# PI-LSTM
pi_lstm, lstm_log, lstm_time = train_pi_model(
    "pi_lstm", pi_lstm,
    X_train_aug, y_train,
    X_val_aug,   y_val,
    epochs=150, patience=20
)
pi_lstm.save(str(MODEL_DIR / "trained" / "pi_lstm_final.keras"))
print(f"  Saved → models/trained/pi_lstm_final.keras")

# PI-Transformer
pi_transformer, trans_log, trans_time = train_pi_model(
    "pi_transformer", pi_transformer,
    X_train_aug, y_train,
    X_val_aug,   y_val,
    epochs=150, patience=20
)
pi_transformer.save(
    str(MODEL_DIR / "trained" / "pi_transformer_final.keras")
)
print(f"  Saved → models/trained/pi_transformer_final.keras")

# =============================================================================
# 10. EVALUATE BOTH PI MODELS
# =============================================================================
print("\n[6/7] Evaluating ...")

# Pure AI baselines for comparison
PURE_LSTM        = {"NSE":0.515,"KGE":0.423,"Peak_Bias_%":-51.7,"PBIAS_%":0.0}
PURE_TRANSFORMER = {"NSE":0.680,"KGE":0.671,"Peak_Bias_%":-37.0,"PBIAS_%":2.3}

all_metrics = []

for name, model, log in [
    ("PI-LSTM",        pi_lstm,        lstm_log),
    ("PI-Transformer", pi_transformer, trans_log),
]:
    # Predict using features WITHOUT PET (PET only used in loss)
    yp_train = model.predict(
        X_train_aug[:,:,:-1], batch_size=256, verbose=0)
    yp_val   = model.predict(
        X_val_aug[:,:,:-1],   batch_size=256, verbose=0)
    yp_test  = model.predict(
        X_test_aug[:,:,:-1],  batch_size=256, verbose=0)

    m_tr, _, _              = compute_metrics(y_train, yp_train, q_min, q_max, "Train")
    m_v,  _, _              = compute_metrics(y_val,   yp_val,   q_min, q_max, "Validation")
    m_te, qtest_true, qtest_pred = compute_metrics(
        y_test, yp_test, q_min, q_max, "Test")

    all_metrics.append({**m_te, "model": name})

    # Save predictions
    for sname, dates, yp in [
        ("train", dates_train, yp_train),
        ("val",   dates_val,   yp_val),
        ("test",  dates_test,  yp_test),
    ]:
        yt = inverse_transform_q(
            {"train":y_train,"val":y_val,"test":y_test}[sname],
            q_min, q_max
        )
        yp_m3s = inverse_transform_q(yp.flatten(), q_min, q_max)
        pd.DataFrame({
            "date"         : pd.to_datetime(dates),
            "observed_m3s" : yt,
            "predicted_m3s": yp_m3s,
            "residual_m3s" : yt - yp_m3s,
        }).to_csv(
            PRED_DIR / f"{name.lower().replace('-','_')}"
                       f"_predictions_{sname}.csv",
            index=False
        )

# Save metrics
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(MET_DIR / "pi_models_metrics.csv", index=False)

# Print comparison table
print(f"\n  {'Model':<18} {'NSE':>8} {'KGE':>8} "
      f"{'Peak Bias':>11} {'PBIAS':>8}")
print(f"  {'-'*58}")
for name, base in [("LSTM",        PURE_LSTM),
                   ("Transformer",  PURE_TRANSFORMER)]:
    print(f"  {name:<18} {base['NSE']:>8.4f} {base['KGE']:>8.4f} "
          f"{base['Peak_Bias_%']:>11.2f}% {base['PBIAS_%']:>7.2f}%")

print(f"  {'-'*58}")
for row in all_metrics:
    print(f"  {row['model']:<18} {row['NSE']:>8.4f} {row['KGE']:>8.4f} "
          f"{row['Peak_Bias_%']:>11.2f}% {row['PBIAS_%']:>7.2f}%")

# =============================================================================
# 11. VISUALIZATION
# =============================================================================
print("\n[7/7] Generating plots ...")

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#080f1a")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Training history comparison ──
ax0 = fig.add_subplot(gs[0, :2])
ax0.set_facecolor("#0d1825")

for log_data, color, label in [
    (lstm_log,  "#3b9eff", "PI-LSTM"),
    (trans_log, "#a855f7", "PI-Transformer"),
]:
    df_log = pd.DataFrame(log_data)
    ax0.plot(df_log["epoch"], df_log["val_nse"],
             color=color, linewidth=1.8, label=label)

ax0.axhline(0.75, color="#00b4a0", linewidth=0.8,
            linestyle="--", alpha=0.4, label="Good (0.75)")
ax0.axhline(0, color="#4a6a82", linewidth=0.8, linestyle="--")
ax0.set_title("PI Models — Validation NSE History",
              color="#e8f4f8", fontsize=11)
ax0.set_xlabel("Epoch", color="#8aafc4")
ax0.set_ylabel("Validation NSE", color="#8aafc4")
ax0.tick_params(colors="#4a6a82")
ax0.spines[:].set_color("#1e3448")
ax0.legend(facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4")
ax0.set_facecolor("#0d1825")

# ── Loss history ──
ax0b = fig.add_subplot(gs[0, 2])
ax0b.set_facecolor("#0d1825")
for log_data, color, label in [
    (lstm_log,  "#3b9eff", "PI-LSTM"),
    (trans_log, "#a855f7", "PI-Transformer"),
]:
    df_log = pd.DataFrame(log_data)
    ax0b.plot(df_log["epoch"], df_log["val_loss"],
              color=color, linewidth=1.5, label=label)
ax0b.set_title("PI Loss (MSE + λ·WB)", color="#e8f4f8", fontsize=11)
ax0b.set_xlabel("Epoch", color="#8aafc4")
ax0b.set_ylabel("Loss", color="#8aafc4")
ax0b.tick_params(colors="#4a6a82")
ax0b.spines[:].set_color("#1e3448")
ax0b.legend(facecolor="#0d1825", edgecolor="#1e3448",
            labelcolor="#8aafc4", fontsize=8)
ax0b.set_facecolor("#0d1825")

# ── Test hydrograph — PI-Transformer ──
ax1 = fig.add_subplot(gs[1, :])
ax1.set_facecolor("#0d1825")
dates_t = pd.to_datetime(dates_test)
yp_test_trans = pi_transformer.predict(
    X_test_aug[:,:,:-1], batch_size=256, verbose=0).flatten()
qt_true = inverse_transform_q(y_test, q_min, q_max)
qt_pred = inverse_transform_q(yp_test_trans, q_min, q_max)

ax1.plot(dates_t, qt_true, color="#8aafc4",
         linewidth=1.2, label="Observed", alpha=0.9)
ax1.plot(dates_t, qt_pred, color="#a855f7",
         linewidth=1.2, linestyle="--", label="PI-Transformer", alpha=0.9)
ax1.fill_between(dates_t, qt_true, qt_pred,
                 alpha=0.15, color="#a855f7", label="Error")

pi_trans_metrics = [r for r in all_metrics
                    if r["model"] == "PI-Transformer"][0]
ax1.set_title(
    f"PI-Transformer Test Hydrograph (2021–2025) — "
    f"NSE={pi_trans_metrics['NSE']:.3f}  "
    f"KGE={pi_trans_metrics['KGE']:.3f}  "
    f"Peak Bias={pi_trans_metrics['Peak_Bias_%']:.1f}%",
    color="#e8f4f8", fontsize=11
)
ax1.set_ylabel("Discharge (m³/s)", color="#8aafc4")
ax1.tick_params(colors="#4a6a82")
ax1.spines[:].set_color("#1e3448")
ax1.legend(facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4")
ax1.set_facecolor("#0d1825")

# ── Pure vs Physics-Informed comparison ──
ax2 = fig.add_subplot(gs[2, :2])
ax2.set_facecolor("#0d1825")
metric_names = ["NSE", "KGE", "Log_NSE"]
models_cmp   = [
    ("LSTM",        [PURE_LSTM["NSE"],       PURE_LSTM["KGE"],       0.633], "#4a6a82"),
    ("PI-LSTM",            [m["NSE"] for m in all_metrics if m["model"]=="PI-LSTM"][0:1]*3,
                           "#3b9eff"),
    ("Transformer", [PURE_TRANSFORMER["NSE"],PURE_TRANSFORMER["KGE"],0.671], "#555555"),
    ("PI-Transformer",     [m["NSE"] for m in all_metrics if m["model"]=="PI-Transformer"][0:1]*3,
                           "#a855f7"),
]

models_cmp = [
    ("LSTM",        [PURE_LSTM["NSE"], PURE_LSTM["KGE"], 0.633],        "#4a6a82"),
    ("PI-LSTM",            [next(m["NSE"] for m in all_metrics if m["model"]=="PI-LSTM"),
                            next(m["KGE"] for m in all_metrics if m["model"]=="PI-LSTM"),
                            next(m["Log_NSE"] for m in all_metrics if m["model"]=="PI-LSTM")],
                           "#3b9eff"),
    ("Transformer",        [PURE_TRANSFORMER["NSE"],PURE_TRANSFORMER["KGE"],0.671], "#555555"),
    ("PI-Transformer",     [next(m["NSE"] for m in all_metrics if m["model"]=="PI-Transformer"),
                            next(m["KGE"] for m in all_metrics if m["model"]=="PI-Transformer"),
                            next(m["Log_NSE"] for m in all_metrics if m["model"]=="PI-Transformer")],
                           "#a855f7"),
]

x = np.arange(len(metric_names))
w = 0.2
for i, (name, vals, color) in enumerate(models_cmp):
    ax2.bar(x + i*w, vals, w, label=name, color=color, alpha=0.85)

ax2.axhline(0.75, color="#00b4a0", linewidth=1,
            linestyle=":", alpha=0.7, label="Good (0.75)")
ax2.set_xticks(x + 1.5*w)
ax2.set_xticklabels(metric_names, color="#8aafc4")
ax2.set_ylabel("Score", color="#8aafc4")
ax2.set_title("Pure AI vs Physics-Informed",
              color="#e8f4f8", fontsize=11)
ax2.tick_params(colors="#4a6a82")
ax2.spines[:].set_color("#1e3448")
ax2.legend(facecolor="#0d1825", edgecolor="#1e3448",
           labelcolor="#8aafc4", fontsize=7)
ax2.set_ylim(-0.1, 1.1)
ax2.set_facecolor("#0d1825")

# ── Water balance residuals ──
ax3 = fig.add_subplot(gs[2, 2])
ax3.set_facecolor("#0d1825")

master_test = master[master.date.dt.year >= 2021].copy()
if len(master_test) > 0:
    pet_test_vals = hamon_pet(
        master_test["temp_mean_c"].values,
        master_test["date"].dt.dayofyear.values
    )
    P  = master_test["precip_mm_day"].values[:len(qt_pred)]
    ET = pet_test_vals[:len(qt_pred)]
    dS = master_test["swe_mm"].diff().clip(upper=0).abs().fillna(0).values[:len(qt_pred)]
    WB_pred = P - ET - qt_pred - dS
    WB_obs  = P - ET - qt_true  - dS

    ax3.hist(WB_obs,  bins=40, color="#8aafc4", alpha=0.6,
             density=True, label=f"Observed Q (mean={WB_obs.mean():.3f})")
    ax3.hist(WB_pred, bins=40, color="#a855f7", alpha=0.6,
             density=True, label=f"PI-Transformer (mean={WB_pred.mean():.3f})")
    ax3.axvline(0, color="#e76f51", linewidth=1.5, linestyle="--")

ax3.set_xlabel("Water Balance Residual (mm/day)", color="#8aafc4")
ax3.set_ylabel("Density", color="#8aafc4")
ax3.set_title("Water Balance Residuals", color="#e8f4f8", fontsize=11)
ax3.tick_params(colors="#4a6a82")
ax3.spines[:].set_color("#1e3448")
ax3.legend(facecolor="#0d1825", edgecolor="#1e3448",
           labelcolor="#8aafc4", fontsize=7)
ax3.set_facecolor("#0d1825")

fig.suptitle("Physics-Informed Models — Nahr Ibrahim Watershed\n"
             f"λ={LAMBDA_WB} | Water Balance: P − ET − Q − ΔS ≈ 0",
             color="#e8f4f8", fontsize=13, y=0.98, fontfamily="monospace")

plt.savefig(FIG_DIR / "pi_models_results.png",
            dpi=150, bbox_inches="tight", facecolor="#080f1a")
plt.show()

# =============================================================================
# 12. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  PHYSICS-INFORMED MODELS SUMMARY")
print("=" * 65)
print(f"\n  Lambda (λ)   : {LAMBDA_WB}")
print(f"  Constraint   : P − ET − Q − ΔS ≈ 0")
print(f"  PET method   : Hamon (1961)")

print(f"\n  {'Model':<20} {'NSE':>8} {'KGE':>8} "
      f"{'RMSE':>8} {'Peak Bias':>11} {'PBIAS':>8}")
print(f"  {'-'*67}")

baselines = [
    ("LSTM",        0.518, 0.503, 0.292, -43.3, -5.0),
    ("Transformer", 0.603, 0.632, 0.268, -41.2, -3.6),
]
for name, nse, kge, rmse, pb, pbias in baselines:
    print(f"  {name:<20} {nse:>8.4f} {kge:>8.4f} "
          f"{rmse:>8.4f} {pb:>10.2f}% {pbias:>7.2f}%")

print(f"  {'-'*67}")
for row in all_metrics:
    print(f"  {row['model']:<20} {row['NSE']:>8.4f} {row['KGE']:>8.4f} "
          f"{row['RMSE']:>8.4f} {row['Peak_Bias_%']:>10.2f}% "
          f"{row['PBIAS_%']:>7.2f}%")

print(f"\n  Files saved:")
print(f"    models/trained/pi_lstm_final.keras")
print(f"    models/trained/pi_transformer_final.keras")
print(f"    results/metrics/pi_models_metrics.csv")
print(f"    results/figures/pi_models_results.png")
print("=" * 65)