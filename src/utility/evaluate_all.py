import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

ROOT = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")
SEQ_DIR = ROOT / "data" / "sequences"
SPLIT_DIR = ROOT / "data" / "splits"
MET_DIR = ROOT / "results" / "metrics"

# ── Load test sequences and scaler ────────────────────────────
y_test = np.load(SEQ_DIR / "y_test.npy")
scaler = pd.read_csv(SPLIT_DIR / "scaler_params.csv", index_col=0)
q_min = scaler.loc["discharge_m3s", "min"]
q_max = scaler.loc["discharge_m3s", "max"]


def inverse_q(q):
    return np.clip(q * (q_max - q_min) + q_min, 0, None)


y_obs = inverse_q(y_test)


# ── Metrics ────────────────────────────────────────────────────
def metrics(obs, pred, name):
    nse = 1 - np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    r = pearsonr(obs, pred)[0]
    alpha = np.std(pred) / np.std(obs)
    beta = np.mean(pred) / np.mean(obs)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    mae = np.mean(np.abs(obs - pred))
    pbias = 100 * np.sum(pred - obs) / np.sum(obs)
    p95 = np.percentile(obs, 95)
    peak = (
        100
        * (np.mean(pred[obs >= p95]) - np.mean(obs[obs >= p95]))
        / np.mean(obs[obs >= p95])
    )
    lnse = 1 - np.sum((np.log(obs + 0.01) - np.log(pred + 0.01)) ** 2) / np.sum(
        (np.log(obs + 0.01) - np.mean(np.log(obs + 0.01))) ** 2
    )
    return {
        "Model": name,
        "NSE": round(nse, 4),
        "KGE": round(kge, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "PBIAS_%": round(pbias, 2),
        "Peak_Bias_%": round(peak, 2),
        "LogNSE": round(lnse, 4),
    }


# ── Load predictions and compute ───────────────────────────────
import tensorflow as tf
import os, warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

X_test = np.load(SEQ_DIR / "X_test.npy")

from tensorflow.keras import layers


class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        pos = np.arange(max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = pos / np.power(10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pos_encoding = tf.cast(angles[np.newaxis], dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, : tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len, "d_model": self.d_model})
        return cfg


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, n_heads, ffn_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attention = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout
        )
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ffn_dim, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(d_model),
            ]
        )
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.drop1(self.attention(x, x, training=training), training=training)
        x = self.ln1(x + attn)
        ffn = self.drop2(self.ffn(x, training=training), training=training)
        return self.ln2(x + ffn)

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


def nse_metric(y_true, y_pred):
    num = tf.reduce_sum(tf.square(y_true - y_pred))
    den = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - num / (den + tf.keras.backend.epsilon())


custom_obj = {
    "nse_metric": nse_metric,
    "PositionalEncoding": PositionalEncoding,
    "TransformerEncoderBlock": TransformerEncoderBlock,
}

MODEL_DIR = ROOT / "models" / "trained"
model_files = {
    "LSTM": "lstm_final.keras",
    "CNN-LSTM": "cnn_lstm_final.keras",
    "Transformer": "transformer_final.keras",
    "PI-LSTM": "pi_lstm_final.keras",
    "PI-Transformer": "pi_transformer_final.keras",
    "TCN": "tcn_final.keras",
    "TCAN": "tcan_final.keras",
}

rows = []
print(f"\n{'='*75}")
print(
    f"  {'Model':<18} {'NSE':>7} {'KGE':>7} {'RMSE':>7} {'MAE':>7} "
    f"{'PBIAS':>8} {'Peak':>8} {'LogNSE':>8}"
)
print(f"  {'-'*73}")

for name, fname in model_files.items():
    path = MODEL_DIR / fname
    if not path.exists():
        print(f"  {name:<18} — file not found, skipping")
        continue
    try:
        model = tf.keras.models.load_model(
            str(path), custom_objects=custom_obj, compile=False
        )
        y_norm = model.predict(X_test, batch_size=512, verbose=0).flatten()
        y_pred = inverse_q(y_norm)
        m = metrics(y_obs, y_pred, name)
        rows.append(m)
        print(
            f"  {name:<18} {m['NSE']:>7} {m['KGE']:>7} {m['RMSE']:>7} "
            f"{m['MAE']:>7} {m['PBIAS_%']:>7}% {m['Peak_Bias_%']:>7}% "
            f"{m['LogNSE']:>8}"
        )
    except Exception as e:
        print(f"  {name:<18} — ERROR: {e}")

print(f"{'='*75}")

# ── Save ───────────────────────────────────────────────────────
if rows:
    df = pd.DataFrame(rows).sort_values("NSE", ascending=False)
    out = MET_DIR / "all_models_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n  Saved → results/metrics/all_models_metrics.csv")
    print(f"\n  Best NSE : {df.iloc[0]['Model']} ({df.iloc[0]['NSE']})")
    print(f"  Best KGE : {df.loc[df.KGE.idxmax(),'Model']} ({df.KGE.max()})")
    print(f"  Best RMSE: {df.loc[df.RMSE.idxmin(),'Model']} ({df.RMSE.min()})")
