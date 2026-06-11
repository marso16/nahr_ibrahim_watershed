import os
import argparse
import inspect
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic model loader (same as run_projections.py)
# ═══════════════════════════════════════════════════════════════════════════
def load_model_class(model_file: str, class_name: str):
    spec = importlib.util.spec_from_file_location("model_module", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, class_name):
        return getattr(module, class_name)
    classes = [
        name
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == "model_module"
    ]
    raise AttributeError(
        f"Class '{class_name}' not found in {model_file}.\nAvailable: {classes}"
    )


# ═══════════════════════════════════════════════════════════════════════════
def compute_nse(obs, sim):
    """Nash–Sutcliffe Efficiency."""
    obs = np.asarray(obs).flatten()
    sim = np.asarray(sim).flatten()
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs, sim = obs[mask], sim[mask]
    if len(obs) == 0:
        return np.nan
    mean_obs = np.mean(obs)
    if abs(mean_obs) < 1e-12:
        return np.nan
    return 1.0 - float(np.sum((sim - obs) ** 2) / np.sum((obs - mean_obs) ** 2))


# ═══════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model_file", type=str, required=True)
    p.add_argument("--model_class", type=str, required=True)
    p.add_argument(
        "--scaler_csv",
        type=str,
        required=True,
        help="Scaler with __target__ and __meta__ rows for inverse transform",
    )
    p.add_argument(
        "--seq_dir",
        type=str,
        default=str(ROOT / "data" / "sequences"),
    )
    p.add_argument(
        "--seq_suffix",
        type=str,
        default="",
        help='e.g. "_hybrid" for GR4J-TCN, "" for plain TCN/LSTM',
    )
    p.add_argument("--lookback", type=int, required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_permutations", type=int, default=10)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--out_csv", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    H, LB = args.horizon, args.lookback
    suf = f"_h{H}_lb{LB}{args.seq_suffix}"

    # ── Load test sequences ───────────────────────────────────────────────
    seq_dir = Path(args.seq_dir)
    X_test = np.load(seq_dir / f"X_test{suf}.npy", allow_pickle=True).astype(np.float32)
    y_test = np.load(seq_dir / f"y_test{suf}.npy", allow_pickle=True).astype(np.float32)
    print(f"Loaded X_test{suf}: {X_test.shape}, y_test: {y_test.shape}")

    n_samples, lookback, n_features = X_test.shape
    assert lookback == LB, f"Lookback mismatch: {lookback} vs {LB}"

    # ── Load scaler (only for inverse transform + feature names) ──────────
    scaler_df = pd.read_csv(args.scaler_csv, index_col=0)

    if "__target__" in scaler_df.index:
        y_min = float(scaler_df.loc["__target__", "min"])
        y_max = float(scaler_df.loc["__target__", "max"])
    else:
        y_min, y_max = 0.0, 1.0

    log_transform = False
    log_eps = 0.0
    if "__meta__" in scaler_df.index:
        log_transform = bool(float(scaler_df.loc["__meta__", "min"]))
        log_eps = float(scaler_df.loc["__meta__", "max"])

    # Feature names from scaler (exclude __target__ and __meta__)
    feature_cols = [c for c in scaler_df.index if not str(c).startswith("__")]
    if len(feature_cols) != n_features:
        print(
            f"WARNING: scaler has {len(feature_cols)} features, "
            f"X_test has {n_features}. Using generic names."
        )
        feature_cols = [f"feat_{i}" for i in range(n_features)]

    # ── Load model dynamically ────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]
    model_class = load_model_class(args.model_file, args.model_class)
    sig = inspect.signature(model_class.__init__)
    model_keys = [p.name for p in sig.parameters.values() if p.name != "self"]
    model_kwargs = {k: v for k, v in cfg.items() if k in model_keys}

    if "input_dim" in model_keys:
        model_kwargs["input_dim"] = n_features
    if "dilations" in model_keys and "dilations" not in model_kwargs:
        if "num_layers" in cfg:
            model_kwargs["dilations"] = [2**i for i in range(cfg["num_layers"])]
    if "units_1" in model_keys and "units_1" not in model_kwargs:
        model_kwargs["units_1"] = cfg.get("units_1", cfg.get("hidden", 128))
    if "units_2" in model_keys and "units_2" not in model_kwargs:
        model_kwargs["units_2"] = cfg.get("units_2", cfg.get("hidden", 128))
    if "precip_idx" in model_keys and "precip_idx" not in model_kwargs:
        model_kwargs["precip_idx"] = cfg.get("precip_idx", 0)
    if "pet_idx" in model_keys and "pet_idx" not in model_kwargs:
        model_kwargs["pet_idx"] = cfg.get("pet_idx", 27)

    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")

    # ── Prediction helper (inverse log + min-max) ─────────────────────────
    def predict(X_arr):
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_arr), args.batch_size):
                xb = torch.tensor(
                    X_arr[i : i + args.batch_size], dtype=torch.float32, device=device
                )
                out = model(xb)
                if out.dim() > 1:
                    out = out.squeeze(-1)
                preds.append(out.cpu().numpy())
        y = np.concatenate(preds).flatten()
        # Inverse min-max
        y_lin = y * (y_max - y_min) + y_min
        # Inverse log if used
        if log_transform:
            y_real = np.exp(y_lin) - log_eps
        else:
            y_real = y_lin
        return np.maximum(y_real, 0.0)

    # ── Inverse-transform observed test target ────────────────────────────
    y_test_real = y_test.flatten() * (y_max - y_min) + y_min
    if log_transform:
        y_test_real = np.exp(y_test_real) - log_eps
    y_test_real = np.maximum(y_test_real, 0.0)

    # ── Baseline NSE ──────────────────────────────────────────────────────
    y_pred_base = predict(X_test)
    baseline_nse = compute_nse(y_test_real, y_pred_base)
    print(f"\nBaseline test NSE: {baseline_nse:.4f}")

    # ── Permutation importance ────────────────────────────────────────────
    print(
        f"\nRunning {args.n_permutations} permutations for each of {n_features} features..."
    )
    rows = []
    rng = np.random.default_rng(42)

    for f in range(n_features):
        nse_drops = []
        for _ in range(args.n_permutations):
            X_perm = X_test.copy()
            # Flatten feature f across all samples and timesteps, shuffle, reshape
            vals = X_perm[:, :, f].flatten()
            rng.shuffle(vals)
            X_perm[:, :, f] = vals.reshape(n_samples, lookback)

            y_pred_perm = predict(X_perm)
            nse_perm = compute_nse(y_test_real, y_pred_perm)
            nse_drops.append(baseline_nse - nse_perm)

        mean_drop = float(np.mean(nse_drops))
        std_drop = float(np.std(nse_drops))
        rows.append(
            {
                "feature": feature_cols[f],
                "baseline_nse": round(baseline_nse, 4),
                "mean_nse_drop": round(mean_drop, 5),
                "std_nse_drop": round(std_drop, 5),
                "perm_nse": round(baseline_nse - mean_drop, 4),
            }
        )
        print(f"  {feature_cols[f]:<<22} drop: {mean_drop:.5f} ± {std_drop:.5f}")

    # ── Save ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows).sort_values("mean_nse_drop", ascending=False)
    if args.out_csv:
        out_path = Path(args.out_csv)
    else:
        out_path = (
            ROOT
            / "results"
            / "sensitivity"
            / f"sensitivity_{args.model_class}_h{H}_lb{LB}{args.seq_suffix}.csv"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
