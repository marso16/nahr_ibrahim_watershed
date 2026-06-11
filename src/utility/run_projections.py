import os
import argparse
import inspect
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic model loader (same as sensitivity.py)
# ═══════════════════════════════════════════════════════════════════════════
def load_model_class(model_file: str, class_name: str):
    spec = importlib.util.spec_from_file_location("model_module", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, class_name):
        return getattr(module, class_name)

    import inspect

    classes = [
        name
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == "model_module"
    ]
    raise AttributeError(
        f"Class '{class_name}' not found in {model_file}.\n"
        f"Available classes: {classes}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained .pt checkpoint",
    )
    p.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="Path to .py file containing the model class",
    )
    p.add_argument(
        "--model_class",
        type=str,
        required=True,
        help="Class name (e.g. HybridGR4J_TCN, WatershedTCN, WatershedLSTM)",
    )
    p.add_argument(
        "--scaler_csv",
        type=str,
        required=True,
        help="Training scaler CSV (e.g. data/sequences/scaler_params_h1_lb60_hybrid.csv)",
    )
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--future_dir",
        type=str,
        default=str(ROOT / "data" / "master" / "future"),
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "data" / "projections"),
    )
    p.add_argument(
        "--gcm",
        type=str,
        default=None,
        help="Process only one GCM (default: all)",
    )
    p.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Process only one scenario (default: all)",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ──────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ckpt["config"]

    # ── Load scaler ─────────────────────────────────────────────────────
    print(f"Loading scaler: {args.scaler_csv}")
    scaler_df = pd.read_csv(args.scaler_csv, index_col=0)

    # Feature scalers (exclude __target__ and __meta__)
    feature_scalers = {}
    for idx, row in scaler_df.iterrows():
        if str(idx).startswith("__"):
            continue
        feature_scalers[idx] = (float(row["min"]), float(row["max"]))
    n_features = len(feature_scalers)
    print(f"  Features to normalize: {n_features}")

    # Target inverse-transform params
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

    print(
        f"  Target transform: min-max → "
        f"{'log' if log_transform else 'linear'} "
        f"(eps={log_eps})"
    )

    # ── Load model class dynamically ────────────────────────────────────
    model_class = load_model_class(args.model_file, args.model_class)
    sig = inspect.signature(model_class.__init__)
    model_keys = [p.name for p in sig.parameters.values() if p.name != "self"]
    model_kwargs = {k: v for k, v in cfg.items() if k in model_keys}

    # Inject data-derived args
    if "input_dim" in model_keys:
        if "input_dim" not in model_kwargs or model_kwargs["input_dim"] != n_features:
            model_kwargs["input_dim"] = n_features
            print(f"  Adjusting input_dim → {n_features}")

    # TCN fix: derive dilations from num_layers
    if "dilations" in model_keys and "dilations" not in model_kwargs:
        if "num_layers" in cfg:
            model_kwargs["dilations"] = [2**i for i in range(cfg["num_layers"])]
            print(f"  Derived dilations: {model_kwargs['dilations']}")

    # LSTM fix: derive units from hidden
    if "units_1" in model_keys and "units_1" not in model_kwargs:
        model_kwargs["units_1"] = cfg.get("units_1", cfg.get("hidden", 128))
    if "units_2" in model_keys and "units_2" not in model_kwargs:
        model_kwargs["units_2"] = cfg.get("units_2", cfg.get("hidden", 128))

    # GR4J fix: default indices
    if "precip_idx" in model_keys and "precip_idx" not in model_kwargs:
        model_kwargs["precip_idx"] = cfg.get("precip_idx", 0)
    if "pet_idx" in model_keys and "pet_idx" not in model_kwargs:
        model_kwargs["pet_idx"] = cfg.get("pet_idx", 27)

    print(f"  Model kwargs: {model_kwargs}")

    # ── Instantiate model ─────────────────────────────────────────────────
    model = model_class(**model_kwargs).to(device)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"  ERROR: Checkpoint incompatible with current model architecture.")
            print(f"  {e}")
            print(
                f"  SKIPPING. Delete or retrain this checkpoint if projections are needed."
            )
        raise
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params:,} parameters")

    # ── Find future CSVs ────────────────────────────────────────────────
    future_dir = Path(args.future_dir)
    pattern = (
        f"{'*' if args.gcm is None else args.gcm}__"
        f"{'*' if args.scenario is None else args.scenario}.csv"
    )
    files = sorted(future_dir.glob(pattern))
    if not files:
        print(f"No future files found in {future_dir} matching {pattern}")
        return

    print(f"\nFound {len(files)} future file(s) to process")

    # ── Process each future file ────────────────────────────────────────
    for f in files:
        print(f"\n{'='*60}")
        print(f"  {f.name}")
        print(f"{'='*60}")

        future = (
            pd.read_csv(f, parse_dates=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        n_future = len(future)
        print(
            f"  Future rows: {n_future}  ({future['date'].min().date()} → {future['date'].max().date()})"
        )

        # Extract feature columns in the exact training order
        feature_cols = list(feature_scalers.keys())

        # Check for missing columns
        missing = [c for c in feature_cols if c not in future.columns]
        if missing:
            # GR4J-TCN computes its own state columns internally
            if "GR4J" in args.model_class or "Hybrid" in args.model_class:
                gr4j_missing = [c for c in missing if c.startswith("gr4j_")]
                other_missing = [c for c in missing if not c.startswith("gr4j_")]
                if gr4j_missing:
                    print(
                        f"  Note: {len(gr4j_missing)} GR4J state columns missing (computed internally)."
                    )
                if other_missing:
                    print(f"  WARNING: Missing columns: {other_missing}")
            else:
                print(f"  WARNING: Missing columns: {missing}")
            for c in missing:
                future[c] = 0.0

        future_features = future[feature_cols].copy()

        # ── Padding strategy ─────────────────────────────────────────────
        # Prepend the first `lookback` rows to themselves so the first
        # prediction is for 2015-01-01. The first 60 days are used as
        # input context only; no leakage because discharge is not used.
        pad = future_features.iloc[: args.lookback].copy()
        concat_features = pd.concat([pad, future_features], ignore_index=True)
        concat_dates = pd.concat(
            [future["date"].iloc[: args.lookback], future["date"]],
            ignore_index=True,
        )
        n_concat = len(concat_features)
        print(f"  Concatenated with padding: {n_concat} rows")

        # ── Normalize ─────────────────────────────────────────────────────
        X = np.zeros((n_concat, n_features), dtype=np.float32)
        for i, col in enumerate(feature_cols):
            vmin, vmax = feature_scalers[col]
            vals = concat_features[col].values.astype(np.float64)
            if vmax - vmin > 1e-12:
                vals = (vals - vmin) / (vmax - vmin)
            else:
                vals = vals - vmin  # identity if no variance
            X[:, i] = vals

        if "GR4J" in args.model_class or "Hybrid" in args.model_class:
            p_idx = feature_cols.index("precip_mm_day")
            pet_idx = feature_cols.index("pet_mm_day")
            X[:, p_idx] = concat_features["precip_mm_day"].values.astype(np.float32)
            X[:, pet_idx] = concat_features["pet_mm_day"].values.astype(np.float32)

        # ── Build sliding windows ─────────────────────────────────────────
        LB, H = args.lookback, args.horizon
        windows = []
        target_dates = []

        # t is the index of the last day in the input window
        for t in range(LB - 1, n_concat - H):
            windows.append(X[t - LB + 1 : t + 1])
            target_dates.append(concat_dates.iloc[t + H])

        if len(windows) == 0:
            print(f"  No valid windows. Skipping.")
            continue

        X_arr = np.array(windows, dtype=np.float32)  # (N, LB, F)
        print(
            f"  Windows built: {len(X_arr)}  (predicting {pd.Timestamp(target_dates[0]).date()} → {pd.Timestamp(target_dates[-1]).date()})"
        )

        # ── Predict ──────────────────────────────────────────────────────
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

        y_pred = np.concatenate(preds).flatten()

        # ── Inverse transform ────────────────────────────────────────────
        y_pred_lin = y_pred * (y_max - y_min) + y_min
        if log_transform:
            y_pred_real = np.exp(y_pred_lin) - log_eps
        else:
            y_pred_real = y_pred_lin
        y_pred_real = np.maximum(y_pred_real, 0.0)

        # ── Save ─────────────────────────────────────────────────────────
        out_df = pd.DataFrame(
            {
                "date": pd.to_datetime(target_dates),
                "discharge_m3s_pred": y_pred_real,
            }
        )

        stem = f.stem  # e.g. "MPI-ESM1-2-HR__ssp245"
        out_file = out_dir / f"{stem}__h{H}.csv"
        out_df.to_csv(out_file, index=False)
        print(
            f"  Saved → {out_file.relative_to(ROOT)}  "
            f"({len(out_df)} rows, {out_df['date'].min().date()} → {out_df['date'].max().date()})"
        )

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
