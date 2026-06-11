import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
PROJ_DIR = ROOT / "data" / "projections"
HIST_FILE = ROOT / "data" / "master" / "nahr_ibrahim_master_model.csv"
OUT_FIG_DIR = ROOT / "results" / "projections" / "figures"
OUT_TAB_DIR = ROOT / "results" / "projections" / "tables"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_TAB_DIR.mkdir(parents=True, exist_ok=True)

# ─── Configuration ──────────────────────────────────────────────────────────
GCMS = [
    "MPI-ESM1-2-HR",
    "EC-Earth3",
    "ACCESS-CM2",
    "NorESM2-MM",
    "MRI-ESM2-0",
    "CMCC-ESM2",
    "INM-CM5-0",
]
SCENARIOS = ["ssp245", "ssp585"]
HORIZONS = [1, 3, 14]

# IPCC AR6-aligned periods
PERIODS = {
    "Historical (1995-2014)": ("1995-01-01", "2014-12-31"),
    "Mid-century (2041-2060)": ("2041-01-01", "2060-12-31"),
    "End-century (2081-2100)": ("2081-01-01", "2100-12-31"),
}

# Color scheme
SCEN_COLOR = {"ssp245": "#2C7FB8", "ssp585": "#D94701"}
HIST_COLOR = "#525252"


# ═══════════════════════════════════════════════════════════════════════════
# Load all projections + historical baseline
# ═══════════════════════════════════════════════════════════════════════════
print("Loading historical baseline...")
hist = pd.read_csv(HIST_FILE, parse_dates=["date"])[["date", "discharge_m3s"]]
hist = hist.sort_values("date").reset_index(drop=True)
hist_baseline = hist[
    (hist["date"] >= PERIODS["Historical (1995-2014)"][0])
    & (hist["date"] <= PERIODS["Historical (1995-2014)"][1])
].copy()
hist_actual_start = hist_baseline["date"].min().date()
print(
    f"  Historical: {len(hist_baseline)} days "
    f"({hist_actual_start} → {hist_baseline['date'].max().date()})"
)
print(
    f"  Historical mean discharge: {hist_baseline['discharge_m3s'].mean():.3f} m³/s\n"
)


def load_projection(gcm, scenario, horizon):
    p = PROJ_DIR / f"{gcm}__{scenario}__h{horizon}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


# Pre-load all projections
print("Loading all projections...")
all_projections = {}  # key: (gcm, scenario, horizon) → DataFrame
for gcm in GCMS:
    for scen in SCENARIOS:
        for h in HORIZONS:
            df = load_projection(gcm, scen, h)
            if df is not None:
                all_projections[(gcm, scen, h)] = df
print(f"  Loaded {len(all_projections)} projection files\n")


# ═══════════════════════════════════════════════════════════════════════════
# 1. CHANGE TABLES per horizon
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  1. Change tables")
print("=" * 70)


def period_mean(df, period_key):
    start, end = PERIODS[period_key]
    sub = df[(df["date"] >= start) & (df["date"] <= end)]
    if len(sub) == 0:
        return np.nan
    col = (
        "discharge_m3s_pred" if "discharge_m3s_pred" in sub.columns else "discharge_m3s"
    )
    return sub[col].mean()


def period_quantile(df, period_key, q):
    start, end = PERIODS[period_key]
    sub = df[(df["date"] >= start) & (df["date"] <= end)]
    if len(sub) == 0:
        return np.nan
    col = (
        "discharge_m3s_pred" if "discharge_m3s_pred" in sub.columns else "discharge_m3s"
    )
    return np.quantile(sub[col].values, q)


hist_mean = hist_baseline["discharge_m3s"].mean()
hist_p95 = np.quantile(hist_baseline["discharge_m3s"].values, 0.95)
hist_p05 = np.quantile(hist_baseline["discharge_m3s"].values, 0.05)

for horizon in HORIZONS:
    rows = []
    for gcm in GCMS:
        for scen in SCENARIOS:
            key = (gcm, scen, horizon)
            if key not in all_projections:
                continue
            df = all_projections[key]
            mid_mean = period_mean(df, "Mid-century (2041-2060)")
            end_mean = period_mean(df, "End-century (2081-2100)")
            mid_p95 = period_quantile(df, "Mid-century (2041-2060)", 0.95)
            end_p95 = period_quantile(df, "End-century (2081-2100)", 0.95)
            rows.append(
                {
                    "gcm": gcm,
                    "scenario": scen,
                    "hist_mean_m3s": round(hist_mean, 3),
                    "mid_mean_m3s": round(mid_mean, 3),
                    "end_mean_m3s": round(end_mean, 3),
                    "mid_change_pct": round(
                        100 * (mid_mean - hist_mean) / hist_mean, 2
                    ),
                    "end_change_pct": round(
                        100 * (end_mean - hist_mean) / hist_mean, 2
                    ),
                    "hist_p95": round(hist_p95, 2),
                    "mid_p95": round(mid_p95, 2),
                    "end_p95": round(end_p95, 2),
                    "mid_p95_change_pct": round(
                        100 * (mid_p95 - hist_p95) / hist_p95, 2
                    ),
                    "end_p95_change_pct": round(
                        100 * (end_p95 - hist_p95) / hist_p95, 2
                    ),
                }
            )
    table = pd.DataFrame(rows)
    out_path = OUT_TAB_DIR / f"change_table_h{horizon}.csv"
    table.to_csv(out_path, index=False)
    print(f"\n  h={horizon} → {out_path.relative_to(ROOT)}")

    if table.empty:
        print(f"    No projections loaded for h={horizon}, skipping ensemble stats.")
        continue

    # Ensemble medians (across GCMs) per scenario
    ens = (
        table.groupby("scenario")
        .agg(
            mid_change_median=("mid_change_pct", "median"),
            end_change_median=("end_change_pct", "median"),
            mid_change_min=("mid_change_pct", "min"),
            mid_change_max=("mid_change_pct", "max"),
            end_change_min=("end_change_pct", "min"),
            end_change_max=("end_change_pct", "max"),
        )
        .round(2)
    )
    print(f"    Ensemble (7 GCMs):")
    print(ens.to_string())


# ═══════════════════════════════════════════════════════════════════════════
# 2. ENSEMBLE TIME-SERIES PLOT (using h=1, the most skillful)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  2. Ensemble time-series plot")
print("=" * 70)

PLOT_HORIZON = 1  # most skillful


def build_ensemble_yearly(scenario, horizon=PLOT_HORIZON):
    """Build a dataframe of yearly mean discharge per GCM."""
    yearly = {}
    for gcm in GCMS:
        df = all_projections.get((gcm, scenario, horizon))
        if df is None:
            continue
        d = df.copy()
        d["year"] = d["date"].dt.year
        agg = d.groupby("year")["discharge_m3s_pred"].mean()
        yearly[gcm] = agg
    return pd.DataFrame(yearly)


# Guard: skip if no data for the plot horizon
has_plot_data = any(
    (gcm, scen, PLOT_HORIZON) in all_projections for gcm in GCMS for scen in SCENARIOS
)

if not has_plot_data:
    print(f"  SKIPPING time-series plot: no projections for h={PLOT_HORIZON}")
else:
    fig, ax = plt.subplots(figsize=(14, 5.5))

    # Historical baseline (annual means)
    hist_yearly = hist.copy()
    hist_yearly["year"] = hist_yearly["date"].dt.year
    hist_annual = hist_yearly.groupby("year")["discharge_m3s"].mean()
    ax.plot(
        hist_annual.index,
        hist_annual.values,
        color=HIST_COLOR,
        lw=1.6,
        label="Historical (GloFAS, 2000-2025)",
    )

    # Two scenarios with ensemble spread
    for scen in SCENARIOS:
        ens = build_ensemble_yearly(scen)
        if ens.empty:
            continue
        median = ens.median(axis=1)
        q25 = ens.quantile(0.25, axis=1)
        q75 = ens.quantile(0.75, axis=1)
        q05 = ens.quantile(0.05, axis=1)
        q95 = ens.quantile(0.95, axis=1)

        # 5-95% spread (light) and 25-75% spread (darker)
        ax.fill_between(ens.index, q05, q95, color=SCEN_COLOR[scen], alpha=0.10)
        ax.fill_between(ens.index, q25, q75, color=SCEN_COLOR[scen], alpha=0.25)
        ax.plot(
            ens.index,
            median.values,
            color=SCEN_COLOR[scen],
            lw=2.0,
            label=f"{scen.upper()} median (7 GCMs)",
        )

    # Period markers
    for period_key, (start, end) in PERIODS.items():
        if "Historical" in period_key:
            continue
        yr_start = pd.Timestamp(start).year
        yr_end = pd.Timestamp(end).year
        ax.axvspan(yr_start, yr_end, color="gold", alpha=0.10, zorder=-1)
        ax.text(
            (yr_start + yr_end) / 2,
            ax.get_ylim()[1] * 0.97,
            period_key.split(" (")[0],
            ha="center",
            fontsize=9,
            style="italic",
            color="#666",
        )

    ax.set_title(
        f"Projected annual mean discharge at Nahr Ibrahim outlet — h=1 forecast\n"
        f"Multi-GCM ensemble (7 GCMs), 2015-2100",
        fontsize=12,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean annual discharge (m³/s)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(1995, 2100)

    plt.tight_layout()
    out_path = OUT_FIG_DIR / "ensemble_timeseries.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. SEASONAL CYCLE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  3. Seasonal cycle comparison")
print("=" * 70)


def monthly_climatology(df, period_key, value_col):
    start, end = PERIODS[period_key]
    sub = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    sub["month"] = sub["date"].dt.month
    return sub.groupby("month")[value_col].mean()


# Guard: skip if no data for the plot horizon
has_seasonal_data = any(
    (gcm, scen, PLOT_HORIZON) in all_projections for gcm in GCMS for scen in SCENARIOS
)

if not has_seasonal_data:
    print(f"  SKIPPING seasonal cycle plot: no projections for h={PLOT_HORIZON}")
else:
    # Historical
    hist_monthly = monthly_climatology(
        hist.assign(discharge=hist["discharge_m3s"]).rename(
            columns={"discharge_m3s": "discharge"}
        ),
        "Historical (1995-2014)",
        "discharge",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, period in zip(axes, ["Mid-century (2041-2060)", "End-century (2081-2100)"]):
        ax.plot(
            hist_monthly.index,
            hist_monthly.values,
            color=HIST_COLOR,
            lw=2.5,
            marker="o",
            label="Historical (1995-2014)",
        )

        for scen in SCENARIOS:
            ens_monthly = []
            for gcm in GCMS:
                df = all_projections.get((gcm, scen, PLOT_HORIZON))
                if df is None:
                    continue
                m = monthly_climatology(df, period, "discharge_m3s_pred")
                ens_monthly.append(m)
            if not ens_monthly:
                continue
            ensdf = pd.concat(ens_monthly, axis=1)
            ax.plot(
                ensdf.index,
                ensdf.median(axis=1),
                color=SCEN_COLOR[scen],
                lw=2.0,
                marker="s",
                label=f"{scen.upper()} median",
            )
            ax.fill_between(
                ensdf.index,
                ensdf.quantile(0.25, axis=1),
                ensdf.quantile(0.75, axis=1),
                color=SCEN_COLOR[scen],
                alpha=0.20,
            )

        ax.set_title(period)
        ax.set_xlabel("Month")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    axes[0].set_ylabel("Mean monthly discharge (m³/s)")
    plt.suptitle("Seasonal cycle changes — h=1 forecast", fontsize=12)
    plt.tight_layout()
    out_path = OUT_FIG_DIR / "seasonal_cycle.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. FLOOD AND DRY DAY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  4. Extreme event statistics")
print("=" * 70)

hist_p95_threshold = np.quantile(hist_baseline["discharge_m3s"].values, 0.95)
hist_p05_threshold = np.quantile(hist_baseline["discharge_m3s"].values, 0.05)
print(f"  Flood threshold (historical P95): {hist_p95_threshold:.2f} m³/s")
print(f"  Dry threshold (historical P05):   {hist_p05_threshold:.2f} m³/s")

rows = []
for gcm in GCMS:
    for scen in SCENARIOS:
        df = all_projections.get((gcm, scen, PLOT_HORIZON))
        if df is None:
            continue
        for period_key in ["Mid-century (2041-2060)", "End-century (2081-2100)"]:
            start, end = PERIODS[period_key]
            sub = df[(df["date"] >= start) & (df["date"] <= end)]
            n_total = len(sub)
            if n_total == 0:
                continue
            flood_pct = (
                100 * (sub["discharge_m3s_pred"] >= hist_p95_threshold).sum() / n_total
            )
            dry_pct = (
                100 * (sub["discharge_m3s_pred"] <= hist_p05_threshold).sum() / n_total
            )
            rows.append(
                {
                    "gcm": gcm,
                    "scenario": scen,
                    "period": period_key,
                    "flood_days_pct": round(flood_pct, 2),
                    "dry_days_pct": round(dry_pct, 2),
                }
            )

extremes_df = pd.DataFrame(rows)
if extremes_df.empty:
    print(f"  No extreme-event data for h={PLOT_HORIZON}, skipping table.")
else:
    extremes_df.to_csv(OUT_TAB_DIR / "extreme_event_stats.csv", index=False)
    ext_summary = (
        extremes_df.groupby(["scenario", "period"])
        .agg(
            flood_median=("flood_days_pct", "median"),
            flood_min=("flood_days_pct", "min"),
            flood_max=("flood_days_pct", "max"),
            dry_median=("dry_days_pct", "median"),
            dry_min=("dry_days_pct", "min"),
            dry_max=("dry_days_pct", "max"),
        )
        .round(2)
    )
    print("\n  Ensemble extreme-event frequencies:")
    print(ext_summary.to_string())


# ═══════════════════════════════════════════════════════════════════════════
# 5. FLOW DURATION CURVES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  5. Flow duration curves")
print("=" * 70)


def fdc(values):
    """Return (exceedance probability, sorted discharge) for FDC plotting."""
    sorted_q = np.sort(values)[::-1]
    n = len(sorted_q)
    exceed = (np.arange(1, n + 1)) / (n + 1) * 100
    return exceed, sorted_q


# Guard: skip if no data for the plot horizon
has_fdc_data = any(
    (gcm, scen, PLOT_HORIZON) in all_projections for gcm in GCMS for scen in SCENARIOS
)

if not has_fdc_data:
    print(f"  SKIPPING FDC plot: no projections for h={PLOT_HORIZON}")
else:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, period in zip(axes, ["Mid-century (2041-2060)", "End-century (2081-2100)"]):
        # Historical FDC
        e_h, q_h = fdc(hist_baseline["discharge_m3s"].values)
        ax.plot(e_h, q_h, color=HIST_COLOR, lw=2.0, label="Historical (1995-2014)")

        # Per-scenario ensemble FDC (using ensemble of all GCM days)
        for scen in SCENARIOS:
            all_days = []
            for gcm in GCMS:
                df = all_projections.get((gcm, scen, PLOT_HORIZON))
                if df is None:
                    continue
                start, end = PERIODS[period]
                sub = df[(df["date"] >= start) & (df["date"] <= end)]
                all_days.append(sub["discharge_m3s_pred"].values)
            if not all_days:
                continue
            # Per-GCM FDC, then ensemble median curve
            ens_q_at_pct = []
            eval_pcts = np.logspace(np.log10(0.5), np.log10(99.5), 50)
            for arr in all_days:
                e, q = fdc(arr)
                ens_q_at_pct.append(np.interp(eval_pcts, e, q))
            ens_arr = np.array(ens_q_at_pct)
            ax.plot(
                eval_pcts,
                np.median(ens_arr, axis=0),
                color=SCEN_COLOR[scen],
                lw=2.0,
                label=f"{scen.upper()} median",
            )
            ax.fill_between(
                eval_pcts,
                np.quantile(ens_arr, 0.25, axis=0),
                np.quantile(ens_arr, 0.75, axis=0),
                color=SCEN_COLOR[scen],
                alpha=0.20,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Exceedance probability (%)")
        ax.set_ylabel("Discharge (m³/s)")
        ax.set_title(period)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(alpha=0.3, which="both")

    plt.suptitle("Flow duration curves — h=1 forecast", fontsize=12)
    plt.tight_layout()
    out_path = OUT_FIG_DIR / "flow_duration_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. HEADLINE SUMMARY CSV
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  6. Headline summary")
print("=" * 70)

# For each scenario × period, compute ensemble median change vs historical
headline_rows = []
for scen in SCENARIOS:
    for period_key in ["Mid-century (2041-2060)", "End-century (2081-2100)"]:
        gcm_means = []
        gcm_p95s = []
        for gcm in GCMS:
            df = all_projections.get((gcm, scen, PLOT_HORIZON))
            if df is None:
                continue
            gcm_means.append(period_mean(df, period_key))
            gcm_p95s.append(period_quantile(df, period_key, 0.95))
        if not gcm_means:
            continue
        means = np.array(gcm_means)
        p95s = np.array(gcm_p95s)
        headline_rows.append(
            {
                "scenario": scen,
                "period": period_key,
                "hist_mean_m3s": round(hist_mean, 3),
                "future_mean_median": round(float(np.median(means)), 3),
                "future_mean_min": round(float(np.min(means)), 3),
                "future_mean_max": round(float(np.max(means)), 3),
                "mean_change_pct_median": round(
                    100 * (np.median(means) - hist_mean) / hist_mean, 2
                ),
                "mean_change_pct_min": round(
                    100 * (np.min(means) - hist_mean) / hist_mean, 2
                ),
                "mean_change_pct_max": round(
                    100 * (np.max(means) - hist_mean) / hist_mean, 2
                ),
                "p95_change_pct_median": round(
                    100 * (np.median(p95s) - hist_p95) / hist_p95, 2
                ),
                "n_gcms": len(means),
            }
        )

headline = pd.DataFrame(headline_rows)
if headline.empty:
    print(f"  No headline data for h={PLOT_HORIZON}, skipping.")
else:
    headline.to_csv(OUT_TAB_DIR / "headline_summary.csv", index=False)
    print(headline.to_string(index=False))

# ─── Done ────────────────────────────────────────────────────────────────────
print(f"\n{'═' * 70}")
print(f"  Analysis complete.")
print(f"  Tables  → {OUT_TAB_DIR.relative_to(ROOT)}")
print(f"  Figures → {OUT_FIG_DIR.relative_to(ROOT)}")
print(f"{'═' * 70}")
