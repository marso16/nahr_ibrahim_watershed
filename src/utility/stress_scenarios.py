import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

ROOT = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")
SCEN_DIR = ROOT / "results" / "scenarios"
FIG_DIR = ROOT / "results" / "figures"
MET_DIR = ROOT / "results" / "metrics"

# =============================================================================
# LOAD SCENARIO RESULTS
# =============================================================================
print("Loading scenario results ...")

# Use SSP5-8.5 as primary — widest spread between models
df = pd.read_csv(SCEN_DIR / "discharge_ssp585_daily.csv", parse_dates=["date"])
df["year"] = df["date"].dt.year

# Model columns
model_cols = [c for c in df.columns if c.startswith("Q_") and "ensemble" not in c]
print(f"  Models: {[c.replace('Q_', '') for c in model_cols]}")

# Annual means per model
annual = df.groupby("year")[model_cols].mean()

# =============================================================================
# DEFINE LOW / MEDIUM / HIGH SCENARIOS
# =============================================================================
# Based on the spread between individual model projections
# Low    = most optimistic model (highest future Q)
# Medium = ensemble mean
# High   = most pessimistic model (lowest future Q)

print("\nDefining stress scenarios ...")

# For each year, compute the spread across models
annual["Q_low"] = annual[model_cols].max(axis=1)  # best case — highest Q
annual["Q_medium"] = annual[model_cols].mean(axis=1)  # ensemble mean
annual["Q_high"] = annual[model_cols].min(axis=1)  # worst case — lowest Q

# Identify which model drives each extreme
best_model = annual[model_cols].mean().idxmax().replace("Q_", "")
worst_model = annual[model_cols].mean().idxmin().replace("Q_", "")

print(f"  Low scenario    (best case)  : {best_model}")
print(f"  Medium scenario (ensemble)   : All-model mean")
print(f"  High scenario   (worst case) : {worst_model}")

# Period statistics
for period, yr_min, yr_max in [
    ("2015–2040", 2015, 2040),
    ("2041–2070", 2041, 2070),
    ("2071–2100", 2071, 2100),
]:
    mask = (annual.index >= yr_min) & (annual.index <= yr_max)
    print(f"\n  Period {period}:")
    print(f"    Low    (best)  : {annual.loc[mask, 'Q_low'].mean():.4f} m³/s")
    print(f"    Medium (mean)  : {annual.loc[mask, 'Q_medium'].mean():.4f} m³/s")
    print(f"    High   (worst) : {annual.loc[mask, 'Q_high'].mean():.4f} m³/s")

# =============================================================================
# TREND ANALYSIS PER SCENARIO
# =============================================================================
print("\nTrend analysis ...")
rows = []
for label, col in [
    ("Low (best case)", "Q_low"),
    ("Medium (ensemble)", "Q_medium"),
    ("High (worst case)", "Q_high"),
]:
    y = annual[col].values
    years = annual.index.values
    slope, _, r, p, _ = stats.linregress(years, y)

    # Mann-Kendall
    n = len(y)
    s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    var_s = n * (n - 1) * (2 * n + 5) / 18
    z_mk = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0
    p_mk = 2 * (1 - stats.norm.cdf(abs(z_mk)))

    early = annual.loc[annual.index <= 2040, col].mean()
    late = annual.loc[annual.index >= 2075, col].mean()
    chg = (late - early) / early * 100

    rows.append(
        {
            "Scenario": label,
            "Slope (m³/s/dec)": round(slope * 10, 4),
            "R²": round(r**2, 4),
            "MK p-value": round(p_mk, 4),
            "Significant": "✓" if p_mk < 0.05 else "✗",
            "Mean 2015-2040": round(early, 4),
            "Mean 2075-2100": round(late, 4),
            "% Change": round(chg, 2),
        }
    )

trends = pd.DataFrame(rows)
trends.to_csv(MET_DIR / "stress_scenario_trends.csv", index=False)

print(f"\n  {'Scenario':<22} {'Slope/decade':>14} {'% Change':>10} {'Sig':>6}")
print(f"  {'-' * 55}")
for _, row in trends.iterrows():
    print(
        f"  {row['Scenario']:<22} {row['Slope (m³/s/dec)']:>14.4f} "
        f"{row['% Change']:>9.1f}% {row['Significant']:>6}"
    )

# =============================================================================
# SEASONAL ANALYSIS PER SCENARIO
# =============================================================================
df_daily = pd.read_csv(SCEN_DIR / "discharge_ssp585_daily.csv", parse_dates=["date"])
df_daily["year"] = df_daily["date"].dt.year
df_daily["month"] = df_daily["date"].dt.month

# Compute seasonal shift for each stress scenario
df_daily["Q_low"] = df_daily[model_cols].max(axis=1)
df_daily["Q_medium"] = df_daily[model_cols].mean(axis=1)
df_daily["Q_high"] = df_daily[model_cols].min(axis=1)

# =============================================================================
# VISUALISATION
# =============================================================================
print("\nGenerating figures ...")

COLORS = {
    "Low (best case)": "#00b4a0",  # teal — optimistic
    "Medium (ensemble)": "#3b9eff",  # blue — neutral
    "High (worst case)": "#e76f51",  # red  — pessimistic
}

fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor("#080f1a")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel 1: Annual discharge — all 3 scenarios ──
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor("#0d1825")

smooth = lambda s: s.rolling(10, center=True, min_periods=3).mean()

for label, col, color in [
    ("Low — best case (highest Q model)", "Q_low", "#00b4a0"),
    ("Medium — ensemble mean", "Q_medium", "#3b9eff"),
    ("High — worst case (lowest Q model)", "Q_high", "#e76f51"),
]:
    sm = smooth(annual[col])
    ax1.plot(annual.index, sm, color=color, linewidth=2.5, label=label)

# Uncertainty band between low and high
sm_low = smooth(annual["Q_low"])
sm_high = smooth(annual["Q_high"])
ax1.fill_between(
    annual.index, sm_high, sm_low, alpha=0.08, color="#3b9eff", label="Uncertainty band"
)

ax1.axvline(2025, color="#4a6a82", linewidth=1.2, linestyle=":", label="Present (2025)")
ax1.axhline(
    annual.loc[annual.index <= 2025, "Q_medium"].mean(),
    color="#8aafc4",
    linewidth=1,
    linestyle="--",
    alpha=0.5,
    label="2015–2025 baseline",
)

ax1.set_title(
    "Projected Discharge — Low / Medium / High Scenarios (SSP5-8.5)",
    color="#e8f4f8",
    fontsize=12,
)
ax1.set_ylabel("Annual Mean Discharge (m³/s)", color="#8aafc4")
ax1.set_xlabel("Year", color="#8aafc4")
ax1.tick_params(colors="#4a6a82")
ax1.spines[:].set_color("#1e3448")
ax1.legend(
    facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4", fontsize=8, ncol=3
)
ax1.grid(alpha=0.08)
ax1.set_facecolor("#0d1825")

# ── Panel 2: Seasonal shift — Low scenario ──
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor("#0d1825")
month_names = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
for period, yr_min, yr_max, color in [
    ("2015–2040", 2015, 2040, "#3b9eff"),
    ("2041–2070", 2041, 2070, "#f4a261"),
    ("2071–2100", 2071, 2100, "#00b4a0"),
]:
    mask = (df_daily.year >= yr_min) & (df_daily.year <= yr_max)
    monthly = df_daily[mask].groupby("month")["Q_low"].mean()
    ax2.plot(
        monthly.index,
        monthly.values,
        color=color,
        linewidth=2,
        marker="o",
        markersize=5,
        label=period,
    )
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(month_names, color="#8aafc4", fontsize=8)
ax2.set_title("Seasonal Shift — Low Scenario (Best Case)", color="#e8f4f8", fontsize=10)
ax2.set_ylabel("Discharge (m³/s)", color="#8aafc4")
ax2.tick_params(colors="#4a6a82")
ax2.spines[:].set_color("#1e3448")
ax2.legend(facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4", fontsize=8)
ax2.grid(alpha=0.08)
ax2.set_facecolor("#0d1825")

# ── Panel 3: Seasonal shift — High scenario ──
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor("#0d1825")
for period, yr_min, yr_max, color in [
    ("2015–2040", 2015, 2040, "#3b9eff"),
    ("2041–2070", 2041, 2070, "#f4a261"),
    ("2071–2100", 2071, 2100, "#e76f51"),
]:
    mask = (df_daily.year >= yr_min) & (df_daily.year <= yr_max)
    monthly = df_daily[mask].groupby("month")["Q_high"].mean()
    ax3.plot(
        monthly.index,
        monthly.values,
        color=color,
        linewidth=2,
        marker="o",
        markersize=5,
        label=period,
    )
ax3.set_xticks(range(1, 13))
ax3.set_xticklabels(month_names, color="#8aafc4", fontsize=8)
ax3.set_title(
    "Seasonal Shift — High Scenario (Worst Case)", color="#e8f4f8", fontsize=10
)
ax3.set_ylabel("Discharge (m³/s)", color="#8aafc4")
ax3.tick_params(colors="#4a6a82")
ax3.spines[:].set_color("#1e3448")
ax3.legend(facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4", fontsize=8)
ax3.grid(alpha=0.08)
ax3.set_facecolor("#0d1825")

# ── Panel 4: % Change bar chart by period ──
ax4 = fig.add_subplot(gs[2, :])
ax4.set_facecolor("#0d1825")

periods = ["2015–2040", "2041–2070", "2071–2100"]
baseline = annual.loc[annual.index <= 2040]

scenario_data = {}
for label, col in [
    ("Low (best)", "Q_low"),
    ("Medium", "Q_medium"),
    ("High (worst)", "Q_high"),
]:
    base = annual.loc[annual.index <= 2040, col].mean()
    vals = []
    for yr_min, yr_max in [(2015, 2040), (2041, 2070), (2071, 2100)]:
        m = annual.loc[(annual.index >= yr_min) & (annual.index <= yr_max), col].mean()
        vals.append((m - base) / base * 100)
    scenario_data[label] = vals

x = np.arange(len(periods))
w = 0.25
colors_bar = ["#00b4a0", "#3b9eff", "#e76f51"]

for i, (label, vals) in enumerate(scenario_data.items()):
    bars = ax4.bar(x + i * w, vals, w, label=label, color=colors_bar[i], alpha=0.85)
    for bar, val in zip(bars, vals):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3 * np.sign(val + 1e-9),
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            color="#e8f4f8",
            fontsize=8,
            fontfamily="monospace",
        )

ax4.axhline(0, color="#8aafc4", linewidth=1)
ax4.set_xticks(x + w)
ax4.set_xticklabels(periods, color="#8aafc4")
ax4.set_ylabel("% Change from Baseline (2015–2040)", color="#8aafc4")
ax4.set_title(
    "Discharge Change by Period and Scenario — SSP5-8.5", color="#e8f4f8", fontsize=12
)
ax4.tick_params(colors="#4a6a82")
ax4.spines[:].set_color("#1e3448")
ax4.legend(facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4")
ax4.grid(axis="y", alpha=0.08)
ax4.set_facecolor("#0d1825")

fig.suptitle(
    "Nahr Ibrahim — Low / Medium / High Climate Stress Scenarios\n"
    "SSP5-8.5 · MPI-ESM1-2-HR · Model Ensemble Spread",
    color="#e8f4f8",
    fontsize=13,
    y=1.01,
    fontfamily="monospace",
)

plt.savefig(
    FIG_DIR / "stress_scenarios.png", dpi=150, bbox_inches="tight", facecolor="#080f1a"
)
plt.show()
