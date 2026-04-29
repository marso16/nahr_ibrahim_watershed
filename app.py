"""
=============================================================================
Nahr Ibrahim Watershed — Thesis Dashboard
Streamlit Web Application
=============================================================================
Run with:  streamlit run app.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Nahr Ibrahim · AI Hydrology",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS — Dark hydrological theme
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg-primary:    #080f1a;
    --bg-secondary:  #0d1825;
    --bg-card:       #111e2e;
    --bg-card-hover: #162535;
    --accent-blue:   #3b9eff;
    --accent-cyan:   #00d4ff;
    --accent-teal:   #00b4a0;
    --accent-snow:   #a8d8ea;
    --accent-warm:   #f4a261;
    --accent-red:    #e76f51;
    --text-primary:  #e8f4f8;
    --text-secondary:#8aafc4;
    --text-muted:    #4a6a82;
    --border:        #1e3448;
    --border-bright: #2a4d6e;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: linear-gradient(135deg, #060d16 0%, #080f1a 50%, #0a1520 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #0d2137 0%, #0a1a2e 40%, #061020 100%);
    border: 1px solid var(--border-bright);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(59,158,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0,212,255,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    line-height: 1.1;
    color: var(--text-primary);
    margin: 0 0 0.5rem 0;
}
.hero-title span {
    color: var(--accent-cyan);
}
.hero-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.hero-tags {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
}
.hero-tag {
    background: rgba(59,158,255,0.1);
    border: 1px solid rgba(59,158,255,0.25);
    color: var(--accent-blue);
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
    opacity: 0.6;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.6rem;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--accent-cyan);
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-unit {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-secondary);
}

/* Section headers */
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent-blue);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin: 2.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Dataset cards */
.dataset-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.dataset-icon {
    font-size: 1.6rem;
    min-width: 40px;
    text-align: center;
}
.dataset-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.2rem;
}
.dataset-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
}
.badge {
    margin-left: auto;
    background: rgba(0,180,160,0.15);
    border: 1px solid rgba(0,180,160,0.3);
    color: var(--accent-teal);
    padding: 0.25rem 0.7rem;
    border-radius: 50px;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    white-space: nowrap;
}

/* Split badges */
.split-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}
.split-card {
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.split-card.train  { background: rgba(59,158,255,0.08);  border: 1px solid rgba(59,158,255,0.2); }
.split-card.val    { background: rgba(244,162,97,0.08);  border: 1px solid rgba(244,162,97,0.2); }
.split-card.test   { background: rgba(231,111,81,0.08);  border: 1px solid rgba(231,111,81,0.2); }
.split-label { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; }
.split-period { font-family: 'DM Serif Display', serif; font-size: 1.1rem; margin: 0.3rem 0; }
.split-card.train .split-period { color: var(--accent-blue); }
.split-card.val   .split-period { color: var(--accent-warm); }
.split-card.test  .split-period { color: var(--accent-red); }
.split-days { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: var(--text-secondary); }

/* Pipeline steps */
.pipeline-step {
    display: flex;
    align-items: flex-start;
    gap: 1.2rem;
    padding: 1rem 0;
    border-bottom: 1px solid var(--border);
}
.step-number {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    color: #000;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    flex-shrink: 0;
    margin-top: 2px;
}
.step-title { font-size: 0.95rem; font-weight: 600; color: var(--text-primary); }
.step-desc  { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--text-muted); margin-top: 0.3rem; }

/* Info box */
.info-box {
    background: rgba(59,158,255,0.06);
    border: 1px solid rgba(59,158,255,0.2);
    border-left: 3px solid var(--accent-blue);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin: 1rem 0;
    font-family: 'DM Mono', monospace;
}

/* Plotly chart container */
.chart-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 1.5rem;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PATHS
# =============================================================================

ROOT = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")

@st.cache_data
def load_master():
    p = ROOT / "data" / "master" / "nahr_ibrahim_master_full.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["date"])
    return None

@st.cache_data
def load_splits():
    splits = {}
    for name in ["train", "val", "test"]:
        p = ROOT / "data" / "splits" / f"{name}_raw.csv"
        if p.exists():
            splits[name] = pd.read_csv(p, parse_dates=["date"])
    return splits

@st.cache_data
def load_scaler():
    p = ROOT / "data" / "splits" / "scaler_params.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0)
    return None

df       = load_master()
splits   = load_splits()
scaler   = load_scaler()

# Plotly theme
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Syne, sans-serif", color="#8aafc4"),
    xaxis=dict(gridcolor="#1e3448", showline=False, zeroline=False),
    yaxis=dict(gridcolor="#1e3448", showline=False, zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e3448"),
)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 2rem 0;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.4rem; color: #e8f4f8;'>Nahr Ibrahim</div>
        <div style='font-family: DM Mono, monospace; font-size: 0.65rem; color: #4a6a82; letter-spacing: 0.15em; text-transform: uppercase; margin-top: 0.3rem;'>Watershed · AI Hydrology</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🌊 Overview", "📡 Dataset", "📈 Exploratory Analysis",
         "✂️ Data Splits", "🔬 Variable Inspector", "📋 Pipeline Summary"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-family: DM Mono, monospace; font-size: 0.68rem; color: #4a6a82; line-height: 1.8;'>
    <div style='color: #8aafc4; margin-bottom: 0.5rem;'>THESIS INFO</div>
    USJ . ESIB<br>
    MSc AI<br>
    2024-2026
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# PAGE: OVERVIEW
# =============================================================================

if page == "🌊 Overview":

    st.markdown("""
    <div class='hero-header'>
        <div class='hero-subtitle'>MSc Thesis · AI Hydrology · Lebanon</div>
        <div class='hero-title'>Testing AI Models for<br><span>Climate-Resilient</span><br>Rainfall–Runoff Modeling</div>
        <br>
        <div class='hero-tags'>
            <span class='hero-tag'>Nahr Ibrahim Watershed</span>
            <span class='hero-tag'>LSTM · Transformer · CNN-LSTM · TFT</span>
            <span class='hero-tag'>GPM IMERG · MERRA-2 · GloFAS · MODIS</span>
            <span class='hero-tag'>SSP2-4.5 · SSP5-8.5</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    if df is not None:
        n_days   = len(df)
        n_years  = round(n_days / 365.25, 1)
        q_max    = df["discharge_m3s"].max()
        snow_max = df["snow_cover_pct"].max()

        st.markdown("""
        <div class='metric-grid'>
            <div class='metric-card'>
                <div class='metric-label'>Study Period</div>
                <div class='metric-value'>{}</div>
                <div class='metric-unit'>years of daily data</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Total Records</div>
                <div class='metric-value'>{:,}</div>
                <div class='metric-unit'>daily timesteps</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Peak Discharge</div>
                <div class='metric-value'>{:.2f}</div>
                <div class='metric-unit'>m³/s (GloFAS)</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Peak Snow Cover</div>
                <div class='metric-value'>{:.0f}%</div>
                <div class='metric-unit'>watershed area</div>
            </div>
        </div>
        """.format(n_years, n_days, q_max, snow_max), unsafe_allow_html=True)

    # Objectives
    st.markdown("<div class='section-header'>Research Objectives</div>", unsafe_allow_html=True)

    objectives = [
        ("Selection of AI models", "LSTM, Transformer, CNN-LSTM, Temporal Fusion Transformer"),
        ("Dataset preparation", "Precipitation, temperature, streamflow, drought/flood indices"),
        ("Train & benchmark models", "Historical and projected climate scenarios (SSP2-4.5, SSP5-8.5)"),
        ("Performance evaluation", "Nash-Sutcliffe, KGE, extremes, robustness, sensitivity analysis"),
        ("Identify best model", "Climate-resilient hydrological modelling in Nahr Ibrahim"),
    ]

    for i, (title, desc) in enumerate(objectives, 1):
        st.markdown(f"""
        <div class='pipeline-step'>
            <div class='step-number'>{i}</div>
            <div>
                <div class='step-title'>{title}</div>
                <div class='step-desc'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Watershed info
    st.markdown("<div class='section-header'>Watershed Characteristics</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='dataset-card'>
            <div class='dataset-icon'>🏔️</div>
            <div>
                <div class='dataset-name'>Location</div>
                <div class='dataset-meta'>Mount Lebanon · Byblos District · 34.09°N 35.88°E</div>
            </div>
        </div>
        <div class='dataset-card'>
            <div class='dataset-icon'>📐</div>
            <div>
                <div class='dataset-name'>Area</div>
                <div class='dataset-meta'>~326 km² · Karstic · Snowmelt-driven</div>
            </div>
        </div>
        <div class='dataset-card'>
            <div class='dataset-icon'>💧</div>
            <div>
                <div class='dataset-name'>Springs</div>
                <div class='dataset-meta'>Afqa · Roueiss · Karstic spring-dominated flow</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='dataset-card'>
            <div class='dataset-icon'>🌡️</div>
            <div>
                <div class='dataset-name'>Climate</div>
                <div class='dataset-meta'>Mediterranean · Wet winters · Dry summers</div>
            </div>
        </div>
        <div class='dataset-card'>
            <div class='dataset-icon'>⛰️</div>
            <div>
                <div class='dataset-name'>Elevation</div>
                <div class='dataset-meta'>Sea level → ~3,000m (Qornet es-Sawda)</div>
            </div>
        </div>
        <div class='dataset-card'>
            <div class='dataset-icon'>🔬</div>
            <div>
                <div class='dataset-name'>CZO Status</div>
                <div class='dataset-meta'>O-LIFE MISTRALS Critical Zone Observatory</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# PAGE: DATASET
# =============================================================================

elif page == "📡 Dataset":

    st.markdown("<div class='hero-title' style='font-family: DM Serif Display, serif; font-size: 2rem; margin-bottom: 0.5rem;'>Data Sources</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-family: DM Mono, monospace; font-size: 0.75rem; color: #4a6a82; margin-bottom: 2rem;'>REMOTE SENSING · REANALYSIS · LAND SURFACE MODELS</div>", unsafe_allow_html=True)

    datasets = [
        ("🌧️", "GPM IMERG Final Run", "NASA Giovanni", "Daily · 0.1° · 2000–2025", "Precipitation (mm/day)", "✅ Ready"),
        ("🌡️", "MERRA-2 T2M", "NASA Giovanni", "Hourly → Daily · 0.5° · 2000–2025", "Temperature (°C) — Tmean, Tmax, Tmin", "✅ Ready"),
        ("❄️", "GLDAS Noah SWE", "NASA Giovanni", "3-hourly → Daily · 0.25° · 2000–2025", "Snow Water Equivalent (mm)", "✅ Ready"),
        ("🏔️", "MODIS MOD10A1.061", "NASA AppEEARS", "Daily · 500m · 2000–2025", "Snow Cover Area (%) — polygon clipped", "✅ Ready"),
        ("🌊", "GloFAS ERA5 v4.0", "Copernicus CDS", "Daily · 0.05° · 2000–2025", "River Discharge (m³/s) — LISFLOOD", "✅ Ready"),
        ("🌍", "NEX-GDDP-CMIP6", "NASA / Copernicus", "Daily · 0.25° · 2015–2100", "Future projections — SSP2-4.5 & SSP5-8.5", "🔜 Pending"),
    ]

    for icon, name, source, resolution, variable, status in datasets:
        color = "#00b4a0" if "Ready" in status else "#f4a261"
        badge_bg = "rgba(0,180,160,0.15)" if "Ready" in status else "rgba(244,162,97,0.15)"
        badge_border = "rgba(0,180,160,0.3)" if "Ready" in status else "rgba(244,162,97,0.3)"
        st.markdown(f"""
        <div class='dataset-card'>
            <div class='dataset-icon'>{icon}</div>
            <div style='flex: 1;'>
                <div class='dataset-name'>{name}</div>
                <div class='dataset-meta'>{source} · {resolution}</div>
                <div style='font-size: 0.78rem; color: #8aafc4; margin-top: 0.3rem;'>{variable}</div>
            </div>
            <div style='background: {badge_bg}; border: 1px solid {badge_border}; color: {color}; padding: 0.25rem 0.8rem; border-radius: 50px; font-family: DM Mono, monospace; font-size: 0.68rem; white-space: nowrap;'>{status}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Spatial Coverage Note</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    📍 MODIS snow cover is clipped to the exact watershed GeoJSON polygon via rasterio.mask.<br>
    📦 Giovanni variables (IMERG, MERRA-2, GLDAS) use bounding box area-averaging (35.64–36.05°E, 33.99–34.21°N).<br>
    🎯 GloFAS discharge extracted from nearest grid cell to watershed centroid (34.093°N, 35.878°E).
    </div>
    """, unsafe_allow_html=True)

    # Feature engineering summary
    st.markdown("<div class='section-header'>Engineered Features</div>", unsafe_allow_html=True)

    features = {
        "precip_mm_day": "Raw daily precipitation",
        "precip_3day": "3-day antecedent precipitation index",
        "precip_7day": "7-day antecedent precipitation index",
        "temp_mean_c / max / min": "Daily Tmean, Tmax, Tmin from hourly MERRA-2",
        "temp_range_c": "Diurnal temperature range",
        "swe_mm": "Snow water equivalent",
        "swe_delta": "Daily SWE decrease (snowmelt proxy)",
        "snow_cover_pct": "% watershed snow-covered (NDSI ≥ 40)",
        "month_sin / month_cos": "Cyclical month encoding",
        "discharge_m3s": "Target variable — GloFAS daily discharge",
    }

    col1, col2 = st.columns(2)
    items = list(features.items())
    for i, (feat, desc) in enumerate(items):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div style='border-left: 2px solid #2a4d6e; padding: 0.5rem 0.8rem; margin-bottom: 0.6rem;'>
                <div style='font-family: DM Mono, monospace; font-size: 0.75rem; color: #3b9eff;'>{feat}</div>
                <div style='font-size: 0.8rem; color: #8aafc4; margin-top: 0.2rem;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# PAGE: EXPLORATORY ANALYSIS
# =============================================================================

elif page == "📈 Exploratory Analysis":

    if df is None:
        st.error("Master dataset not found. Run preprocess.py first.")
        st.stop()

    st.markdown("<div class='hero-title' style='font-family: DM Serif Display, serif; font-size: 2rem; margin-bottom: 0.5rem;'>Exploratory Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-family: DM Mono, monospace; font-size: 0.75rem; color: #4a6a82; margin-bottom: 2rem;'>2000 — 2025 · DAILY TIME SERIES</div>", unsafe_allow_html=True)

    # ── Seasonal climatology ──
    st.markdown("<div class='section-header'>Seasonal Climatology</div>", unsafe_allow_html=True)

    monthly = df.groupby("month").mean(numeric_only=True)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Precipitation (mm/day)", "Temperature (°C)",
                                        "Snow Cover (%)", "Discharge (m³/s)"),
                        vertical_spacing=0.14, horizontal_spacing=0.1)

    fig.add_trace(go.Bar(x=months, y=monthly["precip_mm_day"],
                         marker_color="#3b9eff", marker_opacity=0.85,
                         name="Precipitation"), row=1, col=1)

    fig.add_trace(go.Scatter(x=months, y=monthly["temp_mean_c"],
                             line=dict(color="#f4a261", width=2.5),
                             fill="tozeroy", fillcolor="rgba(244,162,97,0.08)",
                             mode="lines+markers",
                             marker=dict(size=7, color="#f4a261"),
                             name="Temperature"), row=1, col=2)

    fig.add_trace(go.Bar(x=months, y=monthly["snow_cover_pct"],
                         marker_color="#a8d8ea", marker_opacity=0.85,
                         name="Snow Cover"), row=2, col=1)

    fig.add_trace(go.Bar(x=months, y=monthly["discharge_m3s"],
                         marker_color="#00b4a0", marker_opacity=0.85,
                         name="Discharge"), row=2, col=2)

    fig.update_layout(height=520, showlegend=False,
                      title_text="", **PLOT_LAYOUT)
    fig.update_xaxes(gridcolor="#1e3448")
    fig.update_yaxes(gridcolor="#1e3448")

    st.plotly_chart(fig, use_container_width=True)

    # ── Full discharge time series ──
    st.markdown("<div class='section-header'>Discharge Time Series (2000–2025)</div>", unsafe_allow_html=True)

    var_options = {
        "Discharge (m³/s)": "discharge_m3s",
        "Precipitation (mm/day)": "precip_mm_day",
        "Temperature (°C)": "temp_mean_c",
        "Snow Cover (%)": "snow_cover_pct",
        "SWE (mm)": "swe_mm",
    }

    col1, col2 = st.columns([3, 1])
    with col2:
        selected_var = st.selectbox("Variable", list(var_options.keys()), index=0)
        year_range = st.slider("Year range", 2000, 2025, (2000, 2025))

    var_col = var_options[selected_var]
    df_filtered = df[(df["date"].dt.year >= year_range[0]) &
                     (df["date"].dt.year <= year_range[1])]

    colors_map = {
        "discharge_m3s": "#00b4a0",
        "precip_mm_day": "#3b9eff",
        "temp_mean_c":   "#f4a261",
        "snow_cover_pct":"#a8d8ea",
        "swe_mm":        "#00d4ff",
    }
    c = colors_map.get(var_col, "#3b9eff")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_filtered["date"], y=df_filtered[var_col],
        fill="tozeroy",
        fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.15)",
        line=dict(color=c, width=1.2),
        mode="lines",
        name=selected_var
    ))
    fig2.update_layout(height=350, **PLOT_LAYOUT,
                       xaxis_title="", yaxis_title=selected_var)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Correlation matrix ──
    st.markdown("<div class='section-header'>Correlation Matrix</div>", unsafe_allow_html=True)

    corr_cols = ["precip_mm_day", "temp_mean_c", "swe_mm",
                 "snow_cover_pct", "swe_delta", "precip_3day", "discharge_m3s"]
    corr = df[corr_cols].corr()

    labels_short = ["Precip", "Temp", "SWE", "Snow%", "ΔSnow", "P3d", "Q"]
    fig3 = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels_short, y=labels_short,
        colorscale=[[0, "#e76f51"], [0.5, "#111e2e"], [1, "#3b9eff"]],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11, family="DM Mono, monospace"),
        showscale=True,
    ))
    fig3.update_layout(height=400, **PLOT_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div class='info-box'>
    🔍 Key insight: Discharge (Q) correlates positively with precipitation and SWE, and negatively with temperature.
    The snowmelt lag (ΔSnow → Q) confirms karstic spring-mediated flow typical of Nahr Ibrahim.
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# PAGE: DATA SPLITS
# =============================================================================

elif page == "✂️ Data Splits":

    if not splits:
        st.error("Split files not found. Run split.py first.")
        st.stop()

    st.markdown("<div class='hero-title' style='font-family: DM Serif Display, serif; font-size: 2rem; margin-bottom: 0.5rem;'>Train / Validation / Test Split</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-family: DM Mono, monospace; font-size: 0.75rem; color: #4a6a82; margin-bottom: 2rem;'>CHRONOLOGICAL · NO RANDOM SHUFFLING</div>", unsafe_allow_html=True)

    train = splits.get("train")
    val   = splits.get("val")
    test  = splits.get("test")

    # Split cards
    st.markdown("""
    <div class='split-row'>
        <div class='split-card train'>
            <div class='split-label'>Training</div>
            <div class='split-period'>2000 – 2017</div>
            <div class='split-days'>6,575 days · 69.2%</div>
        </div>
        <div class='split-card val'>
            <div class='split-label'>Validation</div>
            <div class='split-period'>2018 – 2020</div>
            <div class='split-days'>1,096 days · 11.5%</div>
        </div>
        <div class='split-card test'>
            <div class='split-label'>Test</div>
            <div class='split-period'>2021 – 2025</div>
            <div class='split-days'>1,826 days · 19.2%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Full time series with split coloring
    st.markdown("<div class='section-header'>Discharge Time Series by Split</div>", unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train["date"], y=train["discharge_m3s"],
        fill="tozeroy", fillcolor="rgba(59,158,255,0.18)",
        line=dict(color="#3b9eff", width=1),
        name="Train (2000–2017)"
    ))
    fig.add_trace(go.Scatter(
        x=val["date"], y=val["discharge_m3s"],
        fill="tozeroy", fillcolor="rgba(244,162,97,0.25)",
        line=dict(color="#f4a261", width=1),
        name="Validation (2018–2020)"
    ))
    fig.add_trace(go.Scatter(
        x=test["date"], y=test["discharge_m3s"],
        fill="tozeroy", fillcolor="rgba(231,111,81,0.25)",
        line=dict(color="#e76f51", width=1),
        name="Test (2021–2025)"
    ))
    fig.add_vline(x="2018-01-01", line_dash="dash", line_color="#f4a261", line_width=1.5)
    fig.add_vline(x="2021-01-01", line_dash="dash", line_color="#e76f51", line_width=1.5)
    fig.update_layout(height=350, **PLOT_LAYOUT, yaxis_title="Discharge (m³/s)")
    st.plotly_chart(fig, use_container_width=True)

    # Distribution comparison
    st.markdown("<div class='section-header'>Distribution Comparison</div>", unsafe_allow_html=True)

    fig2 = go.Figure()
    color_map = {
        "Train":      ("#3b9eff", "rgba(59,158,255,0.2)"),
        "Validation": ("#f4a261", "rgba(244,162,97,0.2)"),
        "Test":       ("#e76f51", "rgba(231,111,81,0.2)"),
    }
    for name, data in [("Train", train), ("Validation", val), ("Test", test)]:
        line_color, fill_color = color_map[name]
        fig2.add_trace(go.Violin(
            y=data["discharge_m3s"],
            name=name,
            fillcolor=fill_color,
            line_color=line_color,
            box_visible=True,
            meanline_visible=True,
            points=False,
        ))
    fig2.update_layout(height=380, **PLOT_LAYOUT, yaxis_title="Discharge (m³/s)")
    st.plotly_chart(fig2, use_container_width=True)

    # Stats table
    st.markdown("<div class='section-header'>Statistics per Split</div>", unsafe_allow_html=True)

    stats_cols = ["precip_mm_day", "temp_mean_c", "swe_mm", "snow_cover_pct", "discharge_m3s"]
    stats_data = []
    for name, data in [("Train", train), ("Validation", val), ("Test", test)]:
        if data is not None:
            row = {"Split": name}
            for col in stats_cols:
                row[col] = f"{data[col].mean():.3f}"
            stats_data.append(row)

    stats_df = pd.DataFrame(stats_data)
    stats_df.columns = ["Split", "Precip (mm/d)", "Temp (°C)", "SWE (mm)", "Snow (%)", "Q (m³/s)"]
    st.dataframe(stats_df.set_index("Split"), use_container_width=True)

    st.markdown("""
    <div class='info-box'>
    ⚠️ Out-of-range values after normalization: Val=3 · Test=7 — Minor distribution shift detected,
    consistent with observed warming trend (+0.8°C from train to test) and increasing SWE.
    This non-stationarity is the core motivation for climate-resilient AI modeling.
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# PAGE: VARIABLE INSPECTOR
# =============================================================================

elif page == "🔬 Variable Inspector":

    if df is None:
        st.error("Master dataset not found.")
        st.stop()

    st.markdown("<div class='hero-title' style='font-family: DM Serif Display, serif; font-size: 2rem; margin-bottom: 0.5rem;'>Variable Inspector</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-family: DM Mono, monospace; font-size: 0.75rem; color: #4a6a82; margin-bottom: 2rem;'>DEEP DIVE · STATISTICS · DISTRIBUTIONS</div>", unsafe_allow_html=True)

    var_map = {
        "Discharge (m³/s)":     "discharge_m3s",
        "Precipitation (mm/d)": "precip_mm_day",
        "Temperature mean (°C)":"temp_mean_c",
        "Temp max (°C)":        "temp_max_c",
        "Temp min (°C)":        "temp_min_c",
        "SWE (mm)":             "swe_mm",
        "Snow Cover (%)":       "snow_cover_pct",
        "SWE Delta (mm)":       "swe_delta",
        "Precip 3-day (mm)":    "precip_3day",
        "Precip 7-day (mm)":    "precip_7day",
    }

    selected = st.selectbox("Select variable", list(var_map.keys()))
    col = var_map[selected]
    series = df[col].dropna()

    # Stats row
    c1, c2, c3, c4, c5 = st.columns(5)
    for container, label, val in [
        (c1, "Mean",   f"{series.mean():.3f}"),
        (c2, "Std",    f"{series.std():.3f}"),
        (c3, "Min",    f"{series.min():.3f}"),
        (c4, "Max",    f"{series.max():.3f}"),
        (c5, "Median", f"{series.median():.3f}"),
    ]:
        container.markdown(f"""
        <div class='metric-card' style='margin-bottom: 1rem;'>
            <div class='metric-label'>{label}</div>
            <div style='font-family: DM Serif Display, serif; font-size: 1.5rem; color: #00d4ff;'>{val}</div>
        </div>
        """, unsafe_allow_html=True)

    # Time series + histogram side by side
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3],
                        subplot_titles=("Time Series", "Distribution"))

    c_hex = "#3b9eff"
    fig.add_trace(go.Scatter(
        x=df["date"], y=df[col],
        line=dict(color=c_hex, width=0.8),
        fill="tozeroy", fillcolor="rgba(59,158,255,0.1)",
        name=selected
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        y=series, nbinsy=60,
        marker_color=c_hex, marker_opacity=0.7,
        name="Distribution"
    ), row=1, col=2)

    fig.update_layout(height=380, showlegend=False, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Annual trend
    st.markdown("<div class='section-header'>Annual Trend</div>", unsafe_allow_html=True)

    df["year"] = df["date"].dt.year
    annual = df.groupby("year")[col].mean()

    z = np.polyfit(annual.index, annual.values, 1)
    trend = np.poly1d(z)(annual.index)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=annual.index, y=annual.values,
                          marker_color="#3b9eff", marker_opacity=0.7, name="Annual mean"))
    fig3.add_trace(go.Scatter(x=annual.index, y=trend,
                              line=dict(color="#e76f51", width=2, dash="dash"),
                              name=f"Trend ({z[0]:+.4f}/yr)"))
    fig3.update_layout(height=320, **PLOT_LAYOUT, yaxis_title=selected)
    st.plotly_chart(fig3, use_container_width=True)

# =============================================================================
# PAGE: PIPELINE SUMMARY
# =============================================================================

elif page == "📋 Pipeline Summary":

    st.markdown("<div class='hero-title' style='font-family: DM Serif Display, serif; font-size: 2rem; margin-bottom: 0.5rem;'>Pipeline Summary</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-family: DM Mono, monospace; font-size: 0.75rem; color: #4a6a82; margin-bottom: 2rem;'>DATA ENGINEERING · STATUS · NEXT STEPS</div>", unsafe_allow_html=True)

    steps = [
        ("✅", "Data Collection", "Downloaded IMERG, MERRA-2, GLDAS SWE from NASA Giovanni as CSV. GloFAS discharge from Copernicus CDS as GRIB. MODIS MOD10A1 snow cover from AppEEARS as GeoTIFF (chunked by 5-year periods)."),
        ("✅", "Preprocessing", "Parsed Giovanni non-standard CSV headers. Unit conversions (mm/hr→mm/day removed — GPM_3IMERGDF already daily). K→°C removed (MERRA-2 in Celsius). 3-hourly GLDAS resampled to daily. MODIS NDSI threshold set to ≥40."),
        ("✅", "MODIS Clipping", "Each daily GeoTIFF clipped to exact watershed GeoJSON polygon using rasterio.mask. Snow cover % computed as fraction of pixels with NDSI ≥ 40 over all valid pixels (0–100 range)."),
        ("✅", "Duplicate Removal", "AppEEARS chunked downloads created overlapping dates. Deduplicated by keeping highest snow_cover_pct per day (least cloudy). Removed 19,984 duplicates from 27,249 files → 7,265 unique days."),
        ("✅", "GloFAS Extraction", "312 yearly zip files extracted with unique naming to prevent overwrite. GRIB files read with cfgrib. dis24 variable extracted at centroid (34.093°N, 35.878°E) using nearest-neighbor. 9,497 daily rows."),
        ("✅", "Gap Filling", "Snow: linear interpolation ≤5 days, then monthly climatology. SWE/Temp: linear interpolation ≤3 days. Precip: NaN→0. Result: 0 missing values in all variables except swe_delta (1 row, first day)."),
        ("✅", "Feature Engineering", "Derived: precip_3day, precip_7day (rolling sums), temp_range_c, swe_delta (snowmelt proxy), month_sin/cos (cyclical encoding), season labels."),
        ("✅", "Train/Val/Test Split", "Chronological split: Train 2000–2017 (6,575 days), Val 2018–2020 (1,096), Test 2021–2025 (1,826). Min-max normalization fitted on train only. Scaler params saved for inverse transform."),
        ("🔜", "Sequence Windowing", "Create sliding window sequences (lookback=30 days) for LSTM/Transformer input. Shape: (samples, timesteps, features)."),
        ("🔜", "Model Training", "Train LSTM, Transformer, CNN-LSTM, TFT on training set. Early stopping on validation NSE."),
        ("🔜", "Evaluation", "Nash-Sutcliffe Efficiency, KGE, RMSE, peak flow bias on test set. Sensitivity analysis. Robustness under projected scenarios."),
        ("🔜", "Climate Scenarios", "Apply trained models to NEX-GDDP-CMIP6 projections (SSP2-4.5, SSP5-8.5) for 2015–2100."),
    ]

    for icon, title, desc in steps:
        color = "#00b4a0" if icon == "✅" else "#4a6a82"
        st.markdown(f"""
        <div class='pipeline-step'>
            <div style='font-size: 1.3rem; min-width: 32px; text-align: center;'>{icon}</div>
            <div>
                <div class='step-title' style='color: {color};'>{title}</div>
                <div class='step-desc'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Files produced
    st.markdown("<div class='section-header'>Files Produced</div>", unsafe_allow_html=True)

    files = [
        ("data/processed/", "precip_imerg_daily.csv · temp_merra2_daily.csv · swe_gldas_daily.csv · discharge_glofas_daily.csv · snow_cover_modis_daily.csv"),
        ("data/master/", "nahr_ibrahim_master_full.csv (9,497 rows × 16 cols) · nahr_ibrahim_master_model.csv"),
        ("data/splits/", "train_raw.csv · val_raw.csv · test_raw.csv · train_norm.csv · val_norm.csv · test_norm.csv · scaler_params.csv"),
        ("results/figures/", "seasonal_overview.png · train_val_test_split.png"),
    ]

    for folder, contents in files:
        st.markdown(f"""
        <div style='border-left: 2px solid #2a4d6e; padding: 0.6rem 1rem; margin-bottom: 0.8rem;'>
            <div style='font-family: DM Mono, monospace; font-size: 0.75rem; color: #3b9eff;'>{folder}</div>
            <div style='font-size: 0.78rem; color: #8aafc4; margin-top: 0.3rem;'>{contents}</div>
        </div>
        """, unsafe_allow_html=True)