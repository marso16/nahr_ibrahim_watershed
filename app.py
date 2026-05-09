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
from plotly.subplots import make_subplots
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Nahr Ibrahim · AI Hydrology",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');
:root {
    --bg-primary:#080f1a; --bg-secondary:#0d1825; --bg-card:#111e2e;
    --accent-blue:#3b9eff; --accent-cyan:#00d4ff; --accent-teal:#00b4a0;
    --accent-snow:#a8d8ea; --accent-warm:#f4a261; --accent-red:#e76f51;
    --accent-purple:#a855f7;
    --text-primary:#e8f4f8; --text-secondary:#8aafc4; --text-muted:#4a6a82;
    --border:#1e3448; --border-bright:#2a4d6e;
}
html,body,[class*="css"]{font-family:'Syne',sans-serif;background-color:var(--bg-primary)!important;color:var(--text-primary)!important;}
.stApp{background:linear-gradient(135deg,#060d16 0%,#080f1a 50%,#0a1520 100%);}
section[data-testid="stSidebar"]{background:var(--bg-secondary)!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] *{color:var(--text-primary)!important;}
#MainMenu,footer,header{visibility:hidden;}
.stDeployButton{display:none;}
.hero-header{background:linear-gradient(135deg,#0d2137 0%,#0a1a2e 40%,#061020 100%);border:1px solid var(--border-bright);border-radius:20px;padding:3rem 3.5rem;margin-bottom:2rem;position:relative;overflow:hidden;}
.hero-title{font-family:'DM Serif Display',serif;font-size:2.8rem;line-height:1.1;color:var(--text-primary);margin:0 0 0.5rem 0;}
.hero-title span{color:var(--accent-cyan);}
.hero-subtitle{font-family:'DM Mono',monospace;font-size:0.8rem;color:var(--text-muted);letter-spacing:.15em;text-transform:uppercase;margin-bottom:1.5rem;}
.hero-tags{display:flex;gap:.6rem;flex-wrap:wrap;}
.hero-tag{background:rgba(59,158,255,.1);border:1px solid rgba(59,158,255,.25);color:var(--accent-blue);padding:.3rem .9rem;border-radius:50px;font-size:.75rem;font-family:'DM Mono',monospace;}
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:2rem;}
.metric-card{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:1.4rem 1.6rem;position:relative;overflow:hidden;}
.metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent-blue),var(--accent-cyan));opacity:.6;}
.metric-label{font-family:'DM Mono',monospace;font-size:.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.12em;margin-bottom:.6rem;}
.metric-value{font-family:'DM Serif Display',serif;font-size:2rem;color:var(--accent-cyan);line-height:1;margin-bottom:.3rem;}
.metric-unit{font-family:'DM Mono',monospace;font-size:.72rem;color:var(--text-secondary);}
.section-header{font-family:'DM Mono',monospace;font-size:.7rem;color:var(--accent-blue);letter-spacing:.2em;text-transform:uppercase;margin:2.5rem 0 1rem 0;display:flex;align-items:center;gap:.8rem;}
.section-header::after{content:'';flex:1;height:1px;background:var(--border);}
.dataset-card{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:.8rem;display:flex;align-items:center;gap:1rem;}
.dataset-icon{font-size:1.6rem;min-width:40px;text-align:center;}
.dataset-name{font-size:.9rem;font-weight:600;color:var(--text-primary);margin-bottom:.2rem;}
.dataset-meta{font-family:'DM Mono',monospace;font-size:.7rem;color:var(--text-muted);}
.split-row{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1rem 0;}
.split-card{border-radius:12px;padding:1.2rem;text-align:center;}
.split-card.train{background:rgba(59,158,255,.08);border:1px solid rgba(59,158,255,.2);}
.split-card.val{background:rgba(244,162,97,.08);border:1px solid rgba(244,162,97,.2);}
.split-card.test{background:rgba(231,111,81,.08);border:1px solid rgba(231,111,81,.2);}
.split-label{font-family:'DM Mono',monospace;font-size:.68rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.1em;}
.split-period{font-family:'DM Serif Display',serif;font-size:1.1rem;margin:.3rem 0;}
.split-card.train .split-period{color:var(--accent-blue);}
.split-card.val .split-period{color:var(--accent-warm);}
.split-card.test .split-period{color:var(--accent-red);}
.split-days{font-family:'DM Mono',monospace;font-size:.75rem;color:var(--text-secondary);}
.pipeline-step{display:flex;align-items:flex-start;gap:1.2rem;padding:1rem 0;border-bottom:1px solid var(--border);}
.step-number{background:linear-gradient(135deg,var(--accent-blue),var(--accent-cyan));color:#000;width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'DM Mono',monospace;font-size:.8rem;font-weight:600;flex-shrink:0;margin-top:2px;}
.step-title{font-size:.95rem;font-weight:600;color:var(--text-primary);}
.step-desc{font-family:'DM Mono',monospace;font-size:.72rem;color:var(--text-muted);margin-top:.3rem;}
.info-box{background:rgba(59,158,255,.06);border:1px solid rgba(59,158,255,.2);border-left:3px solid var(--accent-blue);border-radius:8px;padding:1rem 1.2rem;font-size:.85rem;color:var(--text-secondary);margin:1rem 0;font-family:'DM Mono',monospace;}
.warn-box{background:rgba(244,162,97,.06);border:1px solid rgba(244,162,97,.2);border-left:3px solid var(--accent-warm);border-radius:8px;padding:1rem 1.2rem;font-size:.85rem;color:var(--text-secondary);margin:1rem 0;font-family:'DM Mono',monospace;}
.results-table{width:100%;border-collapse:collapse;margin:1rem 0;}
.results-table th{background:#1F3864;color:#fff;font-family:'DM Mono',monospace;font-size:.72rem;padding:8px 12px;text-align:left;letter-spacing:.05em;}
.results-table td{font-family:'DM Mono',monospace;font-size:.78rem;padding:7px 12px;border-bottom:1px solid var(--border);color:var(--text-secondary);}
.results-table tr:nth-child(even) td{background:rgba(255,255,255,.02);}
.best{color:var(--accent-cyan)!important;font-weight:600;}
.model-pure{color:var(--accent-blue);}
.model-hybrid{color:var(--accent-purple);}
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:var(--bg-primary);}
::-webkit-scrollbar-thumb{background:var(--border-bright);border-radius:3px;}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# PATHS & DATA LOADING
# =============================================================================

ROOT = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")


@st.cache_data
def load_master():
    p = ROOT / "data" / "master" / "nahr_ibrahim_master_full.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


@st.cache_data
def load_splits():
    splits = {}
    for name in ["train", "val", "test"]:
        p = ROOT / "data" / "splits" / f"{name}_raw.csv"
        if p.exists():
            splits[name] = pd.read_csv(p, parse_dates=["date"])
    return splits


@st.cache_data
def load_scenarios():
    results = {}
    for scen in ["ssp245", "ssp585"]:
        p = ROOT / "results" / "scenarios" / f"discharge_{scen}_daily.csv"
        if p.exists():
            results[scen] = pd.read_csv(p, parse_dates=["date"])
    return results


df = load_master()
splits = load_splits()
scenarios = load_scenarios()

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
    st.markdown(
        """
    <div style='padding:1rem 0 2rem 0;'>
        <div style='font-family:DM Serif Display,serif;font-size:1.4rem;color:#e8f4f8;'>Nahr Ibrahim</div>
        <div style='font-family:DM Mono,monospace;font-size:.65rem;color:#4a6a82;letter-spacing:.15em;text-transform:uppercase;margin-top:.3rem;'>Watershed · AI Hydrology</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "🌊 Overview",
            "📡 Dataset",
            "📈 Exploratory Analysis",
            "✂️ Data Splits",
            "🤖 Model Results",
            "🌍 Climate Scenarios",
            "🔬 Variable Inspector",
            "📋 Pipeline Summary",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
    <div style='font-family:DM Mono,monospace;font-size:.68rem;color:#4a6a82;line-height:1.8;'>
    <div style='color:#8aafc4;margin-bottom:.5rem;'>THESIS INFO</div>
    USJ · ESIB<br>MSc AI in Engineering<br>Supervisor: Dr. C. Andraos<br>2024–2026
    </div>
    """,
        unsafe_allow_html=True,
    )

# =============================================================================
# PAGE: OVERVIEW
# =============================================================================

if page == "🌊 Overview":
    st.markdown(
        """
    <div class='hero-header'>
        <div class='hero-subtitle'>MSc Thesis · AI Hydrology · Lebanon</div>
        <div class='hero-title'>Testing AI Models for<br><span>Climate-Resilient</span><br>Rainfall–Runoff Modeling</div>
        <br>
        <div class='hero-tags'>
            <span class='hero-tag'>Nahr Ibrahim Watershed · 330 km²</span>
            <span class='hero-tag'>LSTM · CNN-LSTM · Transformer · PI-LSTM · PI-Transformer</span>
            <span class='hero-tag'>CHIRPS · MERRA-2 · GLDAS · GloFAS · MODIS</span>
            <span class='hero-tag'>SSP2-4.5 · SSP5-8.5 · 2015–2100</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if df is not None:
        n_days = len(df)
        n_years = round(n_days / 365.25, 1)
        q_max = df["discharge_m3s"].max()
        snow_max = df["snow_cover_pct"].max()
        st.markdown(
            f"""
        <div class='metric-grid'>
            <div class='metric-card'><div class='metric-label'>Study Period</div><div class='metric-value'>{n_years}</div><div class='metric-unit'>years · 2000–2025</div></div>
            <div class='metric-card'><div class='metric-label'>Total Records</div><div class='metric-value'>{n_days:,}</div><div class='metric-unit'>daily timesteps · 16 features</div></div>
            <div class='metric-card'><div class='metric-label'>Peak Discharge</div><div class='metric-value'>{q_max:.2f}</div><div class='metric-unit'>m³/s (GloFAS ERA5)</div></div>
            <div class='metric-card'><div class='metric-label'>Peak Snow Cover</div><div class='metric-value'>{snow_max:.0f}%</div><div class='metric-unit'>watershed area (MODIS)</div></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='section-header'>Key Results — Test Period 2021–2025</div>",
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)
    for c, label, val, sub in [
        (col1, "Best NSE", "0.680", "Transformer"),
        (col2, "Best KGE", "0.730", "PI-Transformer"),
        (col3, "Best Peak Bias", "-29.7%", "PI-Transformer"),
        (col4, "SSP5-8.5 Δ", "-8.5%", "2015→2100 Q change"),
    ]:
        c.markdown(
            f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value' style='font-size:1.6rem'>{val}</div>
            <div class='metric-unit'>{sub}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='section-header'>Research Objectives</div>", unsafe_allow_html=True
    )
    for i, (title, desc) in enumerate(
        [
            (
                "Data collection & validation",
                "CHIRPS precipitation · MERRA-2 temperature · GLDAS SWE/soil moisture · MODIS snow · GloFAS discharge",
            ),
            (
                "Feature engineering",
                "16 hydrological features: precip indices · temperature · cryosphere · soil moisture + anomaly · Hamon PET",
            ),
            (
                "Model development",
                "5 architectures: LSTM · CNN-LSTM · Transformer (pure AI) + PI-LSTM · PI-Transformer (physics-informed hybrid)",
            ),
            (
                "Performance evaluation",
                "7 metrics: NSE · KGE · RMSE · MAE · PBIAS · Peak Bias · Log-NSE on chronological test set 2021–2025",
            ),
            (
                "Climate scenario projection",
                "NEX-GDDP-CMIP6 MPI-ESM1-2-HR · SSP2-4.5 & SSP5-8.5 · 2015–2100 · Mann-Kendall trend analysis",
            ),
        ],
        1,
    ):
        st.markdown(
            f"""
        <div class='pipeline-step'>
            <div class='step-number'>{i}</div>
            <div><div class='step-title'>{title}</div><div class='step-desc'>{desc}</div></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='section-header'>Watershed Characteristics</div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        for icon, name, meta in [
            (
                "🏔️",
                "Location",
                "Mount Lebanon · Byblos District · Keserwan-Jbeil Governorate",
            ),
            (
                "📐",
                "Area & Elevation",
                "330 km² · 2–2,684 m asl · Mean 1,571 m · Slope 0.1°–62.7°",
            ),
            (
                "💧",
                "Karstic Springs",
                "Afqa 34.068°N 35.894°E ~1,200 m · Roueiss 34.109°N 35.908°E ~1,265 m",
            ),
        ]:
            st.markdown(
                f"<div class='dataset-card'><div class='dataset-icon'>{icon}</div><div><div class='dataset-name'>{name}</div><div class='dataset-meta'>{meta}</div></div></div>",
                unsafe_allow_html=True,
            )
    with col2:
        for icon, name, meta in [
            (
                "🌡️",
                "Climate",
                "Mediterranean · 900–1,400 mm/yr precip · 25–35% as snow",
            ),
            (
                "🌊",
                "River Outlet",
                "34.062°N, 35.642°E · Mediterranean Sea · ~23 km river length",
            ),
            (
                "📡",
                "Discharge Target",
                "GloFAS ERA5 v4.0 · 0.05° · Q range 0.227–7.312 m³/s",
            ),
        ]:
            st.markdown(
                f"<div class='dataset-card'><div class='dataset-icon'>{icon}</div><div><div class='dataset-name'>{name}</div><div class='dataset-meta'>{meta}</div></div></div>",
                unsafe_allow_html=True,
            )

# =============================================================================
# PAGE: DATASET
# =============================================================================

elif page == "📡 Dataset":
    st.markdown(
        "<div class='hero-title' style='font-family:DM Serif Display,serif;font-size:2rem;margin-bottom:.5rem;'>Data Sources</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:.75rem;color:#4a6a82;margin-bottom:2rem;'>REMOTE SENSING · REANALYSIS · LAND SURFACE MODELS</div>",
        unsafe_allow_html=True,
    )

    for icon, name, source, res, var, status in [
        (
            "🌧️",
            "CHIRPS v2.0",
            "Google Earth Engine",
            "Daily · 0.05° · 2000–2025",
            "Precipitation (mm/day) — gauge-corrected",
            "✅ Ready",
        ),
        (
            "🌡️",
            "MERRA-2 T2M",
            "NASA Giovanni",
            "Hourly→Daily · 0.5° · 2000–2025",
            "Temperature — Tmean/Tmax/Tmin (°C)",
            "✅ Ready",
        ),
        (
            "❄️",
            "GLDAS Noah v2.1",
            "NASA Giovanni",
            "3-hourly→Daily · 0.25° · 2000–2025",
            "SWE (mm) + Soil Moisture 0–10 cm",
            "✅ Ready",
        ),
        (
            "🏔️",
            "MODIS MOD10A1.061",
            "NASA AppEEARS",
            "Daily · 500m · 2000–2025",
            "Snow Cover (%) — NDSI ≥ 40",
            "✅ Ready",
        ),
        (
            "🌊",
            "GloFAS ERA5 v4.0",
            "Copernicus CDS",
            "Daily · 0.05° · 2000–2025",
            "River Discharge (m³/s) — LISFLOOD routing",
            "✅ Ready",
        ),
        (
            "🌍",
            "NEX-GDDP-CMIP6",
            "Google Earth Engine",
            "Daily · 0.25° · 2015–2100",
            "MPI-ESM1-2-HR · pr · tas · tasmin · tasmax",
            "⏳ Downloading",
        ),
    ]:
        ok = "Ready" in status
        color = "#00b4a0" if ok else "#f4a261"
        bbg = "rgba(0,180,160,.15)" if ok else "rgba(244,162,97,.15)"
        bbd = "rgba(0,180,160,.3)" if ok else "rgba(244,162,97,.3)"
        st.markdown(
            f"""
        <div class='dataset-card'>
            <div class='dataset-icon'>{icon}</div>
            <div style='flex:1;'>
                <div class='dataset-name'>{name}</div>
                <div class='dataset-meta'>{source} · {res}</div>
                <div style='font-size:.78rem;color:#8aafc4;margin-top:.3rem;'>{var}</div>
            </div>
            <div style='background:{bbg};border:1px solid {bbd};color:{color};padding:.25rem .8rem;border-radius:50px;font-family:DM Mono,monospace;font-size:.68rem;white-space:nowrap;'>{status}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='section-header'>Spatial Configuration</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <div class='info-box'>
    📍 Spring recharge zone bounding box: 34.02°–34.16°N, 35.84°–35.96°E<br>
    📍 Afqa spring: 34.068°N, 35.894°E · Elevation ~1,200 m asl<br>
    📍 Roueiss spring: 34.109°N, 35.908°E · Elevation ~1,265 m asl<br>
    📍 River outlet: 34.062°N, 35.642°E · Mediterranean Sea<br>
    📦 MODIS clipped to exact watershed GeoJSON polygon via rasterio.mask<br>
    🎯 GloFAS extracted at outlet (34.062°N, 35.642°E) — nearest grid cell
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='section-header'>16 Engineered Features</div>",
        unsafe_allow_html=True,
    )
    features = [
        ("precip_mm_day", "Daily precipitation", "CHIRPS", False),
        ("precip_3day", "3-day antecedent precipitation sum", "Computed", False),
        ("precip_7day", "7-day antecedent precipitation sum", "Computed", False),
        ("temp_mean_c", "Daily mean air temperature", "MERRA-2", False),
        ("temp_max_c", "Daily maximum air temperature", "MERRA-2", False),
        ("temp_min_c", "Daily minimum air temperature", "MERRA-2", False),
        ("temp_range_c", "Diurnal temperature range", "Computed", False),
        ("swe_mm", "Snow water equivalent", "GLDAS", False),
        ("swe_delta", "Daily SWE change — snowmelt proxy", "Computed", False),
        ("snow_cover_pct", "% watershed snow-covered (NDSI ≥ 40)", "MODIS", False),
        ("month_sin", "Cyclical month — sine component", "Computed", False),
        ("month_cos", "Cyclical month — cosine component", "Computed", False),
        ("soil_moisture_mm", "Soil moisture 0–10 cm       ", "GLDAS", False),
        ("sm_7day_mean", "7-day antecedent soil moisture       ", "Computed", False),
        ("sm_anomaly", "Deviation from 30-day mean       ", "Computed", False),
        ("pet_mm_day", "Hamon (1961) PET       ", "Computed", False),
    ]
    col1, col2 = st.columns(2)
    for i, (feat, desc, src, is_new) in enumerate(features):
        color = "#a855f7" if is_new else "#3b9eff"
        with col1 if i % 2 == 0 else col2:
            st.markdown(
                f"""
            <div style='border-left:2px solid {color};padding:.5rem .8rem;margin-bottom:.6rem;'>
                <div style='font-family:DM Mono,monospace;font-size:.75rem;color:{color};'>{feat}</div>
                <div style='font-size:.78rem;color:#8aafc4;margin-top:.2rem;'>{desc}</div>
                <div style='font-family:DM Mono,monospace;font-size:.65rem;color:#4a6a82;margin-top:.1rem;'>Source: {src}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
    <div class='info-box'>
    ★ Adding soil moisture + PET: Transformer NSE +0.077 · PI-Transformer NSE +0.156 · PI-Transformer KGE +0.156
    </div>
    """,
        unsafe_allow_html=True,
    )

# =============================================================================
# PAGE: EXPLORATORY ANALYSIS
# =============================================================================

elif page == "📈 Exploratory Analysis":
    if df is None:
        st.error("Master dataset not found. Run preprocess.py first.")
        st.stop()

    st.markdown(
        "<div class='hero-title' style='font-family:DM Serif Display,serif;font-size:2rem;margin-bottom:.5rem;'>Exploratory Analysis</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:.75rem;color:#4a6a82;margin-bottom:2rem;'>2000–2025 · DAILY · 16 FEATURES</div>",
        unsafe_allow_html=True,
    )

    df["month"] = df["date"].dt.month
    monthly = df.groupby("month").mean(numeric_only=True)
    months = [
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

    st.markdown(
        "<div class='section-header'>Seasonal Climatology</div>", unsafe_allow_html=True
    )
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Precipitation (mm/day)",
            "Temperature (°C)",
            "Snow Cover (%)",
            "Discharge (m³/s)",
            "Soil Moisture (mm)",
            "PET (mm/day)",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Bar(
            x=months,
            y=monthly["precip_mm_day"],
            marker_color="#3b9eff",
            marker_opacity=0.85,
            name="Precip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=monthly["temp_mean_c"],
            line=dict(color="#f4a261", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(244,162,97,.08)",
            mode="lines+markers",
            marker=dict(size=7, color="#f4a261"),
            name="Temp",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=months,
            y=monthly["snow_cover_pct"],
            marker_color="#a8d8ea",
            marker_opacity=0.85,
            name="Snow",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Bar(
            x=months,
            y=monthly["discharge_m3s"],
            marker_color="#00b4a0",
            marker_opacity=0.85,
            name="Q",
        ),
        row=2,
        col=1,
    )
    if "soil_moisture_mm" in monthly.columns:
        fig.add_trace(
            go.Scatter(
                x=months,
                y=monthly["soil_moisture_mm"],
                line=dict(color="#a855f7", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(168,85,247,.08)",
                mode="lines+markers",
                marker=dict(size=7, color="#a855f7"),
                name="SM",
            ),
            row=2,
            col=2,
        )
    if "pet_mm_day" in monthly.columns:
        fig.add_trace(
            go.Bar(
                x=months,
                y=monthly["pet_mm_day"],
                marker_color="#e76f51",
                marker_opacity=0.85,
                name="PET",
            ),
            row=2,
            col=3,
        )
    fig.update_layout(height=580, showlegend=False, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div class='section-header'>Time Series Explorer</div>", unsafe_allow_html=True
    )
    var_options = {
        "Discharge (m³/s)": "discharge_m3s",
        "Precipitation (mm/day)": "precip_mm_day",
        "Temperature (°C)": "temp_mean_c",
        "Snow Cover (%)": "snow_cover_pct",
        "SWE (mm)": "swe_mm",
        "Soil Moisture (mm)": "soil_moisture_mm",
        "PET (mm/day)": "pet_mm_day",
        "SM Anomaly (mm)": "sm_anomaly",
    }
    col1, col2 = st.columns([3, 1])
    with col2:
        selected_var = st.selectbox(
            "Variable", [k for k, v in var_options.items() if v in df.columns]
        )
        year_range = st.slider("Year range", 2000, 2025, (2000, 2025))
    var_col = var_options[selected_var]
    df_f = df[
        (df["date"].dt.year >= year_range[0]) & (df["date"].dt.year <= year_range[1])
    ]
    cmap = {
        "discharge_m3s": "#00b4a0",
        "precip_mm_day": "#3b9eff",
        "temp_mean_c": "#f4a261",
        "snow_cover_pct": "#a8d8ea",
        "swe_mm": "#00d4ff",
        "soil_moisture_mm": "#a855f7",
        "pet_mm_day": "#e76f51",
        "sm_anomaly": "#f4a261",
    }
    c = cmap.get(var_col, "#3b9eff")
    fig2 = go.Figure(
        go.Scatter(
            x=df_f["date"],
            y=df_f[var_col],
            fill="tozeroy",
            fillcolor=f"rgba({int(c[1:3], 16)},{int(c[3:5], 16)},{int(c[5:7], 16)},.15)",
            line=dict(color=c, width=1.2),
            mode="lines",
            name=selected_var,
        )
    )
    fig2.update_layout(height=350, **PLOT_LAYOUT, yaxis_title=selected_var)
    with col1:
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        "<div class='section-header'>Correlation Matrix</div>", unsafe_allow_html=True
    )
    corr_cols = [
        c
        for c in [
            "precip_mm_day",
            "temp_mean_c",
            "swe_mm",
            "snow_cover_pct",
            "swe_delta",
            "precip_3day",
            "soil_moisture_mm",
            "pet_mm_day",
            "discharge_m3s",
        ]
        if c in df.columns
    ]
    corr = df[corr_cols].corr()
    short = ["Precip", "Temp", "SWE", "Snow%", "ΔSWE", "P3d", "SM", "PET", "Q"][
        : len(corr_cols)
    ]
    fig3 = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=short,
            y=short,
            colorscale=[[0, "#e76f51"], [0.5, "#111e2e"], [1, "#3b9eff"]],
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=11, family="DM Mono, monospace"),
            showscale=True,
        )
    )
    fig3.update_layout(height=420, **PLOT_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

# =============================================================================
# PAGE: DATA SPLITS
# =============================================================================

elif page == "✂️ Data Splits":
    if not splits:
        st.error("Split files not found. Run split.py first.")
        st.stop()

    st.markdown(
        "<div class='hero-title' style='font-family:DM Serif Display,serif;font-size:2rem;margin-bottom:.5rem;'>Train / Validation / Test Split</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:.75rem;color:#4a6a82;margin-bottom:2rem;'>CHRONOLOGICAL · NO SHUFFLING · 30-DAY LOOKBACK · 16 FEATURES</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class='split-row'>
        <div class='split-card train'><div class='split-label'>Training</div><div class='split-period'>2000 – 2017</div><div class='split-days'>~6,515 sequences · 69%</div></div>
        <div class='split-card val'><div class='split-label'>Validation</div><div class='split-period'>2018 – 2020</div><div class='split-days'>1,066 sequences · 11%</div></div>
        <div class='split-card test'><div class='split-label'>Test</div><div class='split-period'>2021 – 2025</div><div class='split-days'>1,796 sequences · 19%</div></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    train = splits.get("train")
    val = splits.get("val")
    test = splits.get("test")
    fig = go.Figure()
    for name, data, color, fill in [
        ("Train (2000–2017)", train, "#3b9eff", "rgba(59,158,255,.18)"),
        ("Validation (2018–2020)", val, "#f4a261", "rgba(244,162,97,.25)"),
        ("Test (2021–2025)", test, "#e76f51", "rgba(231,111,81,.25)"),
    ]:
        if data is not None:
            fig.add_trace(
                go.Scatter(
                    x=data["date"],
                    y=data["discharge_m3s"],
                    fill="tozeroy",
                    fillcolor=fill,
                    line=dict(color=color, width=1),
                    name=name,
                )
            )
    fig.add_vline(
        x="2018-01-01", line_dash="dash", line_color="#f4a261", line_width=1.5
    )
    fig.add_vline(
        x="2021-01-01", line_dash="dash", line_color="#e76f51", line_width=1.5
    )
    fig.update_layout(height=350, **PLOT_LAYOUT, yaxis_title="Discharge (m³/s)")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    for name, data, lc, fc in [
        ("Train", train, "#3b9eff", "rgba(59,158,255,.2)"),
        ("Validation", val, "#f4a261", "rgba(244,162,97,.2)"),
        ("Test", test, "#e76f51", "rgba(231,111,81,.2)"),
    ]:
        if data is not None:
            fig2.add_trace(
                go.Violin(
                    y=data["discharge_m3s"],
                    name=name,
                    fillcolor=fc,
                    line_color=lc,
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                )
            )
    fig2.update_layout(height=380, **PLOT_LAYOUT, yaxis_title="Discharge (m³/s)")
    st.plotly_chart(fig2, use_container_width=True)

    stats_data = []
    for name, data in [("Train", train), ("Validation", val), ("Test", test)]:
        if data is not None:
            row = {"Split": name}
            for col in [
                "precip_mm_day",
                "temp_mean_c",
                "swe_mm",
                "snow_cover_pct",
                "discharge_m3s",
            ]:
                if col in data.columns:
                    row[col] = f"{data[col].mean():.3f}"
            stats_data.append(row)
    sdf = pd.DataFrame(stats_data)
    sdf.columns = [
        "Split",
        "Precip (mm/d)",
        "Temp (°C)",
        "SWE (mm)",
        "Snow (%)",
        "Q (m³/s)",
    ]
    st.dataframe(sdf.set_index("Split"), use_container_width=True)

# =============================================================================
# PAGE: MODEL RESULTS
# =============================================================================

elif page == "🤖 Model Results":
    st.markdown(
        "<div class='hero-title' style='font-family:DM Serif Display,serif;font-size:2rem;margin-bottom:.5rem;'>Model Benchmark Results</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:.75rem;color:#4a6a82;margin-bottom:2rem;'>TEST PERIOD 2021–2025 · 16 FEATURES · CHIRPS PRECIPITATION</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='section-header'>Performance Metrics — Test Set</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <table class='results-table'>
    <tr><th>Rank</th><th>Model</th><th>Type</th><th>NSE</th><th>KGE</th><th>RMSE</th><th>Peak Bias</th><th>PBIAS</th></tr>
    <tr><td>1</td><td class='model-pure'>Transformer</td><td>Pure AI</td><td class='best'>0.680</td><td>0.671</td><td>0.254</td><td>−37.0%</td><td>−4.1%</td></tr>
    <tr><td>2</td><td class='model-hybrid'>PI-Transformer</td><td>Hybrid</td><td>0.586</td><td class='best'>0.730</td><td>0.271</td><td class='best'>−29.7%</td><td>+9.4%</td></tr>
    <tr><td>3</td><td class='model-pure'>CNN-LSTM</td><td>Pure AI</td><td>0.559</td><td>0.656</td><td>0.298</td><td>−34.1%</td><td>+13.0%</td></tr>
    <tr><td>4</td><td class='model-hybrid'>PI-LSTM</td><td>Hybrid</td><td>0.512</td><td>0.486</td><td>0.312</td><td>−47.2%</td><td>+2.6%</td></tr>
    <tr><td>5</td><td class='model-pure'>LSTM</td><td>Pure AI</td><td>0.515</td><td>0.423</td><td>0.312</td><td>−51.7%</td><td>+0.1%</td></tr>
    </table>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class='info-box'>
    🏆 Transformer: best NSE (0.680) — best overall predictive accuracy<br>
    🏆 PI-Transformer: best KGE (0.730) and Peak Bias (−29.7%) — best physical consistency<br>
    ⚠️ GloFAS ceiling: LISFLOOD routing smooths karstic pulses → limits max achievable NSE to ~0.68
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='section-header'>Visual Metric Comparison</div>",
        unsafe_allow_html=True,
    )
    models_list = ["LSTM", "CNN-LSTM", "Transformer", "PI-LSTM", "PI-Transformer"]
    nse_vals = [0.515, 0.559, 0.680, 0.512, 0.586]
    kge_vals = [0.423, 0.656, 0.671, 0.486, 0.730]
    peak_vals = [-51.7, -34.1, -37.0, -47.2, -29.7]
    colors_m = ["#3b9eff", "#00b4a0", "#00d4ff", "#f4a261", "#e76f51"]

    metric = st.selectbox("Select metric", ["NSE", "KGE", "Peak Bias (%)"])
    vals = {"NSE": nse_vals, "KGE": kge_vals, "Peak Bias (%)": peak_vals}[metric]
    fig = go.Figure(
        go.Bar(
            x=models_list,
            y=vals,
            marker_color=colors_m,
            marker_opacity=0.85,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(family="DM Mono, monospace", size=11, color="#e8f4f8"),
        )
    )
    if metric in ["NSE", "KGE"]:
        fig.add_hline(
            y=0.75, line_dash="dot", line_color="#00b4a0", annotation_text="Good (0.75)"
        )
    fig.update_layout(height=380, **PLOT_LAYOUT, yaxis_title=metric)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div class='section-header'>Feature Impact — 12 vs 16 Features</div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                name="12 features",
                x=["LSTM", "Transformer", "PI-Transformer"],
                y=[0.518, 0.603, 0.528],
                marker_color="#4a6a82",
                marker_opacity=0.8,
            )
        )
        fig2.add_trace(
            go.Bar(
                name="16 features",
                x=["LSTM", "Transformer", "PI-Transformer"],
                y=[0.515, 0.680, 0.586],
                marker_color="#3b9eff",
                marker_opacity=0.8,
            )
        )
        fig2.update_layout(height=320, barmode="group", title_text="NSE", **PLOT_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = go.Figure()
        fig3.add_trace(
            go.Bar(
                name="12 features",
                x=["LSTM", "Transformer", "PI-Transformer"],
                y=[0.503, 0.632, 0.574],
                marker_color="#4a6a82",
                marker_opacity=0.8,
            )
        )
        fig3.add_trace(
            go.Bar(
                name="16 features",
                x=["LSTM", "Transformer", "PI-Transformer"],
                y=[0.423, 0.671, 0.730],
                marker_color="#a855f7",
                marker_opacity=0.8,
            )
        )
        fig3.update_layout(height=320, barmode="group", title_text="KGE", **PLOT_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        """
    <div class='info-box'>
    ★ Transformer NSE +0.077 · PI-Transformer NSE +0.156 · PI-Transformer KGE +0.156 with 16 vs 12 features<br>
    ★ LSTM shows minimal change — recurrent hidden state already captures antecedent memory implicitly
    </div>
    """,
        unsafe_allow_html=True,
    )

# =============================================================================
# PAGE: CLIMATE SCENARIOS
# =============================================================================

elif page == "🌍 Climate Scenarios":
    st.markdown(
        "<div class='hero-title' style='font-family:DM Serif Display,serif;font-size:2rem;margin-bottom:.5rem;'>Climate Scenario Projections</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:.75rem;color:#4a6a82;margin-bottom:2rem;'>MPI-ESM1-2-HR · NEX-GDDP-CMIP6 · 2015–2100</div>",
        unsafe_allow_html=True,
    )

    if not scenarios:
        st.markdown(
            """
        <div class='warn-box'>
        ⏳ Climate scenario results not yet available — CMIP6 download in progress.<br>
        Run <code>python src/climate_scenarios.py</code> after download completes.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='section-header'>Expected Results (Previous Run)</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
        <table class='results-table'>
        <tr><th>Scenario</th><th>Mean Q 2015–2040</th><th>Mean Q 2075–2100</th><th>Change</th><th>Trend /decade</th><th>p-value</th></tr>
        <tr><td>SSP2-4.5</td><td>~0.85 m³/s</td><td>~0.82 m³/s</td><td class='model-hybrid'>−3.3%</td><td>−0.0031 m³/s</td><td>&lt;0.05 ✓</td></tr>
        <tr><td>SSP5-8.5</td><td>~0.85 m³/s</td><td>~0.78 m³/s</td><td style='color:#e76f51'>−8.5%</td><td>−0.0080 m³/s</td><td>&lt;0.05 ✓</td></tr>
        </table>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class='info-box'>
        📊 PI ensemble projects 48% steeper SSP5-8.5 decline than pure AI ensemble<br>
        📊 All 5 models agree: discharge declining under both scenarios<br>
        📊 Seasonal shift: spring snowmelt peak moving 2–4 weeks earlier by 2100<br>
        📊 PI-Transformer most physically consistent projector due to explicit ET constraint
        </div>
        """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            "<div class='section-header'>Annual Discharge Projections</div>",
            unsafe_allow_html=True,
        )
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=[
                "SSP2-4.5 — Moderate Emissions",
                "SSP5-8.5 — High Emissions",
            ],
            vertical_spacing=0.12,
        )
        model_colors = {
            "Q_LSTM": "#3b9eff",
            "Q_CNN-LSTM": "#00b4a0",
            "Q_Transformer": "#00d4ff",
            "Q_PI-LSTM": "#f4a261",
            "Q_PI-Transformer": "#e76f51",
            "Q_ensemble_all": "#ffffff",
        }
        for row_i, (scen, df_s) in enumerate(scenarios.items(), 1):
            df_s = df_s.copy()
            df_s["year"] = df_s["date"].dt.year
            annual = df_s.groupby("year").mean(numeric_only=True)
            for col, color in model_colors.items():
                if col not in annual or annual[col].isna().all():
                    continue
                sm = annual[col].rolling(5, center=True, min_periods=1).mean()
                fig.add_trace(
                    go.Scatter(
                        x=annual.index,
                        y=sm,
                        line=dict(
                            color=color, width=2.5 if col == "Q_ensemble_all" else 1.2
                        ),
                        name=col.replace("Q_", ""),
                        showlegend=(row_i == 1),
                    ),
                    row=row_i,
                    col=1,
                )
            fig.add_vline(
                x=2025, line_dash="dot", line_color="#4a6a82", row=row_i, col=1
            )
        fig.update_layout(height=600, **PLOT_LAYOUT)
        fig.update_yaxes(title_text="Annual Mean Discharge (m³/s)", gridcolor="#1e3448")
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: VARIABLE INSPECTOR
# =============================================================================

elif page == "🔬 Variable Inspector":
    if df is None:
        st.error("Master dataset not found.")
        st.stop()

    st.markdown(
        "<div class='hero-title' style='font-family:DM Serif Display,serif;font-size:2rem;margin-bottom:.5rem;'>Variable Inspector</div>",
        unsafe_allow_html=True,
    )

    var_map = {
        k: v
        for k, v in {
            "Discharge (m³/s)": "discharge_m3s",
            "Precipitation (mm/d)": "precip_mm_day",
            "Temperature (°C)": "temp_mean_c",
            "Temp max (°C)": "temp_max_c",
            "Temp min (°C)": "temp_min_c",
            "SWE (mm)": "swe_mm",
            "Snow Cover (%)": "snow_cover_pct",
            "SWE Delta": "swe_delta",
            "Precip 3-day": "precip_3day",
            "Precip 7-day": "precip_7day",
            "Soil Moisture (mm)": "soil_moisture_mm",
            "SM 7-day Mean": "sm_7day_mean",
            "SM Anomaly": "sm_anomaly",
            "PET (mm/day)": "pet_mm_day",
        }.items()
        if v in df.columns
    }

    selected = st.selectbox("Select variable", list(var_map.keys()))
    col = var_map[selected]
    series = df[col].dropna()

    c1, c2, c3, c4, c5 = st.columns(5)
    for container, label, val in [
        (c1, "Mean", f"{series.mean():.3f}"),
        (c2, "Std", f"{series.std():.3f}"),
        (c3, "Min", f"{series.min():.3f}"),
        (c4, "Max", f"{series.max():.3f}"),
        (c5, "Median", f"{series.median():.3f}"),
    ]:
        container.markdown(
            f"<div class='metric-card' style='margin-bottom:1rem;'><div class='metric-label'>{label}</div><div style='font-family:DM Serif Display,serif;font-size:1.5rem;color:#00d4ff;'>{val}</div></div>",
            unsafe_allow_html=True,
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=("Time Series", "Distribution"),
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[col],
            line=dict(color="#3b9eff", width=0.8),
            fill="tozeroy",
            fillcolor="rgba(59,158,255,.1)",
            name=selected,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            y=series, nbinsy=60, marker_color="#3b9eff", marker_opacity=0.7, name="Dist"
        ),
        row=1,
        col=2,
    )
    fig.update_layout(height=380, showlegend=False, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    df["year"] = df["date"].dt.year
    annual = df.groupby("year")[col].mean()
    z = np.polyfit(annual.index, annual.values, 1)
    fig3 = go.Figure()
    fig3.add_trace(
        go.Bar(
            x=annual.index,
            y=annual.values,
            marker_color="#3b9eff",
            marker_opacity=0.7,
            name="Annual mean",
        )
    )
    fig3.add_trace(
        go.Scatter(
            x=annual.index,
            y=np.poly1d(z)(annual.index),
            line=dict(color="#e76f51", width=2, dash="dash"),
            name=f"Trend ({z[0]:+.4f}/yr)",
        )
    )
    fig3.update_layout(height=320, **PLOT_LAYOUT, yaxis_title=selected)
    st.plotly_chart(fig3, use_container_width=True)

# =============================================================================
# PAGE: PIPELINE SUMMARY
# =============================================================================

elif page == "📋 Pipeline Summary":
    st.markdown(
        "<div class='hero-title' style='font-family:DM Serif Display,serif;font-size:2rem;margin-bottom:.5rem;'>Pipeline Summary</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:.75rem;color:#4a6a82;margin-bottom:2rem;'>DATA ENGINEERING · MODELLING · STATUS</div>",
        unsafe_allow_html=True,
    )

    for icon, title, desc in [
        (
            "✅",
            "Data Collection",
            "CHIRPS v2.0 (GEE, gauge-corrected) · MERRA-2 temperature · GLDAS Noah SWE + soil moisture 0–10 cm · MODIS MOD10A1 snow cover · GloFAS ERA5 discharge · CMIP6 MPI-ESM1-2-HR via GEE (downloading).",
        ),
        (
            "✅",
            "Input Data Validation",
            "MERRA-2 vs Meteostat WMO 40100: temp r=0.975 daily / r=0.994 monthly / bias=−2.4°C. CHIRPS vs airport: r=0.513 daily / r=0.653 monthly / PBIAS=−6.2%.",
        ),
        (
            "✅",
            "Preprocessing",
            "Gap filling (linear interpolation ≤3 days). Min-max normalisation on training set only. 30-day lookback windowing. Shape: (samples, 30, 16).",
        ),
        (
            "✅",
            "MODIS Processing",
            "27,249 GeoTIFFs clipped to watershed polygon via rasterio.mask. NDSI ≥ 40. 19,984 duplicates removed → 7,265 unique days.",
        ),
        (
            "✅",
            "GloFAS Extraction",
            "312 yearly GRIB files. dis24 at outlet 34.062°N, 35.642°E. Q range: 0.227–7.312 m³/s. 9,497 daily rows.",
        ),
        (
            "✅",
            "Feature Engineering",
            "16 features: precip (1/3/7-day) · temp (mean/min/max/range) · SWE + delta · snow cover · month sin/cos · soil moisture + 7-day mean + anomaly · Hamon PET.",
        ),
        (
            "✅",
            "Train/Val/Test Split",
            "Train 2000–2017 (~6,515) · Val 2018–2020 (1,066) · Test 2021–2025 (1,796). Chronological — no shuffling.",
        ),
        (
            "✅",
            "Model Training",
            "LSTM (125K) · CNN-LSTM (143K) · Transformer (108K) · PI-LSTM · PI-Transformer. PI loss: L = MSE + 0.05·(P−ET−Q−ΔS)².",
        ),
        (
            "✅",
            "Evaluation",
            "Best NSE=0.680 (Transformer) · Best KGE=0.730 (PI-Transformer) · Best Peak Bias=−29.7% (PI-Transformer).",
        ),
        (
            "⏳",
            "Climate Scenarios",
            "CMIP6 download via GEE in progress — 688 files (pr/tas/tasmin/tasmax · ssp245+ssp585 · 2015–2100). Run climate_scenarios.py after.",
        ),
        (
            "🔜",
            "Thesis Writing",
            "Chapter 4 Results · Chapter 5 Discussion · Chapter 6 Conclusion. Chapters 1–3 complete.",
        ),
    ]:
        color = "#00b4a0" if icon == "✅" else "#f4a261" if icon == "⏳" else "#4a6a82"
        st.markdown(
            f"""
        <div class='pipeline-step'>
            <div style='font-size:1.3rem;min-width:32px;text-align:center;'>{icon}</div>
            <div><div class='step-title' style='color:{color};'>{title}</div><div class='step-desc'>{desc}</div></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-header'>Key Files</div>", unsafe_allow_html=True)
    for folder, contents in [
        ("data/raw/chirps/", "chirps_nahr_ibrahim_2000_2025_daily.csv"),
        (
            "data/raw/cmip6/",
            "ssp245/{pr,tas,tasmin,tasmax}/{year}.csv · ssp585/... (⏳ downloading)",
        ),
        ("data/master/", "nahr_ibrahim_master_full.csv (9,497 rows × 17 cols)"),
        (
            "data/splits/",
            "train_raw.csv · val_raw.csv · test_raw.csv · scaler_params.csv",
        ),
        (
            "data/sequences/",
            "X_train.npy (6515,30,16) · X_val.npy · X_test.npy · y_*.npy",
        ),
        (
            "models/trained/",
            "lstm_final.keras · cnn_lstm_final.keras · transformer_final.keras · pi_lstm_final.keras · pi_transformer_final.keras",
        ),
        (
            "results/metrics/",
            "pi_models_metrics.csv · climate_scenario_trends_full.csv",
        ),
        (
            "results/figures/",
            "lstm_results.png · cnn_lstm_results.png · transformer_results.png · pi_models_results.png · scenario_*.png",
        ),
        (
            "qgis/",
            "nahr_ibrahim_watershed_utm37n.shp · dem_nahr_ibrahim_clipped.tif · slope_nahr_ibrahim.tif · hillshade_nahr_ibrahim.tif · key_locations.csv",
        ),
    ]:
        st.markdown(
            f"""
        <div style='border-left:2px solid #2a4d6e;padding:.6rem 1rem;margin-bottom:.8rem;'>
            <div style='font-family:DM Mono,monospace;font-size:.75rem;color:#3b9eff;'>{folder}</div>
            <div style='font-size:.78rem;color:#8aafc4;margin-top:.3rem;'>{contents}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
