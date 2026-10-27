"""
Courier Delivery Performance Analytics - Streamlit Dashboard
---------------------------------------------------------------
Run: streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Courier Delivery Analytics", layout="wide", page_icon="🚚")

# ---------- Custom Styling ----------
st.markdown("""
<style>
    /* Overall app background - richer layered blue with subtle radial glow */
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(79,172,254,0.12) 0%, transparent 45%),
            radial-gradient(circle at 85% 90%, rgba(0,242,254,0.08) 0%, transparent 45%),
            linear-gradient(180deg, #0a0e1a 0%, #131a2e 45%, #0f1626 100%);
    }

    /* Top toolbar/header - make it blend instead of showing white */
    header[data-testid="stHeader"] {
        background: #0a0e1a !important;
        border-bottom: 1px solid #26305090;
    }
    header[data-testid="stHeader"] * {
        color: #e2e6f0 !important;
    }

    div[data-testid="stToolbar"] {
        background: #0a0e1a !important;
    }
    div[data-testid="stDecoration"] {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
    }

    .block-container {
        padding-top: 2rem;
    }

    /* Sidebar - deeper, richer gradient with glow border */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1220 0%, #161d33 55%, #10182c 100%);
        border-right: 1px solid #2f3a5c;
        box-shadow: 4px 0 24px rgba(0,0,0,0.4);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e6f0 !important;
    }

    /* Sidebar logo/brand block */
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 6px 4px 22px 4px;
        margin-bottom: 14px;
        border-bottom: 1px solid #2a3352;
    }
    .sidebar-logo-icon {
        width: 46px;
        height: 46px;
        border-radius: 12px;
        background: linear-gradient(135deg, #4facfe 0%, #00c2fe 50%, #0072ff 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 20px rgba(79,172,254,0.5);
        flex-shrink: 0;
    }
    .sidebar-logo-text {
        display: flex;
        flex-direction: column;
        line-height: 1.15;
    }
    .sidebar-logo-title {
        font-weight: 800;
        font-size: 16px;
        color: #f0f4ff !important;
        letter-spacing: 0.3px;
    }
    .sidebar-logo-sub {
        font-size: 11.5px;
        color: #7c88ad !important;
        font-weight: 500;
    }

    /* Main title */
    h1 {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        padding-bottom: 10px;
    }

    h2, h3 {
        color: #e2e6f0 !important;
        font-weight: 700;
    }

    p, span, label, .stMarkdown, .stCaption {
        color: #b8c0d9 !important;
    }

    /* Metric cards - richer with glow on hover */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a2138, #212a48);
        border: 1px solid #33406b;
        border-radius: 16px;
        padding: 18px 12px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 24px rgba(79,172,254,0.25), inset 0 1px 0 rgba(255,255,255,0.06);
        transform: translateY(-2px);
    }
    div[data-testid="stMetricLabel"] {
        color: #8b93b8 !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #5fb8ff !important;
        font-weight: 800;
        text-shadow: 0 0 18px rgba(79,172,254,0.35);
    }

    /* Dataframes / tables - force dark theme instead of white */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #33406b;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    div[data-testid="stDataFrame"] * {
        color: #e2e6f0 !important;
    }
    div[data-testid="stDataFrame"] > div {
        background-color: #171e33 !important;
    }
    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stDataFrame"] [role="gridcell"] {
        background-color: #171e33 !important;
        color: #e2e6f0 !important;
    }
    div[data-testid="stDataFrame"] canvas {
        filter: invert(0.92) hue-rotate(180deg);
        border-radius: 8px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: #0f1420 !important;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        box-shadow: 0 4px 16px rgba(79,172,254,0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button p {
        color: #0f1420 !important;
        font-weight: 700;
    }
    .stButton > button:hover {
        transform: scale(1.03);
        color: #0f1420 !important;
        box-shadow: 0 6px 22px rgba(79,172,254,0.5);
    }

    /* Sliders/selectbox containers */
    div[data-baseweb="select"] > div, .stSlider {
        background-color: transparent;
    }

    /* Radio buttons in sidebar (nav) - richer active state */
    div[role="radiogroup"] label {
        background: #182036;
        border: 1px solid #263053;
        border-radius: 10px;
        padding: 9px 12px;
        margin-bottom: 6px;
        transition: background 0.15s ease, border 0.15s ease;
    }
    div[role="radiogroup"] label:hover {
        background: #232c4d;
        border: 1px solid #3a4a7a;
    }
    div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(90deg, rgba(79,172,254,0.18), rgba(0,242,254,0.08));
        border: 1px solid #4facfe;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Data & Model Loading (cached) ----------
@st.cache_data
def load_data():
    df = pd.read_csv(
        os.path.join(SCRIPT_DIR, "courier_deliveries.csv"),
        parse_dates=["pickup_time", "expected_delivery_time", "actual_delivery_time"],
    )
    df["pickup_hour"] = df["pickup_time"].dt.hour
    return df


@st.cache_resource
def load_models():
    models = {}
    try:
        models["clf"] = joblib.load(os.path.join(SCRIPT_DIR, "delay_classifier.pkl"))
        models["reg"] = joblib.load(os.path.join(SCRIPT_DIR, "delay_regressor.pkl"))
        models["anomaly"] = joblib.load(os.path.join(SCRIPT_DIR, "anomaly_detector.pkl"))
        models["le_traffic"] = joblib.load(os.path.join(SCRIPT_DIR, "le_traffic.pkl"))
        models["le_weather"] = joblib.load(os.path.join(SCRIPT_DIR, "le_weather.pkl"))
    except FileNotFoundError:
        models = None
    return models


df = load_data()
models = load_models()


def style_chart(fig):
    """Apply consistent dark theme styling to all plotly charts."""
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e6f0"),
        title_font=dict(color="#e2e6f0", size=18),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e6f0")),
        margin=dict(t=50, l=10, r=10, b=10),
    )
    fig.update_xaxes(gridcolor="#2a3352", zerolinecolor="#2a3352")
    fig.update_yaxes(gridcolor="#2a3352", zerolinecolor="#2a3352")
    fig.update_traces(selector=dict(type="pie"), textfont=dict(color="#0f1420"))
    return fig

st.title("🚚 AI-Powered Courier Delivery Performance Analytics")

# ---------- Sidebar Navigation ----------
st.sidebar.markdown("""
<div class="sidebar-logo">
    <div class="sidebar-logo-icon">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 7h11v7H3z" fill="#0f1420" opacity="0.9"/>
            <path d="M14 10h4l3 3v1h-7z" fill="#0f1420" opacity="0.65"/>
            <circle cx="7" cy="17" r="1.8" fill="#0f1420"/>
            <circle cx="17.5" cy="17" r="1.8" fill="#0f1420"/>
        </svg>
    </div>
    <div class="sidebar-logo-text">
        <span class="sidebar-logo-title">CourierIQ</span>
        <span class="sidebar-logo-sub">Delivery Analytics</span>
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Agent Performance", "Delay Analysis", "Predict New Order", "Anomaly Alerts"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df)} orders loaded")


# ---------- PAGE 1: OVERVIEW ----------
if page == "Overview":
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, rgba(79,172,254,0.15), rgba(0,242,254,0.06));
        border: 1px solid #2f3a5c;
        border-radius: 14px;
        padding: 14px 20px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 12px;">
        <span style="font-size: 22px;">📦</span>
        <span style="color: #b8c0d9; font-size: 14.5px;">
            Live snapshot of delivery performance across all agents, routes, and conditions.
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.header("Overview")

    total_orders = len(df)
    on_time = (df["delivery_status"] == "On-Time").sum()
    delayed = (df["delivery_status"] == "Delayed").sum()
    failed = (df["delivery_status"] == "Failed").sum()
    on_time_rate = round((on_time / total_orders) * 100, 1)
    avg_delay = round(df.loc[df["delivery_status"] == "Delayed", "delay_minutes"].mean(), 1)
    avg_rating = round(df["customer_rating"].mean(), 2)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Orders", total_orders)
    c2.metric("On-Time Rate", f"{on_time_rate}%")
    c3.metric("Delayed", delayed)
    c4.metric("Avg Delay", f"{avg_delay} min")
    c5.metric("Avg Rating", f"{avg_rating} / 5")

    col1, col2 = st.columns(2)
    with col1:
        status_counts = df["delivery_status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig = px.pie(status_counts, names="status", values="count", title="Delivery Status Breakdown",
                     color="status", color_discrete_map={"On-Time": "#2ecc71", "Delayed": "#e74c3c", "Failed": "#95a5a6"})
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with col2:
        daily = df.groupby(df["pickup_time"].dt.date)["delay_minutes"].mean().reset_index()
        daily.columns = ["date", "avg_delay"]
        fig2 = px.line(daily, x="date", y="avg_delay", title="Average Delay Trend Over Time")
        st.plotly_chart(style_chart(fig2), use_container_width=True)


# ---------- PAGE 2: AGENT PERFORMANCE ----------
elif page == "Agent Performance":
    st.header("Agent Performance")

    grouped = df.groupby("agent_name").agg(
        total_orders=("order_id", "count"),
        on_time_orders=("delivery_status", lambda x: (x == "On-Time").sum()),
        avg_delay=("delay_minutes", "mean"),
        avg_rating=("customer_rating", "mean"),
    ).reset_index()
    grouped["on_time_rate_%"] = round((grouped["on_time_orders"] / grouped["total_orders"]) * 100, 2)
    grouped["avg_delay"] = grouped["avg_delay"].round(2)
    grouped["avg_rating"] = grouped["avg_rating"].round(2)
    grouped["performance_score"] = round(
        (grouped["on_time_rate_%"] * 0.5) + (grouped["avg_rating"] * 10 * 0.3) - (grouped["avg_delay"].fillna(0) * 0.2), 2
    )
    ranked = grouped.sort_values("performance_score", ascending=False).reset_index(drop=True)

    fig = px.bar(ranked.head(10), x="agent_name", y="performance_score", title="Top 10 Agents by Performance Score",
                 color="performance_score", color_continuous_scale="Blues")
    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.subheader("Full Agent Leaderboard")
    st.dataframe(ranked, use_container_width=True)


# ---------- PAGE 3: DELAY ANALYSIS ----------
elif page == "Delay Analysis":
    st.header("Delay Analysis")

    col1, col2 = st.columns(2)
    with col1:
        traffic_delay = df.groupby("traffic_condition")["delay_minutes"].mean().reset_index()
        fig = px.bar(traffic_delay, x="traffic_condition", y="delay_minutes", title="Avg Delay by Traffic Condition",
                     color="delay_minutes", color_continuous_scale="Oranges")
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with col2:
        weather_delay = df.groupby("weather")["delay_minutes"].mean().reset_index()
        fig2 = px.bar(weather_delay, x="weather", y="delay_minutes", title="Avg Delay by Weather",
                      color="delay_minutes", color_continuous_scale="Blues")
        st.plotly_chart(style_chart(fig2), use_container_width=True)

    hourly_delay = df.groupby("pickup_hour")["delay_minutes"].mean().reset_index()
    fig3 = px.line(hourly_delay, x="pickup_hour", y="delay_minutes", title="Avg Delay by Pickup Hour", markers=True)
    st.plotly_chart(style_chart(fig3), use_container_width=True)

    st.subheader("Delay Distribution")
    fig4 = px.histogram(df[df["delivery_status"] == "Delayed"], x="delay_minutes", nbins=40,
                         title="Distribution of Delay Minutes (Delayed Orders Only)")
    st.plotly_chart(style_chart(fig4), use_container_width=True)


# ---------- PAGE 4: PREDICT NEW ORDER ----------
elif page == "Predict New Order":
    st.header("Predict New Order Delay")

    if models is None:
        st.error("Model files not found. Please run `ml_models.py` first to generate the .pkl files.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            distance = st.slider("Distance (km)", 1.0, 30.0, 10.0)
            traffic = st.selectbox("Traffic Condition", options=list(models["le_traffic"].classes_))
            weather = st.selectbox("Weather", options=list(models["le_weather"].classes_))
        with col2:
            pickup_hour = st.slider("Pickup Hour (24h)", 0, 23, 12)
            agent_id = st.number_input("Agent ID", min_value=1, max_value=25, value=1)

        if st.button("Predict", type="primary"):
            traffic_encoded = models["le_traffic"].transform([traffic])[0]
            weather_encoded = models["le_weather"].transform([weather])[0]

            features = pd.DataFrame([{
                "distance_km": distance,
                "traffic_encoded": traffic_encoded,
                "weather_encoded": weather_encoded,
                "pickup_hour": pickup_hour,
                "agent_id": agent_id,
            }])

            is_delayed_pred = models["clf"].predict(features)[0]
            delay_proba = models["clf"].predict_proba(features)[0][1]

            st.markdown("---")
            if is_delayed_pred == 1:
                st.error(f"⚠️ Prediction: DELAYED (confidence: {delay_proba*100:.1f}%)")
                est_delay = models["reg"].predict(features)[0]
                st.metric("Estimated Delay", f"{est_delay:.1f} minutes")
            else:
                st.success(f"✅ Prediction: ON-TIME (confidence: {(1-delay_proba)*100:.1f}%)")


# ---------- PAGE 5: ANOMALY ALERTS ----------
elif page == "Anomaly Alerts":
    st.header("Anomaly Alerts")
    st.caption("Deliveries flagged as statistical outliers - unusually fast or unusually slow relative to distance, traffic, and weather. These are candidates for manual review, not confirmed fraud.")

    if models is None:
        st.error("Model files not found. Please run `ml_models.py` first.")
    else:
        data = df[df["delivery_status"] != "Failed"].copy()
        duration_minutes = (data["actual_delivery_time"] - data["pickup_time"]).dt.total_seconds() / 60
        data["speed_kmpm"] = data["distance_km"] / duration_minutes.replace(0, 0.1)
        data["traffic_encoded"] = models["le_traffic"].transform(data["traffic_condition"])
        data["weather_encoded"] = models["le_weather"].transform(data["weather"])

        features = data[["distance_km", "traffic_encoded", "weather_encoded", "speed_kmpm"]]
        data["anomaly_flag"] = models["anomaly"].predict(features)

        anomalies = data[data["anomaly_flag"] == -1]
        st.metric("Flagged Deliveries", f"{len(anomalies)} / {len(data)} ({len(anomalies)/len(data)*100:.1f}%)")

        st.dataframe(
            anomalies[["order_id", "agent_name", "distance_km", "traffic_condition", "weather", "speed_kmpm", "delivery_status"]],
            use_container_width=True,
        )
