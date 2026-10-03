"""Component: Live Time-Indexed Contagion Prediction & Interactive Scenario Analysis."""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch

from cryptograph.pipelines.predict import predict_contagion, find_latest_checkpoint
from cryptograph.data.universe import TARGET_UNIVERSE_30


@st.cache_data
def get_dataset_metadata():
    dataset_path = Path("data/processed/v2/1h/processed_dataset.pt")
    if not dataset_path.exists():
        return [], 0, []
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    timestamps = data["timestamps"]
    horizons = data.get("horizons", [1, 3, 6, 12, 24])
    assets = data.get("asset_names", TARGET_UNIVERSE_30)
    return timestamps, len(timestamps), horizons, assets


@st.cache_data(show_spinner=False)
def run_live_inference(
    model_type: str,
    timestep: int,
    horizon_hours: int,
    threshold: float,
    top_k: int,
    explain: bool
):
    return predict_contagion(
        model_type=model_type,
        timestep=timestep,
        horizon_hours=horizon_hours,
        threshold=threshold,
        top_k=top_k,
        explain=explain,
        device_name="auto"
    )


def render_live_prediction():
    st.subheader("🔮 Time-Indexed Contagion Risk Simulator & Scenario Replay Engine")
    st.markdown(
        "Executes model inference using trained **CryptoTGN** (Continuous Dynamic GNN) or **CryptoLSTM** (Sequential RNN) "
        "across chronological hourly snapshots of the audited test split. "
        "Allows dynamic scrubbing across timesteps to simulate emerging shock cascades, "
        "estimate per-token dynamic persistence decay, and inspect TempME explanatory transmission subgraphs."
    )
    st.info(
        "ℹ️ **Inference & Simulation Mode:** This engine performs deep neural forward inference over the "
        "held-out test split of the historical hourly dataset (2024–2026). It operates as an offline research replay and "
        "crisis simulator rather than a live streaming Binance websocket feed. Persistence half-lives are estimated dynamically "
        "per token via trailing AR(1) autocorrelation decay alongside calibrated sector baseline priors."
    )

    meta_res = get_dataset_metadata()
    if len(meta_res) < 4:
        st.error("Processed dataset not found at `data/processed/v2/1h/processed_dataset.pt`.")
        return

    timestamps, num_steps, horizons, assets = meta_res

    # Control Bar
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        model_choice = st.selectbox("Model Architecture", ["tgn", "lstm"], index=0, format_func=lambda x: "CryptoTGN (Dynamic GNN)" if x == "tgn" else "CryptoLSTM (Baseline RNN)")
    with col2:
        horizon_choice = st.selectbox("Forecast Horizon", horizons, index=0, format_func=lambda h: f"+{h} Hour{'s' if h > 1 else ''}")
    with col3:
        threshold_val = st.slider("Alert Probability Threshold", 0.10, 0.90, 0.50, 0.05)
    with col4:
        top_k_val = st.selectbox("Display Top-K Assets", [5, 10, 15, 20, 30], index=1)

    # Timestep Scrubbing
    st.write("#### ⏱️ Chronological Timestep Scrubbing")
    c_time1, c_time2 = st.columns([3, 1])

    with c_time2:
        preset = st.selectbox(
            "Crisis Presets",
            [
                "Latest Available",
                "Yen Carry-Trade Unwind (2024-08-05)",
                "Oct-10 Liquidation Cascade (2025-10-10)",
                "ETF Approval (2024-01-10)"
            ],
            index=0
        )

    if preset == "Latest Available":
        default_t = num_steps - 1
    elif "2024-08-05" in preset:
        default_t = 5208 if 5208 < num_steps else num_steps - 1
    elif "2025-10-10" in preset:
        default_t = 15552 if 15552 < num_steps else num_steps - 1
    elif "2024-01-10" in preset:
        default_t = 216 if 216 < num_steps else num_steps - 1
    else:
        default_t = num_steps - 1

    with c_time1:
        selected_timestep = st.slider(
            "Timestep Index (t)",
            min_value=168,
            max_value=num_steps - 1,
            value=default_t,
            help="Scrub backward or forward through time to inspect dynamic predictions."
        )

    current_ts_str = timestamps[selected_timestep]
    st.info(f"Target Timestamp: **{current_ts_str}** (Index `t={selected_timestep}`) | Selected Horizon: **+{horizon_choice}h**")

    # Run Live Prediction
    with st.spinner("Computing real-time contagion probabilities..."):
        try:
            results = run_live_inference(
                model_type=model_choice,
                timestep=selected_timestep,
                horizon_hours=horizon_choice,
                threshold=threshold_val,
                top_k=top_k_val,
                explain=(model_choice == "tgn")
            )
        except Exception as e:
            st.error(f"Inference failed: {e}")
            return

    meta = results["metadata"]
    preds = results["predictions"]

    # Market Regime Banner
    regime = meta["market_risk_regime"]
    if "HIGH" in regime:
        st.error(f"🚨 **SYSTEMIC REGIME: {regime}** — {meta['assets_in_alert']} of {meta['total_assets_evaluated']} assets in ALERT status!")
    elif "ELEVATED" in regime:
        st.warning(f"⚠️ **REGIME: {regime}** — {meta['assets_in_alert']} of {meta['total_assets_evaluated']} assets in ALERT status.")
    else:
        st.success(f"🟢 **REGIME: {regime}** — Market calm. {meta['assets_in_alert']} assets above alert threshold.")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Mean Market Shock Risk", f"{meta['market_mean_shock_probability']:.2%}")
    with m2:
        st.metric("Assets in Alert", f"{meta['assets_in_alert']} / {meta['total_assets_evaluated']}")
    with m3:
        st.metric("Forecast Horizon", f"+{meta['horizon_hours']}h")
    with m4:
        st.metric("Engine Device", meta["device"])

    # Probability Distribution Plot
    df_preds = pd.DataFrame(preds)

    st.write("#### 📊 Real-Time Predicted Shock Probabilities P(shock(t+h))")
    colors = ["#e74c3c" if s == "ALERT" else "#3498db" for s in df_preds["status"]]

    fig_probs = go.Figure()
    fig_probs.add_trace(go.Bar(
        x=df_preds["asset"],
        y=df_preds["shock_probability"],
        marker_color=colors,
        text=df_preds["shock_probability"].map(lambda p: f"{p:.1%}"),
        textposition="outside"
    ))
    fig_probs.add_hline(
        y=threshold_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Alert Threshold ({threshold_val:.2f})",
        annotation_position="top right"
    )
    fig_probs.update_layout(
        yaxis_title="Predicted Shock Probability",
        xaxis_title="Cryptocurrency Asset",
        yaxis_range=[0.0, 1.05],
        height=400,
        margin=dict(t=30, b=30, l=40, r=40)
    )
    st.plotly_chart(fig_probs, use_container_width=True)

    # Detailed Predictions Table
    st.write("#### 📋 Contagion Risk Rankings & Spatio-Temporal Channels")
    
    # Rename columns for clarity
    df_table = df_preds.rename(columns={
        "rank": "Rank",
        "asset": "Asset",
        "sector": "Sector",
        "price_usdt": "Price (USDT)",
        "shock_probability": "Shock Probability",
        "status": "Alert Status",
        "persistence_halflife_hours": "Dynamic Decay t_1/2",
        "persistence_mode": "Estimation Mode",
        "sector_prior_halflife": "Sector Baseline t_1/2",
        "explanation": "Explanatory Attributions"
    })
    
    st.dataframe(
        df_table[[
            "Rank", "Asset", "Sector", "Price (USDT)",
            "Shock Probability", "Alert Status", "Dynamic Decay t_1/2",
            "Estimation Mode", "Sector Baseline t_1/2", "Explanatory Attributions"
        ]].style.format({
            "Price (USDT)": "${:,.2f}",
            "Shock Probability": "{:.2%}",
            "Dynamic Decay t_1/2": "{:.1f}h",
            "Sector Baseline t_1/2": "{:.1f}h"
        }),
        use_container_width=True
    )
