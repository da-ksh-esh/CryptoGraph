"""Component: Cybersecurity & Market Data Anomaly Detection with Temporal GAE."""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


@st.cache_data
def load_tgae_anomalies():
    csv_path = Path("reports/tgae_anomalies.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def render_anomaly_alerts():
    st.subheader("🚨 Cybersecurity & Structural Anomaly Detection (Temporal GAE)")
    st.markdown(
        "Monitors structural deviations and market microstructure distortions using the **Temporal Graph Autoencoder (T-GAE)**. "
        "Flags market dislocations where reconstruction error exceeds baseline parametric thresholds."
    )

    df_anom = load_tgae_anomalies()

    if df_anom.empty:
        st.info("No anomaly records found in `reports/tgae_anomalies.csv`.")
        return

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Monitored Events", len(df_anom))
    with col2:
        baseline_err = df_anom["baseline_reconstruction_error"].iloc[0]
        st.metric("Baseline Mean Rec. Error", f"{baseline_err:.4f}")
    with col3:
        max_ratio = df_anom["anomaly_ratio"].max()
        st.metric("Max Anomaly Ratio", f"{max_ratio:.2f}x")
    with col4:
        anom_count = int(df_anom["is_statistical_anomaly"].sum())
        st.metric("Statistical Anomalies (p < 0.05)", anom_count)

    # Plot Comparison: Baseline vs Event Reconstruction Error
    st.write("#### 📊 Event Reconstruction Error vs Baseline Distribution")
    fig = go.Figure()

    events = df_anom["event_name"].tolist()
    event_errors = df_anom["event_reconstruction_error"].tolist()
    base_errors = df_anom["baseline_reconstruction_error"].tolist()
    base_stds = df_anom["baseline_std"].tolist()

    fig.add_trace(go.Bar(
        name="Event Reconstruction Error",
        x=events,
        y=event_errors,
        marker_color="crimson"
    ))

    fig.add_trace(go.Bar(
        name="Baseline Reference Error",
        x=events,
        y=base_errors,
        error_y=dict(type="data", array=base_stds, visible=True),
        marker_color="steelblue"
    ))

    fig.update_layout(
        barmode="group",
        title="T-GAE Reconstruction Error Comparison (Normed L2 + Adjacency Loss)",
        yaxis_title="Reconstruction Error",
        xaxis_title="Event",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interactive Table of Anomaly Events
    st.write("#### 📋 Detailed Event Dislocation Audit")
    styled_df = df_anom[[
        "event_name", "event_date", "event_reconstruction_error",
        "baseline_reconstruction_error", "anomaly_ratio", "z_score",
        "p_value", "most_anomalous_asset", "is_statistical_anomaly"
    ]].copy()

    styled_df["status"] = styled_df["is_statistical_anomaly"].map(
        lambda x: "🚨 ANOMALY" if x else "🟢 NORMAL"
    )

    st.dataframe(
        styled_df.style.format({
            "event_reconstruction_error": "{:.4f}",
            "baseline_reconstruction_error": "{:.4f}",
            "anomaly_ratio": "{:.2f}x",
            "z_score": "{:+.2f}",
            "p_value": "{:.4f}"
        }),
        use_container_width=True
    )
