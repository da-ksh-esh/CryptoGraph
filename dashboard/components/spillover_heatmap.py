"""Component: Cross-Asset Spillover Transmission Matrix and TempME Explanations."""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_persistence_data():
    csv_path = Path("reports/persistence_analysis.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_data
def load_tempme_explanations():
    csv_path = Path("reports/tempme_explanations.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def render_spillover_heatmap():
    st.subheader("🔗 Cross-Asset Shock Propagation & Explanatory Channels")
    st.markdown(
        "Empirical analysis of historical cryptocurrency shock contagion events, "
        "tracing transmission delays, blast radii, and TempME explanatory channels."
    )

    df_persist = load_persistence_data()
    df_tempme = load_tempme_explanations()

    if not df_persist.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Analyzed Events", len(df_persist))
        with col2:
            st.metric("Avg Blast Radius", f"{df_persist['blast_radius_pct'].mean() * 100:.1f}%")
        with col3:
            st.metric("Avg Propagation Delay", f"{df_persist['propagation_delay_hours'].mean():.1f}h")
        with col4:
            st.metric("Avg Half-Life (t₁/₂)", f"{df_persist['mean_persistence_half_life_hours'].mean():.1f}h")

        # Blast Radius vs Propagation Delay Plot
        st.write("#### 💥 Blast Radius vs Propagation Delay Across Historical Crises")
        fig_events = px.scatter(
            df_persist,
            x="propagation_delay_hours",
            y="blast_radius_pct",
            size="mean_persistence_half_life_hours",
            color="most_impacted_sector",
            hover_name="event_name",
            text="event_name",
            labels={
                "propagation_delay_hours": "Propagation Delay to Peak (Hours)",
                "blast_radius_pct": "Blast Radius (% of Universe Shocked)",
                "mean_persistence_half_life_hours": "Half-Life (h)"
            },
            title="Crisis Contagion Velocity and Impact Scope"
        )
        fig_events.update_traces(textposition="top center")
        fig_events.update_layout(height=450, yaxis_tickformat=".0%")
        st.plotly_chart(fig_events, use_container_width=True)

        # Historical Events Table
        st.write("#### 📜 Event Propagation Log")
        st.dataframe(
            df_persist[[
                "event_name", "timestamp", "initial_transmitters",
                "blast_radius_count", "blast_radius_pct",
                "propagation_delay_hours", "mean_persistence_half_life_hours"
            ]].style.format({
                "blast_radius_pct": "{:.1%}",
                "propagation_delay_hours": "{:.1f}h",
                "mean_persistence_half_life_hours": "{:.1f}h"
            }),
            use_container_width=True
        )

    # TempME Explanations Section
    if not df_tempme.empty:
        st.markdown("---")
        st.write("#### 🔍 TempME Spatio-Temporal Explanations (Test Shock Events)")
        st.markdown(
            "Identified explanatory subgraphs and rank correlation against statistical connectedness (Diebold-Yilmaz NET)."
        )

        st.dataframe(
            df_tempme[[
                "timestep", "target_asset", "fidelity_plus", "fidelity_minus",
                "sparsity", "spearman_dy_net", "top_explanatory_edges"
            ]].style.format({
                "fidelity_plus": "{:+.4f}",
                "fidelity_minus": "{:.4f}",
                "sparsity": "{:.1%}",
                "spearman_dy_net": "{:+.3f}"
            }),
            use_container_width=True
        )
    else:
        st.info("No TempME explanation records found in `reports/tempme_explanations.csv`.")
