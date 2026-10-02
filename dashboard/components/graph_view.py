"""Component: Dynamic Cryptocurrency Relationship Graph."""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import torch

from cryptograph.data.universe import SECTOR_TAXONOMY


@st.cache_data
def load_rankings_data():
    csv_path = Path("reports/rankings.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_data
def load_correlation_matrix():
    dataset_path = Path("data/processed/v2/1h/processed_dataset.pt")
    if dataset_path.exists():
        data = torch.load(dataset_path, map_location="cpu", weights_only=False)
        features = data["features"]  # (T, N, F)
        returns = features[:, :, 0].numpy()
        assets = data["asset_names"]
        corr = np.corrcoef(returns, rowvar=False)
        return assets, corr
    return [], np.zeros((0, 0))


def render_graph_view():
    st.subheader("🌐 Dynamic Inter-Asset Network Topology & Centrality")
    st.markdown(
        "Interactive topological visualization of the cryptocurrency network. "
        "Node size reflects **PageRank Centrality**, and node color reflects **Diebold-Yilmaz Net Spillover** "
        "(Red = Net Shock Transmitter, Blue = Net Shock Receiver)."
    )

    df_rankings = load_rankings_data()
    assets, corr_matrix = load_correlation_matrix()

    if df_rankings.empty:
        st.warning("Rankings data not found in `reports/rankings.csv`.")
        return

    # Add Sector taxonomy
    df_rankings["sector"] = df_rankings["coin"].map(lambda c: SECTOR_TAXONOMY.get(c, "General"))

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        corr_thresh = st.slider("Correlation Edge Threshold (|r|)", 0.20, 0.80, 0.40, 0.05)
    with col2:
        centrality_metric = st.selectbox("Node Size Metric", ["pagerank", "eigenvector"], index=0)
    with col3:
        filter_sector = st.multiselect("Filter Sectors", sorted(list(set(df_rankings["sector"]))), default=[])

    filtered_df = df_rankings if not filter_sector else df_rankings[df_rankings["sector"].isin(filter_sector)]

    # Compute network layout coordinates using circular or spring-like projection
    n_nodes = len(df_rankings)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    node_x = np.cos(angles)
    node_y = np.sin(angles)
    pos_map = {row["coin"]: (node_x[i], node_y[i]) for i, (_, row) in enumerate(df_rankings.iterrows())}

    # Build Edges
    edge_x = []
    edge_y = []
    edge_weights = []

    if len(assets) > 0 and corr_matrix.shape[0] == len(assets):
        coin_to_idx = {c: i for i, c in enumerate(assets)}
        for i, c1 in enumerate(df_rankings["coin"]):
            for j, c2 in enumerate(df_rankings["coin"]):
                if i >= j or c1 not in coin_to_idx or c2 not in coin_to_idx:
                    continue
                w = abs(corr_matrix[coin_to_idx[c1], coin_to_idx[c2]])
                if w >= corr_thresh:
                    x0, y0 = pos_map[c1]
                    x1, y1 = pos_map[c2]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_weights.append(w)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="rgba(180, 180, 180, 0.4)"),
        hoverinfo="none",
        mode="lines"
    )

    # Build Nodes
    node_x_plot = [pos_map[c][0] for c in filtered_df["coin"]]
    node_y_plot = [pos_map[c][1] for c in filtered_df["coin"]]
    node_text = [
        f"<b>{row['coin']}</b> ({row['sector']})<br>"
        f"Net Spillover: {row['dy_net_spillover']:+.2f}%<br>"
        f"PageRank: {row['pagerank']:.4f} (Rank #{row['pagerank_rank']})<br>"
        f"Eigenvector: {row['eigenvector']:.4f} (Rank #{row['eigenvector_rank']})"
        for _, row in filtered_df.iterrows()
    ]
    node_sizes = 15 + 40 * (filtered_df[centrality_metric] / filtered_df[centrality_metric].max())

    node_trace = go.Scatter(
        x=node_x_plot,
        y=node_y_plot,
        mode="markers+text",
        text=filtered_df["coin"],
        textposition="top center",
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale="RdYlBu_r",
            color=filtered_df["dy_net_spillover"],
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title="DY Net Spillover (%)",
                xanchor="left",
                titleside="right"
            ),
            line=dict(width=2, color="white")
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="Cryptocurrency Correlation Network & Spillover Centrality", x=0.5),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed Rankings Table
    st.subheader("📋 Centrality & Econometric Spillover Rankings")
    display_df = filtered_df[[
        "coin", "sector", "dy_net_spillover", "dy_rank",
        "pagerank", "pagerank_rank", "eigenvector", "eigenvector_rank"
    ]].sort_values(by="dy_net_spillover", ascending=False)

    st.dataframe(
        display_df.style.format({
            "dy_net_spillover": "{:+.2f}%",
            "pagerank": "{:.4f}",
            "eigenvector": "{:.4f}"
        }),
        use_container_width=True
    )
