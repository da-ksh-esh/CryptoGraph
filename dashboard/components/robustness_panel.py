"""Component: Adversarial Robustness and Model Verification Panel."""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_robustness_data():
    csv_path = Path("reports/robustness_degradation.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_data
def load_manifest_data():
    manifest_path = Path("checkpoints/manifest.json")
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}


def render_robustness_panel():
    st.subheader("🛡️ Adversarial Robustness & Production Verification")
    st.markdown(
        "Evaluates model resilience under **Task-Specific Adversarial Attacks (FGSM)** versus "
        "**Random Gaussian Feature & Topology Perturbations** across increasing perturbation budgets (ε)."
    )

    df_rob = load_robustness_data()
    manifest = load_manifest_data()

    if not df_rob.empty:
        col1, col2 = st.columns(2)

        # 1. Prediction Drift vs Epsilon
        with col1:
            st.write("#### 📉 Prediction Drift (MSE) vs Epsilon (ε)")
            fig_drift = px.line(
                df_rob,
                x="epsilon",
                y="prediction_drift_mse",
                color="perturbation_type",
                markers=True,
                labels={
                    "epsilon": "Perturbation Budget (ε)",
                    "prediction_drift_mse": "Prediction Drift MSE",
                    "perturbation_type": "Attack Type"
                },
                title="Forecast Vulnerability Under Perturbation"
            )
            fig_drift.update_layout(height=400)
            st.plotly_chart(fig_drift, use_container_width=True)

        # 2. Explanation Stability (Jaccard) vs Epsilon
        with col2:
            st.write("#### 🎯 Explanation Jaccard Stability vs Epsilon (ε)")
            fig_jaccard = px.line(
                df_rob,
                x="epsilon",
                y="explanation_jaccard_stability",
                color="perturbation_type",
                markers=True,
                labels={
                    "epsilon": "Perturbation Budget (ε)",
                    "explanation_jaccard_stability": "Jaccard Similarity Index",
                    "perturbation_type": "Attack Type"
                },
                title="Top-5 Explanatory Edge Persistence"
            )
            fig_jaccard.update_layout(height=400, yaxis_range=[0.0, 1.0])
            st.plotly_chart(fig_jaccard, use_container_width=True)

        # Degradation Table
        st.write("#### 📋 Empirical Degradation Benchmark Table")
        st.dataframe(
            df_rob.style.format({
                "epsilon": "{:.2f}",
                "prediction_drift_mse": "{:.6f}",
                "explanation_jaccard_stability": "{:.4f}"
            }),
            use_container_width=True
        )

    # Checkpoint Manifest Audit Section
    st.markdown("---")
    st.write("#### 🔒 Production Checkpoint & Artifact Manifest")
    if manifest and "checkpoints" in manifest:
        ckpts = manifest["checkpoints"]
        st.caption(f"Manifest Version: {manifest.get('version')} | Dataset SHA-256: `{manifest.get('dataset', {}).get('sha256', '')[:16]}...`")

        manifest_rows = []
        for c in ckpts:
            metrics = c.get("metrics", {})
            manifest_rows.append({
                "Model": c.get("model_name"),
                "Run ID": c.get("run_id"),
                "Seed": c.get("seed"),
                "Test Mean AUPRC": metrics.get("test_mean_auprc"),
                "Test Mean AUROC": metrics.get("test_mean_auroc"),
                "Git Commit": c.get("git_commit")[:8] if c.get("git_commit") else "N/A",
                "Checkpoint SHA-256": f"`{c.get('checkpoint_sha256', '')[:12]}...`"
            })

        df_man = pd.DataFrame(manifest_rows)
        st.dataframe(df_man, use_container_width=True)
    else:
        st.info("No checkpoint manifest found in `checkpoints/manifest.json`.")
