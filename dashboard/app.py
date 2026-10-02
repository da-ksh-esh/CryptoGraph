import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
dashboard_dir = Path(__file__).resolve().parent
if str(dashboard_dir) not in sys.path:
    sys.path.insert(0, str(dashboard_dir))

import streamlit as st
from components.graph_view import render_graph_view
from components.spillover_heatmap import render_spillover_heatmap
from components.anomaly_alerts import render_anomaly_alerts
from components.robustness_panel import render_robustness_panel

st.set_page_config(
    page_title="CryptoGraph-Secure | Shock Spillover & Security",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ CryptoGraph-Secure: Dynamic Graph & Cybersecurity Dashboard")
st.markdown("Real-time temporal graph neural network monitoring for cryptocurrency shock spillovers and anomaly detection.")

tabs = st.tabs(["📊 Dynamic Network Graph", "🔗 Spillover Propagation", "🚨 Cybersecurity & Anomalies", "🛡️ Adversarial Stress Testing"])

with tabs[0]:
    render_graph_view()

with tabs[1]:
    render_spillover_heatmap()

with tabs[2]:
    render_anomaly_alerts()

with tabs[3]:
    render_robustness_panel()
