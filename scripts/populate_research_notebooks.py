"""Populates research notebooks 03-06 with executable cells, data loaders, and visualizations."""

import json
from pathlib import Path


def update_notebook_03():
    nb_path = Path("notebooks/03_models_vs_baselines.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 5: CryptoLSTM
    nb["cells"][5]["source"] = [
        "# Load best trained CryptoLSTM checkpoint and evaluate architecture\n",
        "import glob\n",
        "lstm_ckpts = sorted(glob.glob('../runs/*lstm*/best_lstm_checkpoint.pt'))\n",
        "if lstm_ckpts:\n",
        "    ckpt_file = lstm_ckpts[-1]\n",
        "    print(f'Loading latest LSTM checkpoint: {ckpt_file}')\n",
        "    ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)\n",
        "    lstm_model = CryptoLSTM(in_features=features.shape[-1], hidden_dim=64, num_layers=2, out_horizons=len(data['horizons']))\n",
        "    lstm_model.load_state_dict(ckpt['model_state_dict'])\n",
        "    lstm_model.eval()\n",
        "    print('CryptoLSTM successfully loaded with 2 LSTM layers and multi-horizon head.')\n",
        "    print(f'Best Validation Mean AUPRC: {ckpt.get(\"best_val_auprc\", 0.0):.4f}')\n",
        "else:\n",
        "    print('No saved LSTM checkpoint found.')\n"
    ]

    # Cell 7: CryptoTGN
    nb["cells"][7]["source"] = [
        "# Load best trained CryptoTGN checkpoint\n",
        "tgn_ckpts = sorted(glob.glob('../runs/*tgn*/best_tgn_checkpoint.pt'))\n",
        "if tgn_ckpts:\n",
        "    ckpt_file = tgn_ckpts[-1]\n",
        "    print(f'Loading latest TGN checkpoint: {ckpt_file}')\n",
        "    ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)\n",
        "    tgn_model = CryptoTGN(num_nodes=30, raw_msg_dim=9, node_feat_dim=16, memory_dim=64, embedding_dim=64, out_horizons=5)\n",
        "    tgn_model.load_state_dict(ckpt['model_state_dict'])\n",
        "    tgn_model.eval()\n",
        "    print('CryptoTGN successfully loaded with dynamic memory and spatial GCNConv.')\n",
        "    print(f'Best Validation Mean AUPRC: {ckpt.get(\"best_val_auprc\", 0.0):.4f}')\n",
        "else:\n",
        "    print('No saved TGN checkpoint found.')\n"
    ]

    # Cell 9: Comparative Leaderboard & Visualization
    nb["cells"][9]["source"] = [
        "# Display Comparative Leaderboard from reports/model_comparison.csv\n",
        "comp_file = Path('../reports/model_comparison.csv')\n",
        "if comp_file.exists():\n",
        "    df_comp = pd.read_csv(comp_file)\n",
        "    pivot_auprc = df_comp.pivot(index='horizon_hours', columns='model', values='test_auprc')\n",
        "    pivot_auroc = df_comp.pivot(index='horizon_hours', columns='model', values='test_auroc')\n",
        "    \n",
        "    print('=== Test AUPRC Across Horizons (Primary Imbalanced Metric) ===')\n",
        "    print(pivot_auprc.to_string())\n",
        "    print('\\n=== Test AUROC Across Horizons ===')\n",
        "    print(pivot_auroc.to_string())\n",
        "    \n",
        "    fig = px.line(\n",
        "        df_comp,\n",
        "        x='horizon_hours',\n",
        "        y='test_auprc',\n",
        "        color='model',\n",
        "        markers=True,\n",
        "        title='Contagion Shock Forecasting: Test AUPRC vs. Horizon (1h - 24h)',\n",
        "        labels={'horizon_hours': 'Forecast Horizon (Hours)', 'test_auprc': 'AUPRC (Average Precision)'}\n",
        "    )\n",
        "    fig.update_layout(template='plotly_white')\n",
        "    fig.show()\n",
        "else:\n",
        "    print('reports/model_comparison.csv not found.')\n"
    ]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Updated 03_models_vs_baselines.ipynb")


def update_notebook_04():
    nb_path = Path("notebooks/04_tempme_explanations.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 3: Load TGN Checkpoint and Instances
    nb["cells"][3]["source"] = [
        "import glob\n",
        "import torch\n",
        "from cryptograph.models.tgn import CryptoTGN\n",
        "\n",
        "tgn_ckpts = sorted(glob.glob('../runs/*tgn*/best_tgn_checkpoint.pt'))\n",
        "if tgn_ckpts:\n",
        "    ckpt_file = tgn_ckpts[-1]\n",
        "    print(f'Loading TGN checkpoint: {ckpt_file}')\n",
        "    ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)\n",
        "    model = CryptoTGN(num_nodes=30, raw_msg_dim=9, node_feat_dim=16, memory_dim=64, embedding_dim=64, out_horizons=5)\n",
        "    model.load_state_dict(ckpt['model_state_dict'])\n",
        "    model.eval()\n",
        "    print('CryptoTGN model ready for explainability attribution.')\n",
        "else:\n",
        "    print('No checkpoint found.')\n"
    ]

    # Cell 5: Load Generated Explanations
    nb["cells"][5]["source"] = [
        "exp_csv = Path('../reports/tempme_explanations.csv')\n",
        "if exp_csv.exists():\n",
        "    df_exp = pd.read_csv(exp_csv)\n",
        "    print(f'Loaded {len(df_exp)} TempME test explanation instances.')\n",
        "    print(df_exp[['timestep', 'target_asset', 'fidelity_plus', 'fidelity_minus', 'sparsity', 'top_explanatory_edges']].head(10).to_string())\n",
        "else:\n",
        "    print('reports/tempme_explanations.csv not found.')\n"
    ]

    # Cell 7: Plotly Visualization of Explanatory Subgraphs
    nb["cells"][7]["source"] = [
        "import networkx as nx\n",
        "\n",
        "if exp_csv.exists() and len(df_exp) > 0:\n",
        "    sample_row = df_exp.iloc[0]\n",
        "    target = sample_row['target_asset']\n",
        "    edge_str = sample_row['top_explanatory_edges']\n",
        "    print(f'Explaining contagion prediction for {target}:')\n",
        "    print(f'Top motif edges: {edge_str}')\n",
        "    \n",
        "    G_exp = nx.DiGraph()\n",
        "    for item in edge_str.split(';'):\n",
        "        if '->' in item:\n",
        "            uv, w_str = item.split('(')\n",
        "            u, v = uv.split('->')\n",
        "            w = float(w_str.rstrip(')'))\n",
        "            G_exp.add_edge(u, v, weight=w)\n",
        "            \n",
        "    pos = nx.spring_layout(G_exp, seed=42)\n",
        "    edge_x = []\n",
        "    edge_y = []\n",
        "    for u, v in G_exp.edges():\n",
        "        x0, y0 = pos[u]\n",
        "        x1, y1 = pos[v]\n",
        "        edge_x.extend([x0, x1, None])\n",
        "        edge_y.extend([y0, y1, None])\n",
        "        \n",
        "    node_x = [pos[n][0] for n in G_exp.nodes()]\n",
        "    node_y = [pos[n][1] for n in G_exp.nodes()]\n",
        "    \n",
        "    fig = go.Figure(\n",
        "        data=[\n",
        "            go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color='#888'), hoverinfo='none', mode='lines'),\n",
        "            go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G_exp.nodes()), textposition='top center', marker=dict(size=14, color='#1f77b4'))\n",
        "        ],\n",
        "        layout=go.Layout(\n",
        "            title=f'TempME Attributed Subgraph for Shock Target: {target}',\n",
        "            showlegend=False,\n",
        "            template='plotly_white',\n",
        "            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),\n",
        "            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)\n",
        "        )\n",
        "    )\n",
        "    fig.show()\n"
    ]

    # Cell 9: Quantitative Validation
    nb["cells"][9]["source"] = [
        "val_csv = Path('../reports/explanation_validation.csv')\n",
        "if val_csv.exists():\n",
        "    df_val = pd.read_csv(val_csv)\n",
        "    print('=== TempME Explanation Validation Summary ===')\n",
        "    print(df_val.T.to_string())\n",
        "    print('\\nKey Findings:')\n",
        "    print(f'1. Fidelity+ (Ablation Degradation): {df_val[\"mean_fidelity_plus\"].values[0]:.4f}')\n",
        "    print(f'2. Fidelity- (Attribution Retention): {df_val[\"mean_fidelity_minus\"].values[0]:.4f}')\n",
        "    print(f'3. Sparsity (% edges pruned): {df_val[\"mean_sparsity\"].values[0]*100:.2f}%')\n",
        "    print(f'4. Spearman Rank Agreement with DY NET Spillover: {df_val[\"mean_spearman_dy_net\"].values[0]:.4f}')\n",
        "else:\n",
        "    print('reports/explanation_validation.csv not found.')\n"
    ]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Updated 04_tempme_explanations.ipynb")


def update_notebook_05():
    nb_path = Path("notebooks/05_persistence_by_shock_type.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 3: Load Historical Events & Persistence Analysis
    nb["cells"][3]["source"] = [
        "pers_file = Path('../reports/persistence_analysis.csv')\n",
        "if pers_file.exists():\n",
        "    df_pers = pd.read_csv(pers_file)\n",
        "    print('=== Historical Crisis Shock Propagation & Persistence ===')\n",
        "    print(df_pers[['event_name', 'timestamp', 'shock_type', 'blast_radius_count', 'blast_radius_pct', 'propagation_delay_hours', 'mean_persistence_half_life_hours', 'most_impacted_sector']].to_string())\n",
        "else:\n",
        "    print('reports/persistence_analysis.csv not found.')\n"
    ]

    # Cell 5: Case Study 1: August 2024 Yen Carry Unwind
    nb["cells"][5]["source"] = [
        "if pers_file.exists():\n",
        "    yen_event = df_pers[df_pers['event_name'].str.contains('Yen', case=False, na=False)]\n",
        "    if len(yen_event) > 0:\n",
        "        row = yen_event.iloc[0]\n",
        "        print('Case Study: August 2024 Yen Carry-Trade Unwind')\n",
        "        print(f'- Initial Transmitter Coins: {row[\"initial_transmitters\"]}')\n",
        "        print(f'- Blast Radius: {row[\"blast_radius_count\"]} coins ({row[\"blast_radius_pct\"]*100:.1f}% of universe)')\n",
        "        print(f'- Propagation Delay: {row[\"propagation_delay_hours\"]} hours to maximum cross-asset transmission')\n",
        "        print(f'- Volatility Persistence Half-Life: {row[\"mean_persistence_half_life_hours\"]:.1f} hours')\n",
        "        print(f'- Primary Sector Impacted: {row[\"most_impacted_sector\"]}')\n"
    ]

    # Cell 7: Case Study 2: FTX Collapse
    nb["cells"][7]["source"] = [
        "events_df = pd.read_csv('../data/external/events.csv')\n",
        "print('=== Historical Events Catalog ===')\n",
        "print(events_df[['date', 'name', 'is_crisis']].to_string())\n"
    ]

    # Cell 9: Cross-Sector Kruskal-Wallis Non-Parametric Test
    nb["cells"][9]["source"] = [
        "from cryptograph.analysis.persistence import ShockPersistenceAnalyzer\n",
        "analyzer = ShockPersistenceAnalyzer()\n",
        "kw_res = analyzer.compute_sector_kruskal_wallis(\n",
        "    dataset_path='../data/processed/v2/1h/processed_dataset.pt',\n",
        "    sectors_csv_path='../data/external/sectors.csv'\n",
        ")\n",
        "print('=== Kruskal-Wallis Non-Parametric Test Across Sectors ===')\n",
        "print(f'- H-Statistic: {kw_res[\"kruskal_statistic\"]:.4f}')\n",
        "print(f'- p-value: {kw_res[\"p_value\"]:.4e}')\n",
        "print(f'- Evaluated Sectors: {kw_res[\"num_sectors\"]}')\n",
        "if kw_res[\"p_value\"] < 0.05:\n",
        "    print('=> Statistically significant: Shock persistence durations differ fundamentally across sectors (p < 0.05).')\n",
        "else:\n",
        "    print('=> Null hypothesis not rejected.')\n"
    ]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Updated 05_persistence_by_shock_type.ipynb")


def update_notebook_06():
    nb_path = Path("notebooks/06_robustness_tgae.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 3: T-GAE Anomaly Reconstruction Errors
    nb["cells"][3]["source"] = [
        "anom_file = Path('../reports/tgae_anomalies.csv')\n",
        "if anom_file.exists():\n",
        "    df_anom = pd.read_csv(anom_file)\n",
        "    print('=== Spatio-Temporal Graph Autoencoder (T-GAE) Reconstruction Errors ===')\n",
        "    print(df_anom[['event_name', 'event_date', 'baseline_reconstruction_error', 'event_reconstruction_error', 'anomaly_ratio', 'most_anomalous_asset', 'is_statistical_anomaly']].to_string())\n",
        "else:\n",
        "    print('reports/tgae_anomalies.csv not found.')\n"
    ]

    # Cell 5: Anomaly Ratio Visualization
    nb["cells"][5]["source"] = [
        "if anom_file.exists():\n",
        "    fig = px.bar(\n",
        "        df_anom,\n",
        "        x='event_name',\n",
        "        y='anomaly_ratio',\n",
        "        color='is_statistical_anomaly',\n",
        "        title='T-GAE Reconstruction Error Ratio: Market Stress Events vs. Baseline',\n",
        "        labels={'anomaly_ratio': 'Reconstruction Error Ratio (Event / Baseline)', 'event_name': 'Market Event'}\n",
        "    )\n",
        "    fig.add_hline(y=1.5, line_dash='dash', line_color='red', annotation_text='Anomaly Threshold (1.5x)')\n",
        "    fig.update_layout(template='plotly_white')\n",
        "    fig.show()\n"
    ]

    # Cell 7: Adversarial Perturbation Degradation Curves
    nb["cells"][7]["source"] = [
        "deg_file = Path('../reports/robustness_degradation.csv')\n",
        "if deg_file.exists():\n",
        "    df_deg = pd.read_csv(deg_file)\n",
        "    print('=== Perturbation Degradation Curves ===')\n",
        "    print(df_deg.to_string())\n",
        "    \n",
        "    fig = px.line(\n",
        "        df_deg,\n",
        "        x='epsilon',\n",
        "        y=['prediction_drift_mse', 'explanation_jaccard_stability'],\n",
        "        markers=True,\n",
        "        title='Adversarial Resilience: Prediction Drift & Explanation Stability vs. Perturbation Budget',\n",
        "        labels={'epsilon': 'Perturbation Budget (Epsilon)', 'value': 'Metric Score'}\n",
        "    )\n",
        "    fig.update_layout(template='plotly_white')\n",
        "    fig.show()\n",
        "else:\n",
        "    print('reports/robustness_degradation.csv not found.')\n"
    ]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Updated 06_robustness_tgae.ipynb")


if __name__ == "__main__":
    update_notebook_03()
    update_notebook_04()
    update_notebook_05()
    update_notebook_06()
