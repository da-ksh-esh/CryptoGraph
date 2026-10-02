"""Fixes notebook 02 and notebook 06 cell errors."""

import json
from pathlib import Path


def fix_notebook_02():
    nb_path = Path("notebooks/02_network_analysis.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find cell 11 (the one containing mst_daily)
    for c in nb["cells"]:
        if any("mst_daily" in line for line in c.get("source", [])):
            c["source"] = [
                "net_feat_path = Path(\"../data/processed/v2/network_features.pt\")\n",
                "net_data = torch.load(net_feat_path, map_location=\"cpu\", weights_only=False)\n",
                "\n",
                "mst_series = pd.Series(net_data[\"mst_norm_length\"], index=timestamps)\n",
                "mod_series = pd.Series(net_data[\"modularity\"], index=timestamps)\n",
                "\n",
                "# Daily resample for smooth longitudinal visualization\n",
                "mst_daily = mst_series.resample(\"1D\").mean().dropna()\n",
                "\n",
                "fig_mst = go.Figure()\n",
                "fig_mst.add_trace(go.Scatter(\n",
                "    x=mst_daily.index, y=mst_daily.values, mode=\"lines\",\n",
                "    name=\"MST Normalized Tree Length L(t)\", line=dict(color=\"#ff9100\", width=2)\n",
                "))\n",
                "\n",
                "# Overlay events within sample window\n",
                "mst_tz = mst_daily.index.tz\n",
                "for _, ev in events_df.iterrows():\n",
                "    ev_date = pd.to_datetime(ev[\"date\"])\n",
                "    if mst_tz is not None:\n",
                "        ev_date = ev_date.tz_localize(mst_tz)\n",
                "    ev_label = str(ev.get(\"name\", ev.get(\"event\", \"Event\")))\n",
                "    if mst_daily.index.min() <= ev_date <= mst_daily.index.max():\n",
                "        fig_mst.add_vline(\n",
                "            x=ev_date, line_dash=\"dot\", line_color=\"#ef5350\",\n",
                "            annotation_text=ev_label[:18], annotation_position=\"top left\"\n",
                "        )\n",
                "\n",
                "fig_mst.update_layout(\n",
                "    title=\"MST Normalized Tree Length L(t) Across Historical Crises (Topological Compression)\",\n",
                "    xaxis_title=\"Date\", yaxis_title=\"Normalized Tree Length (lower = higher market panic)\",\n",
                "    template=\"plotly_dark\", height=450\n",
                ")\n",
                "fig_mst.show()\n"
            ]
            break

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Fixed 02_network_analysis.ipynb timezone and event name handling.")


def fix_notebook_06():
    nb_path = Path("notebooks/06_robustness_tgae.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Fix cell 1 imports
    nb["cells"][1]["source"] = [
        "%load_ext autoreload\n",
        "%autoreload 2\n",
        "\n",
        "import os\n",
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "src_path = str(Path(\"../src\").resolve())\n",
        "if src_path not in sys.path:\n",
        "    sys.path.insert(0, src_path)\n",
        "\n",
        "import torch\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import plotly.express as px\n",
        "import plotly.graph_objects as go\n",
        "\n",
        "from cryptograph.models.tgae import TemporalGAE, TemporalGraphAE\n"
    ]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Fixed 06_robustness_tgae.ipynb import.")


if __name__ == "__main__":
    fix_notebook_02()
    fix_notebook_06()
