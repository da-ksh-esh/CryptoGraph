"""Light, print-friendly figure style and a savefig helper (PNG to reports/figures)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

PALETTE = ["#2a6fb0", "#d9822b", "#3a9e5f", "#c8423e", "#7b5ea7", "#8c6d4f", "#d47fb4", "#6f6f6f",
           "#b5b53a", "#2fb3c4"]
EVENT_COLORS = {"single_asset_collapse": "#c8423e", "exchange_or_lender_failure": "#d9822b",
                "regulatory": "#7b5ea7", "macro": "#2a6fb0", "market_structure": "#6f6f6f"}


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": "#444444", "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8, "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 150, "font.family": "DejaVu Sans",
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def savefig(fig, name: str, figures_dir: Path) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    p = figures_dir / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    return p


def add_event_lines(ax, events: pd.DataFrame, label: bool = True, ymax_frac: float = 0.98) -> None:
    for _, e in events.iterrows():
        ax.axvline(e["date"], color=EVENT_COLORS.get(e["event_type"], "#999999"), lw=0.8, ls="--", alpha=0.8)
        if label:
            ax.text(e["date"], ax.get_ylim()[1], " " + e["name"].split("/")[0].strip()[:18], rotation=90,
                    va="top", ha="right", fontsize=6.5, color="#555555")
