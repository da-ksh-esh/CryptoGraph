"""Data pipeline stages. Each module exposes `run(cfg, refresh=False)`."""

from __future__ import annotations

import pandas as pd

from ...data.config import DataConfig
from ...data.storage import read_json


def load_universe(cfg: DataConfig) -> tuple[pd.DataFrame, dict, list[str]]:
    """Audit table, panel definition and the ordered master coin list (A ∪ B ∪ event coins)."""
    audit_path = cfg.reports_dir / "universe_audit.csv"
    panels_path = cfg.interim_dir / "universe_panels.json"
    if not audit_path.exists() or not panels_path.exists():
        raise FileNotFoundError("Run `cryptograph-data --stage audit` first (universe not defined).")
    audit = pd.read_csv(audit_path)
    panels = read_json(panels_path)
    ordered = audit.sort_values("market_cap_usd", ascending=False)["coin"].tolist()
    members = set(panels["panel_A"]) | set(panels["panel_B"]) | set(cfg.universe.event_coins)
    coins = [c for c in ordered if c in members]
    return audit.set_index("coin"), panels, coins
