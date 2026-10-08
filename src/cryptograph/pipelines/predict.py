"""Per-coin contagion risk from the final TGN on the latest processed data, with local run tracking.

    cryptograph-predict [--config configs/experiment/predict_tgn_1h.yaml] [--as-of "YYYY-MM-DD HH:MM"] [--run runs/<id>]

Prints the riskiest coins per horizon and the top alerts with their drivers; writes runs/<run_id>/ (risk.csv,
alerts.json, summary.json, shock_sources.csv, persistence.csv), which the dashboard reads. See serving/predict.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..training.runner import run_experiment
from ._experiment import load_experiment

DEFAULT_CONFIG = "configs/experiment/predict_tgn_1h.yaml"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Contagion risk per coin after the most recent shocks")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--as-of", default=None, help='UTC hour to predict from, e.g. "2025-10-10 21:00" (default: latest)')
    ap.add_argument("--run", default=None, help="trained TGN run directory (default: latest tgn_1h_final_v2 seed 0)")
    ap.add_argument("--lookback-hours", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_experiment(args.config)
    pc = cfg.setdefault("predict", {})
    for key, val in (("as_of", args.as_of), ("target_run", args.run), ("lookback_hours", args.lookback_hours)):
        if val is not None:
            pc[key] = val
    out = run_experiment(cfg, seed=args.seed, device="cpu")
    rd = Path(out["run_dir"])
    s = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    risk = pd.read_csv(rd / "risk.csv")
    print(f"\nas of {s['as_of']} UTC (data end {s['data_end']}), shocks from {', '.join(s['source_coins'])}; "
          f"model {s['target_run']}" + ("" if s["calibrated"] else " (uncalibrated scores)"))
    from ..serving.dashboard_data import risk_matrix

    wide = risk_matrix(risk).head(10)
    print("\nriskiest coins (by lift over the base rate), P(shock within k hours):\n"
          + wide.map("{:.1%}".format).to_string())
    if "base_rate" in risk:
        br = risk.drop_duplicates("horizon_h").set_index("horizon_h")["base_rate"]
        print("base rate  " + "  ".join(f"{v:.1%}" for v in br.reindex(wide.columns)))
    alerts = json.loads((rd / "alerts.json").read_text(encoding="utf-8"))
    print("\ntop alerts:")
    for a in alerts[:5]:
        drv = "; ".join(f"{d['input']} ({d['direction']})" for d in a["drivers"][:3])
        print(f"  {a['coin']:>5} within {a['horizon_h']:>2} h: {a['risk']:.1%} (from {a['source']}) - {drv}")
    print(f"\nrun_id: {out['run_id']}\nrun dir: {rd}")


if __name__ == "__main__":
    main()
