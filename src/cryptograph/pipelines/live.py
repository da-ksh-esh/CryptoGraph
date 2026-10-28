"""Live scanning with the final TGN (`cryptograph-live`); see serving/live.py.

    cryptograph-live                 # one cycle now: fetch the newest closed hours, rebuild, verify, predict, score alerts
    cryptograph-live --loop          # one cycle per closed hour (at hh:02 UTC), until stopped with Ctrl+C
    cryptograph-live --status        # the last cycle and the live track record, without fetching

Prints the riskiest coins of the newest prediction and the track record of earlier alerts; the dashboard
(streamlit run dashboard/app.py) shows the same, live.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from ..serving.live import LIVE_CONFIG_PATH, load_live_config, read_alert_log, run_cycle, project_path, track_record


def _print_status(lc) -> None:
    runs = project_path(lc.runs_dir)
    sp = runs / "status.json"
    if not sp.exists():
        print("no live cycle yet: run `cryptograph-live`")
        return
    s = json.loads(sp.read_text(encoding="utf-8"))
    print(f"last cycle {s.get('finished_utc', '?')} UTC: {s['status']}"
          + (f" ({s.get('error')})" if s["status"] != "ok" else
             f", data to {s.get('data_end')} UTC, latest shocks at {s.get('as_of')} ({', '.join(s.get('source_coins') or [])}), "
             f"{s.get('duration_s')} s"))
    if s.get("run_id"):
        risk = pd.read_csv(runs / s["run_id"] / "risk.csv")
        if "lift" in risk:
            top = risk.sort_values(["lift", "risk"], ascending=False).drop_duplicates("coin").head(5)
            print("\nriskiest now (lift over a typical coin):")
            for r in top.itertuples():
                print(f"  {r.coin:>5} within {r.horizon_h:>2} h: {r.risk:.0%} ({r.lift:.1f}x typical; after {r.source})")
    log = read_alert_log(runs / "alert_log.csv")
    tr = track_record(log)
    if len(tr):
        a = tr[tr["horizon_h"] == "all"].iloc[0]
        ratio = (f"{a['times_chance']:.2f}x chance" if a["expected_by_chance"] > 0
                 else "no coin made a big move in those windows")
        print(f"\ntrack record: {a['hits']} of {a['alerts']} resolved alerts came true; random coins: "
              f"{a['expected_by_chance']:.1f} ({ratio})")
    pending = int((log["status"] == "pending").sum()) if len(log) else 0
    print(f"alerts logged {len(log)}, waiting for their window: {pending}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Live contagion risk from Binance's newest hourly bars")
    ap.add_argument("--config", default=LIVE_CONFIG_PATH)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--loop", action="store_true", help="run one cycle per closed hour until stopped")
    mode.add_argument("--status", action="store_true", help="show the last cycle and the track record only")
    ap.add_argument("--no-predict", action="store_true", help="update the data and the alert log only")
    args = ap.parse_args(argv)
    lc = load_live_config(args.config)
    if args.status:
        _print_status(lc)
        return
    while True:
        try:
            s = run_cycle(lc, predict=not args.no_predict)
            print(f"\ncycle done in {s['duration_s']} s: data to {s['data_end']} UTC "
                  f"({s['fetch']['new_bars']} new coin-hours){'; rebuilt and verified' if s.get('rebuilt') else ''}")
            _print_status(lc)
        except Exception as e:                       # in a loop, a failed hour (e.g. no network) must not stop it
            if not args.loop:
                raise
            print(f"cycle failed: {type(e).__name__}: {e}")
        if not args.loop:
            return
        now = pd.Timestamp.now(tz="UTC")
        nxt = now.floor("h") + pd.Timedelta(hours=1, minutes=lc.cycle_minute)
        if nxt - pd.Timedelta(hours=1) > now:                # still before this hour's start minute
            nxt -= pd.Timedelta(hours=1)
        print(f"next cycle at {nxt:%H:%M} UTC")
        time.sleep(max(5.0, (nxt - pd.Timestamp.now(tz="UTC")).total_seconds()))


if __name__ == "__main__":
    main()
