"""Shock Volatility Persistence and Event Studies (hourly resolution).

Measures post-shock volatility decay using deseasonalised returns r_ds without
rolling-window volatility smoothing. Evaluates excess volatility profiles
E[|r_ds, t+h|] / baseline for h = 1..72 against matched non-shock null profiles,
identifying empirical decay half-life across cryptocurrency sectors with >=3 coins
via joint calendar-day moving-block bootstrap stratified by temporal split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse
import numpy as np
import pandas as pd
import torch

from ..common.logger import get_logger

logger = get_logger(__name__)


def compute_excess_profile(
    r_ds_coin: np.ndarray,
    t: int,
    baseline_lookback_hours: int = 168,
    baseline_lead_hours: int = 24,
    horizon_hours: int = 72
) -> Tuple[float, np.ndarray]:
    """
    Computes post-event excess volatility profile E[|r_ds, t+h|] / baseline for h = 1..H.
    Strictly excludes the shock bar t itself.

    Args:
        r_ds_coin: Deseasonalised return series for a single coin of length T.
        t: Event trigger index.
        baseline_lookback_hours: Window start for pre-event baseline (default 168h = 7d).
        baseline_lead_hours: Window end for pre-event baseline (default 24h = 1d before t).
        horizon_hours: Number of post-event hours to measure (default 72h = 3d).

    Returns:
        (baseline_volatility, relative_volatility_profile)
    """
    abs_r = np.abs(r_ds_coin)
    start_base = t - baseline_lookback_hours
    end_base = t - baseline_lead_hours
    base = float(np.median(abs_r[start_base:end_base]))
    if base <= 1e-8:
        base = 1e-8
    profile = abs_r[t + 1 : t + horizon_hours + 1] / base
    return base, profile


def estimate_decay_half_life(
    gap_curve: np.ndarray,
    interpolate: bool = True
) -> float:
    """
    Estimates the empirical decay half-life from an excess gap curve.
    Finds the first hour h >= 1 at which the excess gap falls to <= 50% of its initial value at h = 1.
    """
    if len(gap_curve) == 0:
        return 0.0

    gap_1 = float(gap_curve[0])
    if gap_1 <= 1e-6:
        return 0.0

    half_target = gap_1 * 0.5
    crossings = np.where(gap_curve <= half_target)[0]
    if len(crossings) == 0:
        return float(len(gap_curve))

    first_idx = int(crossings[0])
    if not interpolate or first_idx == 0:
        return float(first_idx + 1)

    # Linear interpolation between first_idx - 1 and first_idx
    y_prev = float(gap_curve[first_idx - 1])
    y_curr = float(gap_curve[first_idx])
    if abs(y_prev - y_curr) > 1e-8:
        frac = (y_prev - half_target) / (y_prev - y_curr)
        return float(first_idx + max(0.0, min(1.0, frac)))
    return float(first_idx + 1)


class ShockPersistenceAnalyzer:
    """
    Analyzes post-shock volatility persistence and propagation across cryptocurrency sectors
    using honest deseasonalised returns panel['r_ds'] and matched non-shock null baselines.
    """

    def __init__(
        self,
        horizon_hours: int = 72,
        baseline_lookback_hours: int = 168,
        baseline_lead_hours: int = 24,
        min_cluster_spacing_hours: int = 24
    ):
        self.horizon_hours = horizon_hours
        self.baseline_lookback_hours = baseline_lookback_hours
        self.baseline_lead_hours = baseline_lead_hours
        self.min_cluster_spacing_hours = min_cluster_spacing_hours

    def extract_sector_episodes(
        self,
        panel: Dict[str, Any],
        sectors_df: pd.DataFrame,
        min_coins_per_sector: int = 3,
        seed: int = 42
    ) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]]]:
        """
        Extracts de-clustered shock episodes and matched null non-shock episodes per sector.
        """
        rng = np.random.default_rng(seed)
        coins = panel["coins"]
        model_coins = panel["model_coins"]
        model_idx = [coins.index(c) for c in model_coins]

        r_ds = panel["r_ds"].numpy()[:, model_idx]
        shocks = (panel["shock_today"] & panel["shock_today_mask"]).numpy()[:, model_idx]
        active = panel["active"].numpy()[:, model_idx]
        T, N = r_ds.shape
        times = pd.to_datetime(panel["times"])
        split_id = panel["split_id"].numpy()
        abs_r = np.abs(r_ds)

        sec_map = dict(zip(sectors_df["coin"], sectors_df["sector"]))
        sec_counts = pd.Series([sec_map.get(c, "other") for c in model_coins]).value_counts()
        eligible_sectors = sorted([s for s, count in sec_counts.items() if count >= min_coins_per_sector])
        logger.info("Eligible sectors (>=%d coins): %s", min_coins_per_sector, eligible_sectors)

        # Precompute 24h trailing shock mask per coin
        trailing_shock = np.zeros_like(shocks, dtype=bool)
        for i in range(N):
            trailing_shock[:, i] = pd.Series(shocks[:, i]).rolling(self.min_cluster_spacing_hours + 1, min_periods=1).max().astype(bool)

        # Group eligible null hours by (coin_idx, year*100 + month)
        ym_arr = times.year.values * 100 + times.month.values
        unique_ym = np.unique(ym_arr)
        H = self.horizon_hours
        W = self.baseline_lookback_hours

        eligible_null: Dict[Tuple[int, int], np.ndarray] = {}
        for i in range(N):
            for ym in unique_ym:
                m = (ym_arr == ym) & active[:, i] & (~trailing_shock[:, i])
                idx = np.where(m)[0]
                valid_idx = idx[(idx >= W) & (idx + H < T)]
                if len(valid_idx) > 0:
                    eligible_null[(i, int(ym))] = valid_idx

        # Extract episodes per sector
        sec_data: Dict[str, Dict[str, np.ndarray]] = {}
        for sec in eligible_sectors:
            sec_coins = [c for c in model_coins if sec_map.get(c) == sec]
            sec_coin_idx = [model_coins.index(c) for c in sec_coins]
            t_list, sh_list, nu_list, sp_list = [], [], [], []

            for i in sec_coin_idx:
                sh_idx = np.where(shocks[:, i] & active[:, i])[0]
                declust = []
                for t in sh_idx:
                    if t >= W and t + H < T:
                        if len(declust) == 0 or t - declust[-1] >= self.min_cluster_spacing_hours:
                            declust.append(t)

                for t in declust:
                    base = np.median(abs_r[t - W : t - self.baseline_lead_hours, i])
                    if base > 1e-6 and np.all(active[t - W : t + H + 1, i]):
                        cands = eligible_null.get((i, int(ym_arr[t])))
                        if cands is not None and len(cands) > 0:
                            tn = rng.choice(cands)
                            base_nu = np.median(abs_r[tn - W : tn - self.baseline_lead_hours, i])
                            if base_nu > 1e-6 and np.all(active[tn - W : tn + H + 1, i]):
                                t_list.append(t)
                                sh_list.append(abs_r[t + 1 : t + H + 1, i] / base)
                                nu_list.append(abs_r[tn + 1 : tn + H + 1, i] / base_nu)
                                sp_list.append(split_id[t])

            sec_data[sec] = {
                "t": np.array(t_list, dtype=np.int64),
                "sh": np.array(sh_list, dtype=np.float32) if sh_list else np.empty((0, H), dtype=np.float32),
                "nu": np.array(nu_list, dtype=np.float32) if nu_list else np.empty((0, H), dtype=np.float32),
                "sp": np.array(sp_list, dtype=np.int8)
            }
            logger.info("Sector '%s' extracted %d shock episodes.", sec, len(t_list))

        return eligible_sectors, sec_data

    def bootstrap_sector_persistence(
        self,
        panel_path: str = "data/processed/v2/1h/panel.pt",
        sectors_csv_path: str = "data/external/sectors.csv",
        output_csv_path: str = "reports/persistence_sector_block_bootstrap.csv",
        n_bootstrap: int = 500,
        block_hours: int = 24,
        min_coins_per_sector: int = 3,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Executes joint calendar-day moving-block bootstrap (24h blocks) across all coins,
        reporting post-shock decay half-lives, excess gaps over matched nulls, and 95% CIs
        stratified by temporal split (train, val, test, and all).
        """
        logger.info("Loading panel from %s...", panel_path)
        panel = torch.load(panel_path, map_location="cpu", weights_only=False)
        sectors_df = pd.read_csv(sectors_csv_path)

        eligible_sectors, sec_data = self.extract_sector_episodes(
            panel, sectors_df, min_coins_per_sector=min_coins_per_sector, seed=seed
        )

        T = len(panel["times"])
        H = self.horizon_hours
        rng = np.random.default_rng(seed)

        # Pre-generate joint calendar-day blocks across the entire timeline
        n_blocks = int(np.ceil(T / block_hours))
        starts_max = max(1, T - block_hours + 1)
        offsets = np.arange(block_hours)
        block_starts = rng.integers(0, starts_max, size=(n_bootstrap, n_blocks))

        split_defs = {
            "all": None,
            "train": 0,
            "val": 1,
            "test": 2
        }

        output_rows: List[Dict[str, Any]] = []

        for split_name, split_code in split_defs.items():
            logger.info("Bootstrapping sector persistence for split: %s...", split_name)

            # Filter data to split
            split_sec_data = {}
            for sec in eligible_sectors:
                d = sec_data[sec]
                if split_code is None:
                    m = np.ones(len(d["t"]), dtype=bool)
                else:
                    m = d["sp"] == split_code
                split_sec_data[sec] = {
                    "t": d["t"][m],
                    "sh": d["sh"][m],
                    "nu": d["nu"][m]
                }

            # 1. Point Estimates per Sector
            point_curves = {}
            point_gaps = {}
            point_hls = {}

            for sec in eligible_sectors:
                d = split_sec_data[sec]
                n_eps = len(d["t"])
                if n_eps > 0:
                    sh_m = np.nanmean(d["sh"], axis=0)
                    nu_m = np.nanmean(d["nu"], axis=0)
                    gap = sh_m - nu_m
                    hl = estimate_decay_half_life(gap)
                else:
                    sh_m = np.full(H, np.nan)
                    nu_m = np.full(H, np.nan)
                    gap = np.full(H, np.nan)
                    hl = np.nan

                point_curves[sec] = (sh_m, nu_m)
                point_gaps[sec] = gap
                point_hls[sec] = hl

            # 2. Joint Moving-Block Bootstrap Replicates
            boot_hls: Dict[str, List[float]] = {s: [] for s in eligible_sectors}
            boot_gap_h1: Dict[str, List[float]] = {s: [] for s in eligible_sectors}
            boot_gap_h24: Dict[str, List[float]] = {s: [] for s in eligible_sectors}

            for b in range(n_bootstrap):
                sampled_t = (block_starts[b, :, None] + offsets).ravel()
                sampled_t = sampled_t[sampled_t < T]
                time_counts = np.bincount(sampled_t, minlength=T)

                for sec in eligible_sectors:
                    d = split_sec_data[sec]
                    if len(d["t"]) > 0:
                        w = time_counts[d["t"]]
                        sw = np.sum(w)
                        if sw > 0:
                            sh_b = np.sum(w[:, None] * d["sh"], axis=0) / sw
                            nu_b = np.sum(w[:, None] * d["nu"], axis=0) / sw
                            gap_b = sh_b - nu_b
                            hl_b = estimate_decay_half_life(gap_b)
                            boot_hls[sec].append(hl_b)
                            boot_gap_h1[sec].append(float(gap_b[0]))
                            boot_gap_h24[sec].append(float(gap_b[23]) if len(gap_b) > 23 else np.nan)
                        else:
                            boot_hls[sec].append(np.nan)
                            boot_gap_h1[sec].append(np.nan)
                            boot_gap_h24[sec].append(np.nan)
                    else:
                        boot_hls[sec].append(np.nan)
                        boot_gap_h1[sec].append(np.nan)
                        boot_gap_h24[sec].append(np.nan)

            # Record sector summary rows
            for sec in eligible_sectors:
                d = split_sec_data[sec]
                n_eps = len(d["t"])
                if n_eps == 0:
                    continue

                sh_m, nu_m = point_curves[sec]
                gap = point_gaps[sec]
                pt_hl = point_hls[sec]

                hl_draws = np.array(boot_hls[sec])[np.isfinite(boot_hls[sec])]
                low, high = np.quantile(hl_draws, [0.025, 0.975]) if len(hl_draws) else (np.nan, np.nan)

                g1_draws = np.array(boot_gap_h1[sec])[np.isfinite(boot_gap_h1[sec])]
                p_beat_null = float(np.mean(g1_draws <= 0)) if len(g1_draws) else np.nan

                output_rows.append({
                    "record_type": "sector_decay_half_life",
                    "split": split_name,
                    "sector": sec,
                    "comparison": "excess_gap_over_matched_null",
                    "episode_count": n_eps,
                    "estimate_hours": round(pt_hl, 2),
                    "ci_95_low": round(float(low), 2),
                    "ci_95_high": round(float(high), 2),
                    "gap_h1": round(float(gap[0]), 3),
                    "gap_h24": round(float(gap[23]), 3) if len(gap) > 23 else np.nan,
                    "gap_h72": round(float(gap[71]), 3) if len(gap) > 71 else np.nan,
                    "p_value_beat_null": round(p_beat_null, 4),
                    "block_hours": block_hours,
                    "bootstrap_replicates": n_bootstrap
                })

            # Pairwise sector differences
            for i, sec_a in enumerate(eligible_sectors):
                for sec_b in eligible_sectors[i + 1 :]:
                    hl_a = point_hls[sec_a]
                    hl_b = point_hls[sec_b]
                    if np.isnan(hl_a) or np.isnan(hl_b):
                        continue

                    diff_point = float(hl_a - hl_b)
                    draws_a = np.array(boot_hls[sec_a])
                    draws_b = np.array(boot_hls[sec_b])
                    pair_diffs = draws_a - draws_b
                    pair_diffs = pair_diffs[np.isfinite(pair_diffs)]
                    p_low, p_high = np.quantile(pair_diffs, [0.025, 0.975]) if len(pair_diffs) else (np.nan, np.nan)

                    output_rows.append({
                        "record_type": "sector_pairwise_diff",
                        "split": split_name,
                        "sector": sec_a,
                        "comparison": f"{sec_a} - {sec_b}",
                        "episode_count": len(split_sec_data[sec_a]["t"]) + len(split_sec_data[sec_b]["t"]),
                        "estimate_hours": round(diff_point, 2),
                        "ci_95_low": round(float(p_low), 2),
                        "ci_95_high": round(float(p_high), 2),
                        "gap_h1": np.nan,
                        "gap_h24": np.nan,
                        "gap_h72": np.nan,
                        "p_value_beat_null": np.nan,
                        "block_hours": block_hours,
                        "bootstrap_replicates": n_bootstrap
                    })

        res_df = pd.DataFrame(output_rows)
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(output_csv_path, index=False)
        logger.info("Saved %d bootstrap report rows to %s.", len(res_df), output_csv_path)
        return res_df

    def run_historical_event_study(
        self,
        panel_path: str = "data/processed/v2/1h/panel.pt",
        events_csv_path: str = "data/external/events.csv",
        sectors_csv_path: str = "data/external/sectors.csv",
        output_csv_path: str = "reports/persistence_analysis.csv"
    ) -> pd.DataFrame:
        """
        Runs empirical event study around historical crises using deseasonalised returns r_ds.
        """
        panel = torch.load(panel_path, map_location="cpu", weights_only=False)
        coins = panel["coins"]
        model_coins = panel["model_coins"]
        model_idx = [coins.index(c) for c in model_coins]

        r_ds = panel["r_ds"].numpy()[:, model_idx]
        shocks = (panel["shock_today"] & panel["shock_today_mask"]).numpy()[:, model_idx]
        active = panel["active"].numpy()[:, model_idx]
        T, N = r_ds.shape
        ts_str_list = panel["times"]
        events_df = pd.read_csv(events_csv_path)
        sec_df = pd.read_csv(sectors_csv_path)
        sec_map = dict(zip(sec_df["coin"], sec_df["sector"]))
        abs_r = np.abs(r_ds)

        records = []
        for _, row in events_df.iterrows():
            ev_name = str(row["name"]) if "name" in row else str(row.get("event", "Event"))
            ev_date = str(row["date"])

            matching = [i for i, ts in enumerate(ts_str_list) if ts.startswith(ev_date)]
            if not matching:
                continue

            trigger_t = matching[0]
            end_t = min(T, trigger_t + self.horizon_hours)

            # Pre-event control baseline (168h to 24h prior)
            start_ctrl = max(0, trigger_t - self.baseline_lookback_hours)
            end_ctrl = max(0, trigger_t - self.baseline_lead_hours)
            ctrl_shocks = shocks[start_ctrl:trigger_t] & active[start_ctrl:trigger_t]
            ctrl_shock_rate = float(np.sum(ctrl_shocks) / max(1, np.sum(active[start_ctrl:trigger_t])))

            # Event window
            init_shocked = [model_coins[i] for i in range(N) if shocks[trigger_t, i] and active[trigger_t, i]]
            sub_shocks = shocks[trigger_t:end_t] & active[trigger_t:end_t]
            event_shock_rate = float(np.sum(sub_shocks) / max(1, np.sum(active[trigger_t:end_t])))

            any_shocked = np.any(sub_shocks, axis=0)
            affected = [model_coins[i] for i in range(N) if any_shocked[i]]
            blast_radius_count = len(affected)
            blast_radius_pct = float(blast_radius_count / max(1, np.sum(active[trigger_t])))

            # Propagation delay to peak secondary shocks
            if end_t > trigger_t + 1:
                sec_counts = np.sum(shocks[trigger_t + 1 : end_t] & active[trigger_t + 1 : end_t], axis=1)
                peak_delay = float(np.argmax(sec_counts) + 1) if np.max(sec_counts) > 0 else 0.0
            else:
                peak_delay = 0.0

            # Persistence half-life of affected coins
            half_lives = []
            for i in range(N):
                if any_shocked[i] and trigger_t >= self.baseline_lookback_hours:
                    base = np.median(abs_r[start_ctrl:end_ctrl, i])
                    if base > 1e-6 and trigger_t + self.horizon_hours < T:
                        prof = abs_r[trigger_t + 1 : trigger_t + self.horizon_hours + 1, i] / base
                        # Excess relative to 1.0 (baseline)
                        excess = prof - 1.0
                        if excess[0] > 0:
                            cross = np.where(excess <= 0.5 * excess[0])[0]
                            hl = (cross[0] + 1) if len(cross) > 0 else self.horizon_hours
                            half_lives.append(hl)

            mean_hl = float(np.mean(half_lives)) if half_lives else 0.0
            shock_type = "macro_systemic" if len(init_shocked) >= 5 or blast_radius_pct >= 0.50 else "idiosyncratic"

            impacted_sectors = [sec_map.get(a, "Other") for a in affected]
            lead_sector = str(pd.Series(impacted_sectors).value_counts().index[0]) if impacted_sectors else "Unknown"

            records.append({
                "event_name": ev_name,
                "timestamp": ts_str_list[trigger_t],
                "trigger_step": trigger_t,
                "shock_type": shock_type,
                "initial_transmitters": ";".join(init_shocked) if init_shocked else "None",
                "blast_radius_count": blast_radius_count,
                "blast_radius_pct": blast_radius_pct,
                "propagation_delay_hours": peak_delay,
                "mean_persistence_half_life_hours": mean_hl,
                "control_shock_rate": ctrl_shock_rate,
                "event_shock_rate": event_shock_rate,
                "abnormal_shock_rate": event_shock_rate - ctrl_shock_rate,
                "most_impacted_sector": lead_sector
            })

        res_df = pd.DataFrame(records)
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(output_csv_path, index=False)
        logger.info("Saved event study report to %s.", output_csv_path)
        return res_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dependence-aware sector persistence analysis on deseasonalised returns.")
    parser.add_argument("--panel", default="data/processed/v2/1h/panel.pt")
    parser.add_argument("--sectors", default="data/external/sectors.csv")
    parser.add_argument("--events", default="data/external/events.csv")
    parser.add_argument("--output", default="reports/persistence_sector_block_bootstrap.csv")
    parser.add_argument("--events-output", default="reports/persistence_analysis.csv")
    parser.add_argument("--block-hours", type=int, default=24)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--min-coins", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    analyzer = ShockPersistenceAnalyzer()
    logger.info("Executing Sector Block Bootstrap...")
    results = analyzer.bootstrap_sector_persistence(
        panel_path=args.panel,
        sectors_csv_path=args.sectors,
        output_csv_path=args.output,
        n_bootstrap=args.bootstrap_replicates,
        block_hours=args.block_hours,
        min_coins_per_sector=args.min_coins,
        seed=args.seed
    )
    logger.info("Wrote %d dependence-aware summary rows to %s", len(results), args.output)

    logger.info("Executing Historical Event Study...")
    analyzer.run_historical_event_study(
        panel_path=args.panel,
        events_csv_path=args.events,
        sectors_csv_path=args.sectors,
        output_csv_path=args.events_output
    )


if __name__ == "__main__":
    main()
