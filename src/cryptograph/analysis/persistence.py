"""Persistence and Event Studies: Empirical shock propagation delay, time-to-hit, and duration."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch

from ..common.logger import get_logger

logger = get_logger(__name__)


class ShockPersistenceAnalyzer:
    """
    Quantifies empirical shock propagation delay, time-to-hit, affected nodes (blast radius),
    and volatility persistence duration across historical crises and cryptocurrency sectors.
    Does not use placeholder approximations.
    """

    def __init__(self, shock_quantile: float = 0.95):
        self.shock_quantile = shock_quantile

    def estimate_half_life(self, response: np.ndarray) -> float:
        """
        Estimates the empirical half-life of a volatility or return impulse.
        Measures the number of hours for the response to decay to 50% of its peak amplitude.
        """
        if len(response) == 0:
            return 0.0
        peak_idx = int(np.argmax(np.abs(response)))
        peak_val = float(np.abs(response[peak_idx]))
        if peak_val < 1e-6:
            return 0.0

        target = peak_val / 2.0
        post_peak = np.abs(response[peak_idx:])
        below = np.where(post_peak <= target)[0]
        if len(below) > 0:
            return float(below[0])
        return float(len(post_peak))

    def compute_propagation_delay(
        self,
        shocks: np.ndarray,
        trigger_t: int,
        max_lookahead: int = 48
    ) -> float:
        """
        Measures propagation delay: hours from the initial trigger until the peak
        number of secondary peer assets experience shocks.
        """
        T, N = shocks.shape
        end_t = min(T, trigger_t + max_lookahead + 1)
        sub_shocks = shocks[trigger_t + 1 : end_t] # (H, N)

        if len(sub_shocks) == 0:
            return 0.0

        shock_counts = np.sum(sub_shocks, axis=1) # (H,)
        if np.max(shock_counts) == 0:
            return 0.0

        peak_h = int(np.argmax(shock_counts)) + 1
        return float(peak_h)

    def analyze_event_propagation(
        self,
        event_name: str,
        event_timestamp: str,
        trigger_t: int,
        features: np.ndarray,
        shocks: np.ndarray,
        active_mask: np.ndarray,
        asset_names: List[str],
        sectors_map: Dict[str, str],
        lookahead_hours: int = 48,
        control_hours: int = 48
    ) -> Dict[str, Any]:
        """
        Analyzes a single historical event's empirical shock propagation cascade against
        a pre-event baseline control window [t - W_control, t].
        
        NOTE: This quantifies temporal co-occurrence and abnormal volatility/shock surges
        relative to calm pre-event conditions, not counterfactual causality.
        """
        T, N, F = features.shape
        end_t = min(T, trigger_t + lookahead_hours)

        # 1. Pre-event control baseline: [start_ctrl, trigger_t]
        start_ctrl = max(0, trigger_t - control_hours)
        ctrl_shocks = shocks[start_ctrl:trigger_t] & active_mask[start_ctrl:trigger_t]
        ctrl_active_count = np.sum(active_mask[start_ctrl:trigger_t])
        ctrl_shock_rate = float(np.sum(ctrl_shocks) / max(1, ctrl_active_count))

        vol_col = 1 # Parkinson volatility
        ctrl_vol = float(np.mean(features[start_ctrl:trigger_t, :, vol_col])) if trigger_t > start_ctrl else 0.0

        # 2. Event window: [trigger_t, end_t]
        init_shocked = [asset_names[i] for i in range(N) if shocks[trigger_t, i] and active_mask[trigger_t, i]]

        sub_shocks = shocks[trigger_t:end_t] & active_mask[trigger_t:end_t]
        event_active_count = np.sum(active_mask[trigger_t:end_t])
        event_shock_rate = float(np.sum(sub_shocks) / max(1, event_active_count))
        abnormal_shock_rate = float(event_shock_rate - ctrl_shock_rate)

        event_vol = float(np.mean(features[trigger_t:end_t, :, vol_col]))
        vol_surge_ratio = float(event_vol / max(1e-6, ctrl_vol)) if ctrl_vol > 0 else 1.0

        any_shocked = np.any(sub_shocks, axis=0)
        affected_assets = [asset_names[i] for i in range(N) if any_shocked[i]]
        blast_radius_count = len(affected_assets)
        total_active = int(np.sum(active_mask[trigger_t]))
        blast_radius_pct = float(blast_radius_count / max(1, total_active))

        # Propagation delay
        delay_h = self.compute_propagation_delay(shocks, trigger_t, max_lookahead=lookahead_hours)

        # Average volatility persistence duration (half-life)
        half_lives = []
        for i in range(N):
            if any_shocked[i]:
                vol_window = features[trigger_t : min(T, trigger_t + 72), i, vol_col]
                hl = self.estimate_half_life(vol_window)
                half_lives.append(hl)

        mean_duration = float(np.mean(half_lives)) if half_lives else 0.0

        # Classify shock type: macro vs idiosyncratic
        shock_type = "macro_systemic" if len(init_shocked) >= 5 or blast_radius_pct >= 0.50 else "idiosyncratic"

        # Most impacted sector
        impacted_sectors = [sectors_map.get(a, "Other") for a in affected_assets]
        sector_counts = pd.Series(impacted_sectors).value_counts()
        lead_sector = str(sector_counts.index[0]) if len(sector_counts) > 0 else "Unknown"

        return {
            "event_name": event_name,
            "timestamp": event_timestamp,
            "trigger_step": trigger_t,
            "shock_type": shock_type,
            "initial_transmitters": ";".join(init_shocked) if init_shocked else "None",
            "blast_radius_count": blast_radius_count,
            "blast_radius_pct": blast_radius_pct,
            "propagation_delay_hours": delay_h,
            "mean_persistence_half_life_hours": mean_duration,
            "control_shock_rate": ctrl_shock_rate,
            "event_shock_rate": event_shock_rate,
            "abnormal_shock_rate": abnormal_shock_rate,
            "control_mean_volatility": ctrl_vol,
            "event_mean_volatility": event_vol,
            "volatility_surge_ratio": vol_surge_ratio,
            "most_impacted_sector": lead_sector
        }

    def run_historical_event_study(
        self,
        events_csv_path: str = "data/external/events.csv",
        sectors_csv_path: str = "data/external/sectors.csv",
        dataset_path: str = "data/processed/v2/1h/processed_dataset.pt",
        output_csv_path: str = "reports/persistence_analysis.csv"
    ) -> pd.DataFrame:
        """
        Runs comprehensive persistence analysis across historical market stress events.
        """
        data = torch.load(dataset_path, weights_only=False)
        features = data["features"].numpy()
        shocks = data["shocks"].numpy()
        act_mask = data["active_mask"].numpy()
        timestamps = [str(ts) for ts in data["timestamps"]]
        asset_names = data["asset_names"]

        # Load sectors
        sec_df = pd.read_csv(sectors_csv_path)
        sectors_map = dict(zip(sec_df["coin"], sec_df["sector"]))

        # Load events
        events_df = pd.read_csv(events_csv_path)

        records = []
        for _, row in events_df.iterrows():
            ev_name = str(row["name"]) if "name" in row else str(row.get("event", "Event"))
            ev_date = str(row["date"])

            # Find matching timestamp in dataset
            matching_indices = [idx for idx, ts in enumerate(timestamps) if ts.startswith(ev_date)]
            if not matching_indices:
                continue

            trigger_t = matching_indices[0]
            rec = self.analyze_event_propagation(
                event_name=ev_name,
                event_timestamp=timestamps[trigger_t],
                trigger_t=trigger_t,
                features=features,
                shocks=shocks,
                active_mask=act_mask,
                asset_names=asset_names,
                sectors_map=sectors_map,
                lookahead_hours=48
            )
            records.append(rec)

        res_df = pd.DataFrame(records)
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved persistence analysis report to {output_csv_path}.")
        return res_df

    def compute_sector_kruskal_wallis(
        self,
        dataset_path: str = "data/processed/v2/1h/processed_dataset.pt",
        sectors_csv_path: str = "data/external/sectors.csv"
    ) -> Dict[str, float]:
        """
        Computes the legacy independent-sample Kruskal-Wallis statistic.

        This method is retained for reproducibility only. Its p-value is not valid for
        the dependent asset-time panel and must not be used as inferential evidence.
        """
        data = torch.load(dataset_path, weights_only=False)
        features = data["features"].numpy()
        shocks = data["shocks"].numpy()
        asset_names = data["asset_names"]

        sec_df = pd.read_csv(sectors_csv_path)
        sec_map = dict(zip(sec_df["coin"], sec_df["sector"]))

        # Collect half lives per sector
        sector_durations: Dict[str, List[float]] = {}
        T, N, F = features.shape

        for i, asset in enumerate(asset_names):
            sector = sec_map.get(asset, "Other")
            if sector not in sector_durations:
                sector_durations[sector] = []

            # De-cluster shocks: enforce a minimum 24-hour refractory period between independent shock episodes
            raw_shocks = np.where(shocks[:, i])[0]
            distinct_episodes = []
            last_t = -999
            for t in raw_shocks:
                if t - last_t >= 24:
                    distinct_episodes.append(t)
                    last_t = t

            for t in distinct_episodes:
                vol_window = features[t : min(T, t + 48), i, 1]
                hl = self.estimate_half_life(vol_window)
                sector_durations[sector].append(hl)

        # Retain sectors with >= 10 independent samples
        valid_groups = [dur for s, dur in sector_durations.items() if len(dur) >= 10]
        # Deliberately do not emit a nominal p-value: the ordinary KW assumptions do
        # not hold for episodes clustered by asset and shared market time.
        if len(valid_groups) >= 2:
            all_values = np.concatenate([np.asarray(group, dtype=float) for group in valid_groups])
            ranks = rankdata(all_values, method="average")
            offset = 0
            rank_term = 0.0
            for group in valid_groups:
                rank_sum = float(ranks[offset : offset + len(group)].sum())
                rank_term += rank_sum * rank_sum / len(group)
                offset += len(group)
            n_total = len(all_values)
            stat = (12.0 / (n_total * (n_total + 1)) * rank_term) - 3.0 * (n_total + 1)
            _, tie_counts = np.unique(all_values, return_counts=True)
            tie_correction = 1.0 - float(np.sum(tie_counts**3 - tie_counts)) / (n_total**3 - n_total)
            stat = stat / tie_correction if tie_correction > 0 else 0.0
        else:
            stat = float("nan")

        return {
            "kruskal_statistic": float(stat),
            "p_value": float("nan"),
            "num_sectors": len(valid_groups),
            "total_declustered_episodes": sum(len(g) for g in valid_groups),
            "inference_status": "withdrawn_dependent_panel"
        }

    def bootstrap_sector_persistence(
        self,
        dataset_path: str = "data/processed/v2/1h/processed_dataset.pt",
        sectors_csv_path: str = "data/external/sectors.csv",
        output_csv_path: str = "reports/persistence_sector_block_bootstrap.csv",
        block_hours: int = 24,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Estimate sector median half-lives and uncertainty by joint moving-block bootstrap.

        Each sampled time block contains every asset together, preserving contemporaneous
        cross-sectional dependence and serial dependence within each block. Outputs are
        confidence intervals, not hypothesis-test p-values; they do not establish causality.
        """
        if block_hours < 1 or n_bootstrap < 100:
            raise ValueError("block_hours must be positive and n_bootstrap must be at least 100")
        data = torch.load(dataset_path, map_location="cpu", weights_only=False)
        features = data["features"].detach().cpu().numpy()
        shocks = data["shocks"].detach().cpu().numpy().astype(bool)
        active = data["active_mask"].detach().cpu().numpy().astype(bool)
        assets = data["asset_names"]
        T, N, _ = features.shape
        sectors_df = pd.read_csv(sectors_csv_path)
        sector_map = dict(zip(sectors_df["coin"], sectors_df["sector"]))

        events: list[tuple[int, str, float]] = []
        for node, asset in enumerate(assets):
            sector = sector_map.get(asset, "Other")
            last_t = -block_hours
            for t in np.flatnonzero(shocks[:, node] & active[:, node]):
                if t - last_t < block_hours:
                    continue
                last_t = int(t)
                window = features[t : min(T, t + 48), node, 1]
                events.append((int(t), str(sector), self.estimate_half_life(window)))
        if not events:
            raise ValueError("No active shock episodes found; cannot estimate sector persistence")

        event_df = pd.DataFrame(events, columns=["timestep", "sector", "half_life_hours"])
        sectors = sorted(event_df["sector"].unique())
        rng = np.random.default_rng(seed)
        n_blocks = int(np.ceil(T / block_hours))
        starts_max = max(1, T - block_hours + 1)
        boot = {sector: [] for sector in sectors}
        sector_events = {}
        for s in sectors:
            s_df = event_df[event_df["sector"] == s]
            sector_events[s] = (s_df["timestep"].values.astype(int), s_df["half_life_hours"].values.astype(float))

        offsets = np.arange(block_hours)
        for _ in range(n_bootstrap):
            starts = rng.integers(0, starts_max, size=n_blocks)
            sampled_times = (starts[:, None] + offsets).ravel()
            sampled_times = sampled_times[sampled_times < T]
            time_counts = np.bincount(sampled_times, minlength=T)

            for sector in sectors:
                t_arr, d_arr = sector_events[sector]
                counts = time_counts[t_arr]
                valid = counts > 0
                if np.any(valid):
                    repeated_vals = np.repeat(d_arr[valid], counts[valid])
                    boot[sector].append(float(np.median(repeated_vals)))
                else:
                    boot[sector].append(np.nan)

        rows = []
        for sector in sectors:
            point = event_df.loc[event_df["sector"] == sector, "half_life_hours"].median()
            draws = np.asarray(boot[sector], dtype=float)
            draws = draws[np.isfinite(draws)]
            low, high = np.quantile(draws, [0.025, 0.975]) if len(draws) else (np.nan, np.nan)
            rows.append({
                "record_type": "sector_median",
                "sector": sector,
                "comparison": "",
                "episode_count": int((event_df["sector"] == sector).sum()),
                "estimate_hours": float(point),
                "ci_95_low": float(low),
                "ci_95_high": float(high),
                "block_hours": block_hours,
                "bootstrap_replicates": n_bootstrap,
                "inference_status": "joint_temporal_block_bootstrap_descriptive"
            })
        for i, first in enumerate(sectors):
            for second in sectors[i + 1 :]:
                point = float(event_df.loc[event_df.sector == first, "half_life_hours"].median() -
                              event_df.loc[event_df.sector == second, "half_life_hours"].median())
                paired = np.asarray(boot[first]) - np.asarray(boot[second])
                paired = paired[np.isfinite(paired)]
                low, high = np.quantile(paired, [0.025, 0.975]) if len(paired) else (np.nan, np.nan)
                rows.append({
                    "record_type": "sector_median_difference",
                    "sector": first,
                    "comparison": f"{first} - {second}",
                    "episode_count": int((event_df.sector == first).sum() + (event_df.sector == second).sum()),
                    "estimate_hours": point,
                    "ci_95_low": float(low),
                    "ci_95_high": float(high),
                    "block_hours": block_hours,
                    "bootstrap_replicates": n_bootstrap,
                    "inference_status": "joint_temporal_block_bootstrap_descriptive"
                })
        result = pd.DataFrame(rows)
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv_path, index=False)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dependence-aware sector persistence analysis.")
    parser.add_argument("--dataset", default="data/processed/v2/1h/processed_dataset.pt")
    parser.add_argument("--sectors", default="data/external/sectors.csv")
    parser.add_argument("--output", default="reports/persistence_sector_block_bootstrap.csv")
    parser.add_argument("--block-hours", type=int, default=24)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = ShockPersistenceAnalyzer().bootstrap_sector_persistence(
        dataset_path=args.dataset,
        sectors_csv_path=args.sectors,
        output_csv_path=args.output,
        block_hours=args.block_hours,
        n_bootstrap=args.bootstrap_replicates,
        seed=args.seed,
    )
    logger.info("Wrote %d dependence-aware summary rows to %s", len(results), args.output)


if __name__ == "__main__":
    main()
