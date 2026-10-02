"""Data processing pipeline CLI: builds resolution-aware datasets (daily v1 and hourly v2)."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import json
import yaml
import numpy as np
import pandas as pd
import torch

from ..common.logger import get_logger
from ..data.universe import TARGET_UNIVERSE_30
from ..data.splits import TemporalSplitter
from ..data.features import SeasonalityFilter, CollinearityPruner, ResolutionAwareFeatureBuilder
from ..data.targets import ContagionTargetBuilder
from ..data.dataloader import prepare_processed_dataset, load_processed_dataset

logger = get_logger(__name__)


def process_hourly_dataset(
    config_path: str = "configs/data/hourly.yaml",
    resolution_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Builds the research-grade hourly (or 2-hour) multi-asset dynamic graph dataset.
    Enforces strict chronological splitting, 7-day purge embargoes, train-only
    seasonality fitting, collinearity pruning (|rho| >= 0.90), and train-calibrated shock targets.
    """
    cfg_file = Path(config_path)
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    resolution = resolution_override or cfg.get("resolution", "1h")
    logger.info(f"Processing hourly dataset at resolution='{resolution}'...")

    raw_dir = Path(f"data/raw/binance/{resolution}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    # 1. Load aligned OHLCV DataFrames for all 30 coins
    btc_path = raw_dir / "BTC.parquet"
    if not btc_path.exists():
        raise FileNotFoundError(f"Reference BTC file not found at {btc_path}")

    btc_df = pd.read_parquet(btc_path)
    all_ts = btc_df.index
    T = len(all_ts)
    num_nodes = len(TARGET_UNIVERSE_30)
    logger.info(f"Loaded reference timeline: {T} bars from {all_ts.min()} to {all_ts.max()}")

    dfs: Dict[str, pd.DataFrame] = {}
    active_matrix = np.zeros((T, num_nodes), dtype=bool)
    close_matrix = np.zeros((T, num_nodes), dtype=np.float32)

    for idx, coin in enumerate(TARGET_UNIVERSE_30):
        coin_path = raw_dir / f"{coin}.parquet"
        if not coin_path.exists():
            logger.warning(f"Raw file missing for coin {coin} at {coin_path}, creating empty placeholder.")
            c_df = pd.DataFrame(index=all_ts, columns=["open", "high", "low", "close", "volume"]).fillna(0.0)
        else:
            c_df = pd.read_parquet(coin_path).reindex(all_ts)

        close_s = c_df["close"].ffill().bfill()
        active_s = c_df["close"].notnull() & (c_df["close"] > 0)

        dfs[coin] = c_df
        active_matrix[:, idx] = active_s.values
        close_matrix[:, idx] = close_s.fillna(0.0).values.astype(np.float32)

    # 2. Chronological Splits & 7-Day Purge Embargoes
    splitter = TemporalSplitter(
        train_start=cfg.get("start_date", "2024-01-01"),
        train_end="2025-08-31 23:59:59",
        val_start="2025-09-08 00:00:00",
        val_end="2026-03-15 23:59:59",
        test_start="2026-03-23 00:00:00",
        test_end=cfg.get("end_date", "2026-09-27") + " 23:59:59",
        embargo_days=cfg.get("embargo_days", 7)
    )
    splits = splitter.split_timestamps(all_ts)

    # 3. Fit Seasonality Filter strictly on Train
    season_filter = SeasonalityFilter()
    train_volumes = pd.DataFrame({
        coin: dfs[coin]["volume"].loc[splits.train_mask].fillna(0.0)
        for coin in TARGET_UNIVERSE_30
    })
    train_hl = pd.DataFrame({
        coin: (((dfs[coin]["high"] - dfs[coin]["low"]) / (dfs[coin]["close"] + 1e-8)).loc[splits.train_mask]).fillna(0.0)
        for coin in TARGET_UNIVERSE_30
    })
    season_filter.fit(train_volumes, train_hl)

    # 4. Feature Engineering
    builder = ResolutionAwareFeatureBuilder(resolution=resolution)
    raw_coin_features = {
        coin: builder.compute_single_asset_features(dfs[coin], seasonality_filter=season_filter)
        for coin in TARGET_UNIVERSE_30
    }
    candidate_feature_names = raw_coin_features["BTC"].columns.tolist()

    # 5. Collinearity Pruning (fit strictly on pooled train features)
    train_feat_dfs = [raw_coin_features[coin].loc[splits.train_mask] for coin in TARGET_UNIVERSE_30]
    pooled_train = pd.concat(train_feat_dfs, axis=0)

    pruner = CollinearityPruner(
        correlation_threshold=0.90,
        protected_features=["parkinson_volatility"],
        disallowed_features=["garman_klass", "garman_klass_volatility"]
    )
    pruner.fit(pooled_train)
    kept_features = pruner.kept_features_

    # Assemble 3D feature array (T, num_nodes, num_features)
    num_kept = len(kept_features)
    features_3d = np.zeros((T, num_nodes, num_kept), dtype=np.float32)

    for idx, coin in enumerate(TARGET_UNIVERSE_30):
        c_feats = raw_coin_features[coin][kept_features].fillna(0.0).values
        features_3d[:, idx, :] = c_feats.astype(np.float32)

    # Fit Feature Scaler strictly on train split
    train_slice = features_3d[splits.train_mask]
    builder.fit_scaler(train_slice, kept_features)
    scaled_features_3d = builder.transform_scaler(features_3d)

    # 6. Contagion Targets & Train-Calibrated Shocks
    # Extract Parkinson volatility for systemic shock calibration
    park_idx = kept_features.index("parkinson_volatility")
    vola_matrix = features_3d[:, :, park_idx]

    horizons = cfg.get("horizons_hours", [1, 3, 6, 12, 24])
    if resolution == "2h":
        # Distinct 2-hour bars representing 2h, 4h, 6h, 12h, and 24h horizons (no duplicates)
        horizons = [1, 2, 3, 6, 12]

    target_builder = ContagionTargetBuilder(horizons=horizons, shock_quantile=0.95)
    target_builder.fit_thresholds(vola_matrix, splits.train_mask)
    shocks_matrix = target_builder.identify_shocks(vola_matrix)

    shock_targets, return_targets, target_validity = target_builder.build_targets(
        close_prices=close_matrix,
        shocks=shocks_matrix,
        active_mask=active_matrix
    )

    # 7. Compute Shock-Triggered Event Statistics
    split_masks = {
        "train": splits.train_mask,
        "val": splits.val_mask,
        "test": splits.test_mask
    }
    event_counts = target_builder.compute_shock_events_count(shocks_matrix, active_matrix, split_masks)

    # 8. Assemble & Serialize Dataset
    out_dir = Path(f"data/processed/v2/{resolution}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "processed_dataset.pt"

    payload = {
        "features": torch.from_numpy(scaled_features_3d),
        "raw_features": torch.from_numpy(features_3d),
        "targets_shock": torch.from_numpy(shock_targets),
        "targets_return": torch.from_numpy(return_targets),
        "target_validity_mask": torch.from_numpy(target_validity),
        "active_mask": torch.from_numpy(active_matrix),
        "shocks": torch.from_numpy(shocks_matrix),
        "close_prices": torch.from_numpy(close_matrix),
        "timestamps": [str(ts) for ts in all_ts],
        "asset_names": TARGET_UNIVERSE_30,
        "feature_names": kept_features,
        "candidate_features": candidate_feature_names,
        "dropped_features": pruner.dropped_features_,
        "horizons": horizons,
        "split_indices": {
            "train_indices": splits.train_indices.tolist(),
            "val_indices": splits.val_indices.tolist(),
            "test_indices": splits.test_indices.tolist(),
            "train_mask": splits.train_mask.tolist(),
            "val_mask": splits.val_mask.tolist(),
            "test_mask": splits.test_mask.tolist(),
            "embargo_mask": splits.embargo_mask.tolist()
        },
        "metadata": {
            "resolution": resolution,
            "num_timesteps": T,
            "num_assets": num_nodes,
            "num_features": num_kept,
            "horizons": horizons,
            "shock_quantile": 0.95,
            "shock_thresholds": target_builder.thresholds_.tolist(),
            "event_counts": event_counts,
            "scaler_mean": builder.mean_.tolist(),
            "scaler_std": builder.scale_.tolist()
        }
    }

    torch.save(payload, out_file)
    logger.info(f"Saved processed dataset to {out_file} ({out_file.stat().st_size / 1e6:.2f} MB)")

    # Save metadata JSON
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload["metadata"], f, indent=2)

    return payload["metadata"]


def run_smoke_test() -> bool:
    """Verifies that processed files can be loaded, have valid shapes, and no unexpected NaNs."""
    logger.info("Running smoke test across dataset artifacts...")

    # 1. Daily baseline check
    daily_pt = Path("data/processed/crypto_graph_dataset.pt")
    if not daily_pt.exists():
        logger.error(f"Missing daily dataset: {daily_pt}")
        return False
    daily_data = torch.load(daily_pt, map_location="cpu", weights_only=False)
    assert "features" in daily_data and "targets" in daily_data
    logger.info(f"Daily v1 dataset OK: features={daily_data['features'].shape}, targets={daily_data['targets'].shape}")

    # 2. Hourly v2 check
    hourly_pt = Path("data/processed/v2/1h/processed_dataset.pt")
    if not hourly_pt.exists():
        logger.error(f"Missing hourly dataset: {hourly_pt}")
        return False
    h_data = torch.load(hourly_pt, map_location="cpu", weights_only=False)
    assert "features" in h_data and "targets_shock" in h_data
    assert not torch.isnan(h_data["features"]).any(), "Hourly features contain NaN!"
    logger.info(
        f"Hourly v2 dataset OK: features={h_data['features'].shape}, "
        f"targets_shock={h_data['targets_shock'].shape}, "
        f"shocks={h_data['shocks'].shape}"
    )

    logger.info("All smoke tests passed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="CryptoGraph data preprocessing pipeline.")
    parser.add_argument("--config", type=str, default="configs/data/hourly.yaml", help="Path to config file")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "build", "verify", "smoke"], help="Pipeline stage")
    parser.add_argument("--resolution", type=str, default=None, choices=["1h", "2h", "1d"], help="Observation resolution")
    args = parser.parse_args()

    if args.stage == "smoke":
        success = run_smoke_test()
        exit(0 if success else 1)

    # Determine daily vs hourly based on config or resolution
    is_daily = (args.resolution == "1d") or ("daily" in args.config)

    if is_daily:
        logger.info(f"Running daily data pipeline with config: {args.config}")
        prepare_processed_dataset(
            raw_parquet_path="data/raw/crypto_top20_10y_1d.parquet",
            output_path="data/processed/crypto_graph_dataset.pt",
            config_path=args.config
        )
    else:
        logger.info(f"Running hourly data pipeline with config: {args.config}")
        # Build 1h
        meta_1h = process_hourly_dataset(config_path=args.config, resolution_override="1h")
        # Build 2h
        meta_2h = process_hourly_dataset(config_path=args.config, resolution_override="2h")

        logger.info("Hourly (1h and 2h) preprocessing completed successfully.")


if __name__ == "__main__":
    main()
