"""Futures stage: download Binance USD-M premium index (1h) and funding rates, build leakage-free node features.

Writes data/raw/binance_futures/{premium,funding}/<SYMBOL>/*.zip (checksum-verified), and for every build resolution
data/processed/v2/<res>/futures_features.pt = {"X": (T, N_graph, F) float32, "names", "graph_coins", coverage}.
Standardisation uses training rows only. Models opt in with `data.futures_features: true`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ...common.logger import get_logger
from ...data.config import HourlyConfig
from ...data.sources.binance import BinanceArchive
from ...data.sources.binance_futures import download_futures, futures_feature_table
from ...data.storage import load_pt, write_json

logger = get_logger(__name__)


def run(cfg: HourlyConfig, refresh: bool = False) -> None:
    b = cfg.binance
    arch = BinanceArchive(b.archive_url, b.list_url, "futures/um", "1h", b.retries, b.backoff_base_seconds)
    raw = cfg.raw_dir / "binance_futures"
    first = cfg.processed_dir(cfg.datasets.build_resolutions[0])
    coins = load_pt(first / "panel.pt")["graph_coins"]
    rep = download_futures(arch, list(coins), raw)
    logger.info(f"futures files: {rep['files']}, failures: {len(rep['failures'])}")
    if rep["failures"]:
        raise RuntimeError(f"futures download failures: {rep['failures'][:5]}")
    for res in cfg.datasets.build_resolutions:
        d = cfg.processed_dir(res)
        p = load_pt(d / "panel.pt")
        times = pd.DatetimeIndex(p["times"])
        if p["bar_hours"] != 1:                         # 2h bars: hourly features at the last hour of each bar
            times_1h = times + pd.Timedelta(hours=p["bar_hours"] - 1)
        else:
            times_1h = times
        train = p["sample_mask"]["train"].numpy()
        X, names = futures_feature_table(times_1h, list(p["graph_coins"]), raw, train)
        cov = {c: float(X[:, j, -1].mean()) for j, c in enumerate(p["graph_coins"])}
        cov_test = {c: float(X[p["split_id"].numpy() == 2, j, -1].mean()) for j, c in enumerate(p["graph_coins"])}
        torch.save({"X": torch.from_numpy(X), "names": names, "graph_coins": list(p["graph_coins"]),
                    "coverage": cov, "coverage_test": cov_test}, d / "futures_features.pt")
        write_json({"resolution": res, "names": names, "coverage_all": cov, "coverage_test": cov_test,
                    "files": rep["symbols"]}, cfg.reports_dir / f"futures_features_{res}.json")
        logger.info(f"{res}: futures features {X.shape}; mean coverage {np.mean(list(cov.values())):.2f}, "
                    f"test {np.mean(list(cov_test.values())):.2f}")
