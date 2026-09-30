"""Typed, validated data configuration.

`load_config` reads ``configs/data/daily.yaml`` into a pydantic
``DataConfig``. Every path is resolved relative to the project root so the
pipeline behaves the same from any working directory.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data" / "daily.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GraphConfig(_Strict):
    correlation_metric: str = "pearson"
    rolling_window: int = 30
    threshold: float = 0.40
    top_k: int = 3
    use_adaptive_adjacency: bool = True
    directed_lag_threshold: float = 0.20
    directed_top_k: int = 0
    threshold_grid: list[float] = [0.3, 0.4, 0.5, 0.6]
    min_obs: int = 20


class PathsConfig(_Strict):
    raw_file: str = "data/raw/crypto_top20_10y_1d.parquet"
    processed_file: str = "data/processed/crypto_top20_processed.pt"
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    external_dir: str = "data/external"
    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"
    legacy_graph_dataset: str = "data/processed/crypto_graph_dataset.pt"


class WindowConfig(_Strict):
    start_date: date
    end_date: date
    reference_date: date

    @model_validator(mode="after")
    def _ordered(self) -> "WindowConfig":
        if not self.start_date < self.end_date:
            raise ValueError("window.start_date must precede window.end_date")
        return self


class Candidate(_Strict):
    coin: str
    coingecko: str
    yahoo: str | None = None
    coinmetrics: list[str] = Field(default_factory=list)
    category: Literal["large_cap", "legacy", "event"]


class UniverseConfig(_Strict):
    panel_size: int = 20
    panel_a_start_date: date | None = None  # None -> window.start_date
    panel_b_min_train_days: int = 0
    candidates: list[Candidate]
    stablecoins: list[str]
    wrapped_or_staked: list[str]
    rwa_or_commodity: list[str] = Field(default_factory=list)
    event_coins: list[str]
    include_event_coins_in_panel_b: bool = False

    @field_validator("candidates")
    @classmethod
    def _unique(cls, v: list[Candidate]) -> list[Candidate]:
        coins = [c.coin for c in v]
        if len(coins) != len(set(coins)):
            raise ValueError("duplicate coin in universe.candidates")
        return v

    @property
    def excluded_symbols(self) -> set[str]:
        return {s.lower() for s in (*self.stablecoins, *self.wrapped_or_staked, *self.rwa_or_commodity)}


class SourcesConfig(_Strict):
    priority: list[str]
    retries: int = 5
    backoff_base_seconds: float = 2.0
    request_pause_seconds: float = 0.5
    coinmetrics_url: str
    coingecko_markets_url: str
    coindesk_api_key_env: str = "COINDESK_API_KEY"
    identity_price_tolerance: float = 0.10


class QualityConfig(_Strict):
    min_coverage_panel_a: float = 0.98
    min_median_volume_usd: float = 1e6
    max_ffill_gap: int = 2
    xsource_close_tol: float = 0.02
    volume_spike_mult: float = 20.0
    volume_spike_window: int = 30
    stale_days: int = 3
    abs_logret_flag: float = 0.40


class SplitsConfig(_Strict):
    train: tuple[date, date]
    val: tuple[date, date]
    test: tuple[date, date]
    embargo_days: int = 30

    @model_validator(mode="after")
    def _chronological(self) -> "SplitsConfig":
        spans = [self.train, self.val, self.test]
        for a, b in spans:
            if a > b:
                raise ValueError("split start after end")
        for (_, e1), (s2, _) in zip(spans, spans[1:]):
            if not e1 < s2:
                raise ValueError("splits overlap or are out of order")
        return self

    def as_dict(self) -> dict[str, tuple[date, date]]:
        return {"train": self.train, "val": self.val, "test": self.test}


class TargetsConfig(_Strict):
    horizons: list[int] = [1, 3, 7, 14, 30]
    shock_vol_window: int = 30
    shock_z: float = 2.5
    contagion_k: list[int] = [1, 2, 3]


class FeaturesConfig(_Strict):
    vol_windows: list[int] = [7, 14, 30]
    volume_median_window: int = 30
    rsi_window: int = 14
    days_listed_cap: int | None = 730


class TGNConfig(_Strict):
    event_source: Literal["shock", "lag_corr"] = "shock"


class ScalingConfig(_Strict):
    method: Literal["robust"] = "robust"
    scope: Literal["per_coin", "global"] = "per_coin"
    clip: float = 5.0


class DataConfig(_Strict):
    # legacy
    source: str = "yfinance"
    tickers: list[str] = Field(default_factory=list)
    period: str | None = None  # deprecated, ignored
    interval: str = "1d"
    lookback_window: int = 30
    forecast_horizons: list[int] = [1, 3, 7, 14, 30]
    graph: GraphConfig = GraphConfig()
    paths: PathsConfig = PathsConfig()
    # new
    dataset_version: str = "v1"
    seed: int = 42
    window: WindowConfig
    universe: UniverseConfig
    sources: SourcesConfig
    quality: QualityConfig = QualityConfig()
    splits: SplitsConfig
    targets: TargetsConfig = TargetsConfig()
    features: FeaturesConfig = FeaturesConfig()
    tgn: TGNConfig = TGNConfig()
    scaling: ScalingConfig = ScalingConfig()

    # ---- resolved paths -------------------------------------------------------
    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def raw_dir(self) -> Path:
        return self.path(self.paths.raw_dir)

    @property
    def interim_dir(self) -> Path:
        return self.path(self.paths.interim_dir)

    @property
    def processed_dir(self) -> Path:
        """Versioned output directory, e.g. data/processed/v1."""
        return self.path(self.paths.processed_dir) / self.dataset_version

    @property
    def external_dir(self) -> Path:
        return self.path(self.paths.external_dir)

    @property
    def reports_dir(self) -> Path:
        return self.path(self.paths.reports_dir)

    @property
    def figures_dir(self) -> Path:
        return self.path(self.paths.figures_dir)

    def fingerprint(self) -> str:
        """SHA-256 of the canonical JSON dump; changes whenever any setting changes."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def load_config(path: str | Path | None = None) -> DataConfig:
    """Load and validate the data config (defaults to configs/data/daily.yaml)."""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DataConfig.model_validate(raw)
