"""Legacy data collection facade (multi-ticker OHLCV frame).

The audited, reproducible pipeline lives in `cryptograph.data.sources` /
`cryptograph.pipelines.build_data` (`cryptograph-data --stage ...`).
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import argparse
import yaml
import pandas as pd

from ..common.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TICKERS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "DOGE-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "DOT-USD",
    "TRX-USD", "BCH-USD", "LTC-USD", "NEAR-USD", "XLM-USD",
    "ETC-USD", "ATOM-USD", "XMR-USD", "FIL-USD", "SHIB-USD"
]

class CryptoDataCollector:
    """Collects historical OHLCV cryptocurrency data via Yahoo Finance or CCXT."""

    def __init__(self, tickers: Optional[List[str]] = None, config_path: Optional[str | Path] = None):
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                self.tickers = cfg.get("tickers", DEFAULT_TICKERS)
        else:
            self.tickers = tickers or DEFAULT_TICKERS

    def fetch_yfinance_ohlcv(
        self,
        tickers: Optional[List[str]] = None,
        period: str = "10y",
        interval: str = "1d",
        auto_adjust: bool = False,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches historical multi-ticker daily candlestick data from Yahoo Finance.

        Legacy entry point. The audited pipeline is `cryptograph-data` (see README);
        this method now fetches one ticker per request with retries (batch downloads
        silently drop failing tickers) over a fixed [start, end] window. `period` is
        only used when start/end are omitted and is not reproducible across runs.

        Returns:
            pd.DataFrame: Multi-index DataFrame with (Price Metric, Ticker) columns.
        """
        from datetime import date, timedelta

        from .sources.yahoo import YahooSource

        symbols = tickers or self.tickers
        if interval != "1d":
            raise ValueError("only daily data is supported")
        end_d = date.fromisoformat(end) if end else date.today() - timedelta(days=1)
        years = int(period.rstrip("y")) if period.endswith("y") else 10
        start_d = date.fromisoformat(start) if start else end_d.replace(year=end_d.year - years)
        logger.info(f"Downloading {len(symbols)} tickers {start_d}..{end_d} from Yahoo Finance (one per request)...")

        src = YahooSource(start_d, end_d)
        frames = {}
        for sym in symbols:
            try:
                d = src.fetch(sym)
            except RuntimeError as e:
                logger.warning(f"{sym}: {e}")
                continue
            if auto_adjust and "adj_close" in d:
                d["close"] = d["adj_close"]
            frames[sym] = d.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
        if not frames:
            raise ValueError("Downloaded dataset is empty. Check ticker symbols or network connection.")
        df = pd.concat(frames, axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1)
        df.columns = df.columns.remove_unused_levels()

        if df.empty:
            raise ValueError("Downloaded dataset is empty. Check ticker symbols or network connection.")

        logger.info(f"Successfully downloaded dataset with shape {df.shape} from {df.index[0]} to {df.index[-1]}.")
        return df

    def save_dataset(self, df: pd.DataFrame, file_path: str | Path) -> Path:
        """
        Saves the downloaded DataFrame to disk in Parquet or CSV format.
        """
        out_path = Path(file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.suffix == ".parquet":
            # Flatten multi-index columns for parquet compatibility if needed, or serialize directly
            df.to_parquet(out_path)
            logger.info(f"Saved dataset to Parquet: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
        elif out_path.suffix == ".csv":
            df.to_csv(out_path)
            logger.info(f"Saved dataset to CSV: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
        else:
            raise ValueError(f"Unsupported file format '{out_path.suffix}'. Use .parquet or .csv.")

        return out_path

    def load_dataset(self, file_path: str | Path) -> pd.DataFrame:
        """Loads dataset from Parquet or CSV."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path, index_col=0, parse_dates=True, header=[0, 1])
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

    def export_per_asset_csv(self, df: pd.DataFrame, output_dir: str | Path, tickers: Optional[List[str]] = None) -> List[Path]:
        """
        Splits multi-ticker DataFrame into individual clean CSV files per asset in output_dir.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        symbols = tickers or self.tickers
        saved_paths = []

        metrics = ["Open", "High", "Low", "Close", "Volume"]

        for sym in symbols:
            coin_code = sym.replace("-USD", "")
            try:
                coin_dict = {}
                for m in metrics:
                    if m in df.columns.levels[0] and sym in df[m].columns:
                        coin_dict[m.lower()] = df[m][sym]
                if coin_dict:
                    coin_df = pd.DataFrame(coin_dict).dropna(subset=["close"])
                    file_path = out_dir / f"{coin_code}.csv"
                    coin_df.to_csv(file_path)
                    saved_paths.append(file_path)
            except Exception as e:
                logger.warning(f"Could not export individual CSV for {sym}: {e}")

        logger.info(f"Exported {len(saved_paths)} individual asset CSV files to {out_dir}")
        return saved_paths

    def get_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates coverage summary for each asset."""
        summary = {
            "start_date": str(df.index[0]),
            "end_date": str(df.index[-1]),
            "total_timesteps": len(df),
            "assets": {}
        }
        if "Close" in df.columns.levels[0]:
            close_df = df["Close"]
            for col in close_df.columns:
                valid_series = close_df[col].dropna()
                summary["assets"][col] = {
                    "valid_rows": int(len(valid_series)),
                    "first_available": str(valid_series.index[0]) if len(valid_series) > 0 else None,
                    "last_available": str(valid_series.index[-1]) if len(valid_series) > 0 else None,
                    "missing_pct": round(float(close_df[col].isnull().mean() * 100), 2)
                }
        return summary


def main():
    parser = argparse.ArgumentParser(description="Collect top-20 cryptocurrency historical OHLCV data.")
    parser.add_argument("--config", type=str, default="configs/data/crypto_default.yaml", help="Path to config file")
    parser.add_argument("--period", type=str, default="10y", help="Historical period (e.g., 10y, 5y)")
    parser.add_argument("--interval", type=str, default="1d", help="Candlestick interval (e.g., 1d, 1h)")
    parser.add_argument("--out-dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--export-per-asset", action="store_true", help="Also export per-coin CSVs")
    args = parser.parse_args()

    collector = CryptoDataCollector(config_path=args.config)
    data = collector.fetch_yfinance_ohlcv(period=args.period, interval=args.interval)

    # Save consolidated dataset
    parquet_path = Path(args.out_dir) / f"crypto_top20_{args.period}_{args.interval}.parquet"
    csv_path = Path(args.out_dir) / f"crypto_top20_{args.period}_{args.interval}.csv"
    collector.save_dataset(data, parquet_path)
    collector.save_dataset(data, csv_path)

    # Summary
    summary = collector.get_summary(data)
    logger.info(f"Dataset date range: {summary['start_date']} to {summary['end_date']} ({summary['total_timesteps']} steps)")

    # Optionally export per-coin individual CSVs
    if args.export_per_asset:
        per_asset_dir = Path(args.out_dir) / "per_asset"
        collector.export_per_asset_csv(data, per_asset_dir)


if __name__ == "__main__":
    main()
