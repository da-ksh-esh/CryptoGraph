import pytest
import pandas as pd
from cryptograph.data.feature_builder import FeatureBuilder

def test_feature_builder():
    builder = FeatureBuilder()
    df = pd.DataFrame({"close": [100.0, 102.0, 101.0, 105.0], "volume": [1000, 1200, 900, 1500]})
    features = builder.compute_features(df)
    assert "log_return" in features.columns
    assert "volume_ratio" in features.columns

def test_collector_load():
    from cryptograph.data.collector import CryptoDataCollector
    from pathlib import Path
    collector = CryptoDataCollector()
    data_path = Path("data/raw/crypto_top20_10y_1d.parquet")
    if data_path.exists():
        df = collector.load_dataset(data_path)
        assert not df.empty
        assert "Close" in df.columns.levels[0]

