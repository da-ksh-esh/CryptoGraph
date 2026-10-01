"""Script to execute the Phase 10 Network Compression research experiment across the 30-coin universe."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from cryptograph.network.compression import NetworkCompressor

def main():
    dataset_path = Path("data/processed/v2/1h/processed_dataset.pt")
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)

    assets = data["asset_names"]
    close = pd.DataFrame(data["close_prices"].numpy(), index=pd.to_datetime(data["timestamps"]), columns=assets)
    test_mask = np.array(data["split_indices"]["test_mask"])

    # Evaluate on the frozen test period
    test_close = close.loc[test_mask]
    test_ret = np.log(test_close / test_close.shift(1)).dropna()
    corr = test_ret.corr().values

    print(f"Running Network Compression Benchmark on 30 coins over test window ({len(test_ret)} hours)...")
    df_results = NetworkCompressor.run_benchmark(corr, assets, seed=42)

    rep_path = Path("reports/network_compression_experiment.csv")
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(rep_path, index=False)

    print("\n=== NETWORK COMPRESSION BENCHMARK RESULTS ===")
    print(df_results.to_string(index=False))
    print(f"\nSaved results to {rep_path}")

if __name__ == "__main__":
    main()
