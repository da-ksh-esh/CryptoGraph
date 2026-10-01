"""Script to compute causal network features across 24,024 hourly timesteps."""

from pathlib import Path
import pandas as pd
import numpy as np
import torch
from cryptograph.network.features import CausalNetworkFeatureExtractor

def main():
    dataset_path = Path("data/processed/v2/1h/processed_dataset.pt")
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)

    assets = data["asset_names"]
    timestamps = data["timestamps"]
    close = pd.DataFrame(data["close_prices"].numpy(), columns=assets)
    returns = np.log(close / close.shift(1)).fillna(0.0).values
    active_mask = data["active_mask"].numpy()

    # Load sectors
    sectors_df = pd.read_csv("data/external/sectors.csv")
    sectors_dict = dict(zip(sectors_df["coin"], sectors_df["sector"]))

    extractor = CausalNetworkFeatureExtractor(rolling_window=24, min_history=12, seed=42)
    feat_tensor, feat_names, summary_df = extractor.build_network_feature_tensors(
        returns_matrix=returns,
        active_mask=active_mask,
        asset_names=assets,
        sectors_dict=sectors_dict
    )

    out_file = Path("data/processed/v2/network_features.pt")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "network_features": feat_tensor,       # (T, N, num_network_features)
        "feature_names": feat_names,
        "timestamps": timestamps,
        "asset_names": assets,
        "modularity": summary_df["modularity"].values,
        "mst_norm_length": summary_df["mst_norm_length"].values,
        "sectors": [sectors_dict.get(a, "Other") for a in assets]
    }

    torch.save(payload, out_file)
    print(f"Saved network features to {out_file} ({out_file.stat().st_size / 1e6:.2f} MB)")
    print(f"Tensor shape: {feat_tensor.shape}, features: {feat_names}")

if __name__ == "__main__":
    main()
