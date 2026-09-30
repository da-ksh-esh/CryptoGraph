# Model Strategy, Comparative Zoo, & Serving Roadmap

## 1. Model Zoo & Roles

| Model | Architecture | Role & Hypothesis |
| :--- | :--- | :--- |
| **VAR-GFEVD** | Vector Autoregression + Generalized FEVD | Econometric reference baseline. Computes order-invariant systemic spillover matrices $\theta_{ij}^g(H)$ and NET directional transmitter/receiver rankings. |
| **Independent Cascade** | Stochastic epidemic graph diffusion | Propagation-process reference baseline. Tests whether classical probabilistic network diffusion explains crypto contagion cascades. |
| **LSTM Baseline** | Multi-layer Recurrent Neural Network | Non-graph temporal baseline. Evaluates performance of sequence history without spatial relational message passing. |
| **CryptoTGN** | Continuous-Time Dynamic Graph Neural Network | Primary experimental model. Fuses node memory states with spatial graph convolutions across dynamic correlation and lead-lag edges. |
| **DyGFormer** | Dynamic Graph Transformer (Co-occurrence Patching) | Advanced graph transformer baseline. Feasibility study conducted in Phase E6 before full continuous-time training integration. |
| **T-GAE** | Temporal Graph Autoencoder | Security and anomaly layer. Evaluates reconstruction error to flag corrupted or manipulated feeds without unsupported labels. |

---

## 2. Evaluation & Validation Protocol

- **Node Regression:** RMSE, MAE (return and volatility forecasting).
- **Node Classification:** AUROC, AUPRC (Average Precision prioritized due to severe class imbalance in extreme shocks).
- **Explanation Validation (TempME):**
  - **Fidelity+:** Prediction accuracy degradation under top-$K\%$ edge motif ablation.
  - **Characterization:** Sparsity and motif entropy.
  - **Rank Correlation:** Spearman rank correlation against VAR-GFEVD NET spillovers and PageRank centrality.
- **Network Science & Compression:**
  - Full vs. Top-$K$ vs. MST vs. Thresholded graphs.
  - Preservation of centrality and community modularity under structural compression.

---

## 3. Future Work: Production Serving & UI Roadmap

*(Deferred until empirical research models and checkpoints stabilize)*

1. **Production Serving Contract (`cryptograph-predict` CLI):**
   - Headless prediction command ingesting latest OHLCV stream.
   - Outputs: current contagion risk per coin, estimated propagation delay, and top-3 transmission source coins.
2. **Interactive Security Dashboard:**
   - Preservation of existing [dashboard/](file:///D:/GitHubProjects/CryptoGraph/dashboard/) components.
   - Interactive dynamic PyVis network, TempME explanation subgraph viewer, and real-time anomaly alerts.
