# Model Strategy, Comparative Zoo, Servicing Roadmap & Feasibility Reports

## 1. Model Zoo & Formal Methodological Roles

| Model | Architecture | Role & Scientific Hypothesis |
| :--- | :--- | :--- |
| **VAR-GFEVD** | Vector Autoregression + Generalized FEVD | Econometric reference baseline. Computes order-invariant systemic spillover matrices $\theta_{ij}^g(H)$ and NET directional transmitter/receiver rankings based on Pesaran & Shin (1998) and Diebold & Yilmaz (2012). |
| **Independent Cascade** | Stochastic epidemic graph diffusion | Propagation-process baseline. Tests whether classical probabilistic network diffusion across trailing correlation edges explains crypto contagion cascades without deep feature representations. |
| **CryptoLSTM** | Multi-layer Recurrent Neural Network | Non-graph temporal baseline. Evaluates the predictive power of multi-asset chronological sequence history in the absence of spatial relational message passing. |
| **CryptoTGN** | Continuous-Time Dynamic Graph Neural Network | Primary experimental model. Fuses node memory states with spatial graph convolutions across dynamic correlation and lead-lag edges to capture continuous-time shock propagation. |
| **DyGFormer** | Dynamic Graph Transformer (Co-occurrence Patching) | Advanced graph transformer baseline. Explores neighbor co-occurrence patching across temporal interaction sequences to evaluate self-attention mechanisms against recurrent memory. |
| **T-GAE** | Temporal Graph Autoencoder | Security and anomaly layer. Evaluates spatio-temporal reconstruction error to flag corrupted or manipulated exchange feeds without relying on unsupported synthetic labels. |

---

### Detailed Methodological Descriptions

#### 1. Econometric Baseline: VAR-GFEVD (Diebold & Yilmaz, 2012)
The Vector Autoregressive Generalized Forecast Error Variance Decomposition framework serves as the econometric ground truth for directional interconnectedness. Using an unconstrained VAR($p$) process on stationary log returns $R_t = \sum_{i=1}^p \Phi_i R_{t-i} + \epsilon_t$ with covariance matrix $\Sigma$, the $H$-step generalized forecast error variance share is computed as:
$$\theta_{ij}^g(H) = \frac{\sigma_{jj}^{-1} \sum_{h=0}^{H-1} \left( e_i' \Phi_h \Sigma e_j \right)^2}{\sum_{h=0}^{H-1} \left( e_i' \Phi_h \Sigma \Phi_h' e_i \right)}$$
Unlike Cholesky factorizations, generalized FEVD is strictly order-invariant. Row-normalized spillovers determine whether each coin is a net systemic transmitter ($\text{NET}_i > 0$) or receiver ($\text{NET}_i < 0$), providing an independent benchmark to validate learned neural attention and TempME explanations.

#### 2. Process Baseline: Independent Cascade Model (Kempe et al., 2003)
The Independent Cascade (IC) model represents the standard epidemiological diffusion process on networks. Given an empirical graph $G = (V, E)$ constructed from training correlations, directed edge activation probabilities $P(v \mid u)$ are calibrated from training shock co-occurrences. When coin $u$ is shocked, it receives a single attempt to trigger each inactive neighbor $v$ with probability $P(v \mid u)$. This baseline establishes whether complex multi-horizon contagion can be explained purely by classical epidemic spreading over static network backbones.

#### 3. Temporal Non-Graph Baseline: CryptoLSTM
The multi-layer LSTM model processes multivariate node features $X_{t-L:t}^{(i)}$ across rolling history windows $L \in \{24, 72\}$ hours independently per node. By capturing temporal autocorrelations, volatility clustering, and momentum without relational graph edges, CryptoLSTM isolates the exact performance margin contributed by spatial message passing and topological graph learning.

#### 4. Primary Model: CryptoTGN (Temporal Graph Network)
CryptoTGN implements continuous-time dynamic graph learning based on Rossi et al. (2020). For each node $i$, persistent memory vectors $s_i(t)$ evolve over continuous timestamps. When interaction events occur (shock triggers and correlation edges), messages $m_i(t) = \text{msg}(s_i(t^-), s_j(t^-), \Delta t, e_{ij}(t))$ are aggregated via a LastAggregator or MeanAggregator and fed into a GRU memory cell. Instantaneous node representations are computed via multi-head graph convolutions fusing node memory with contemporaneous technical features and directional lead-lag weights. A multi-horizon forecast head predicts future shock contagion probabilities across $k \in \{1, 3, 6, 12, 24\}$ hours.

#### 5. Dynamic Graph Transformer: DyGFormer (Yu et al., 2023)
DyGFormer is an advanced dynamic graph transformer architecture that replaces recurrent memory modules with neighbor co-occurrence patching. It extracts first-hop and second-hop temporal interaction sequences and applies cross-attention between source and destination patches to capture structural and temporal dependencies. Integrating DyGFormer tests whether attention-based neighbor co-occurrence improves high-frequency crypto shock forecasting relative to recurrent memory state maintenance.

#### 6. Spatio-Temporal Security Layer: T-GAE (Temporal Graph Autoencoder)
The T-GAE framework acts as an unsupervised anomaly detection engine. Comprising a spatio-temporal GCN-GRU encoder and decoder, it reconstructs input node features and dynamic adjacency matrices. Abnormal market events, oracle manipulation, and extreme flash crashes manifest as sudden surges in Frobenius reconstruction error $\|X_t - \hat{X}_t\|_F^2$. The model reports pure reconstruction anomalies without inventing unsupported categorical labels.

---

## 2. Official Implementation Feasibility Reports

### A. DyGLib Feasibility Report (`yule-BUAA/DyGLib`)

An audit of the official DyGLib repository (Yu et al., *DyGLib: A Unified Framework for Dynamic Graph Learning*) confirms the following:

1. **Task & Target Support:**
   - **Dynamic Link Prediction:** Native support in transductive and inductive modes.
   - **Dynamic Node Classification:** Fully supported natively via `train_node_classification.py`.
2. **Data & Event Stream Compatibility:**
   - DyGLib ingests continuous-time interaction logs formatted as CSV tables (`ml_{dataset}.csv`) with columns: `[u, i, ts, label, idx]`.
   - Node and edge feature matrices are ingested as companion NumPy arrays: `ml_{dataset}.npy` (edge features, shape $[E, D_{\text{edge}}]$) and `ml_{dataset}_node.npy` (node features, shape $[N, D_{\text{node}}]$).
3. **Required Adapter:**
   - A lightweight export adapter (`src/cryptograph/data/adapters/dyglib_adapter.py`) converting `data/processed/v2/1h/processed_dataset.pt` into DyGLib's `(u, i, ts, label, idx)` format.
   - **Estimated Implementation Effort:** $\approx 1$ engineering day.
   - **Recommendation:** Proceed with early feasibility tests in Phase E6 before committing to full training.

---

### B. TempME Feasibility Report (`Graph-and-Geometric-Learning/TempME`)

An audit of the official TempME repository (Chen & Ying, NeurIPS 2023, *Towards the Explainability of Temporal Graph Neural Networks via Motif Discovery*) confirms the following:

1. **Base Model Compatibility:**
   - TempME's official codebase directly integrates with PyTorch Geometric implementations of **TGN**, **TGAT**, and **GraphMixer**.
   - It trains a post-hoc temporal motif explainer that identifies the minimal, most influential temporal edge sequence $(u_k, v_k, t_k)$ explaining target node prediction $\hat{y}_i(t)$.
2. **Operational Requirements:**
   - Requires extracting 2-hop temporal computation subgraphs around prediction events $(v, t)$.
   - The explainer is parameterized as a lightweight temporal attention network trained with mutual information maximization and sparsity penalties.
3. **Quantitative Explanation Validation Protocol:**
   - **Fidelity+:** Evaluates prediction degradation when top explanatory motif edges are ablated:
     $$\text{Fidelity}^+(f, \mathcal{M}) = f(G) - f(G \setminus \mathcal{M})$$
   - **Fidelity-:** Evaluates prediction retention when only explanatory motifs are preserved:
     $$\text{Fidelity}^-(f, \mathcal{M}) = f(G) - f(\mathcal{M})$$
   - **Characterization Score:** Evaluates motif sparsity and temporal entropy.
   - **Econometric Validation (Spearman Rank Agreement):** Compares node explanation importances $\sum_{e \in \mathcal{M}_i} w_e$ against independent Diebold-Yilmaz NET spillover rankings and PageRank centralities via Spearman $\rho_{\text{rank}}$.
4. **Estimated Implementation Effort:** $\approx 2$ engineering days for wrapper and evaluation integration.

---

## 3. Evaluation & Validation Protocol

- **Node Regression (Return & Volatility Forecasting):** RMSE, MAE.
- **Node Classification (Contagion Shock Prediction):**
  - **AUPRC (Average Precision):** Prioritized as the primary metric due to severe class imbalance (<5% positive shock prevalence).
  - **AUROC, Precision, Recall, F1 Score.**
- **Network Compression Protocol (Phase 10):**
  - Compare Full Graph vs Top-$K$ vs MST vs Thresholded representations.
  - Measure edge reduction percentage, network density, centrality rank preservation, and downstream TGN predictive accuracy.

---

## 4. Serving Roadmap & Production Contract

*(Deferred until empirical research models and checkpoints stabilize)*

1. **Headless Prediction CLI (`cryptograph-predict`):**
   - Ingests latest Binance OHLCV stream.
   - Outputs: current contagion risk per coin, estimated propagation delay, and top-3 transmission source coins.
2. **Interactive Security Dashboard:**
   - Preservation of existing `dashboard/` components.
   - Dynamic network visualization, TempME explanation subgraph viewer, and real-time anomaly alerts.
