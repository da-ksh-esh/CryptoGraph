# CryptoGraph-Secure 🛡️⚡

A cybersecurity-focused deep learning framework that analyzes how price shocks propagate across interconnected cryptocurrencies using **Temporal Graph Neural Networks (TGNN)** and evaluates model resilience against adversarial market manipulation.

---

## 🌟 Key Features

- **📊 Dynamic Relationship Graph**: Rolling dynamic correlations & learned adaptive adjacency matrices between cryptocurrencies.
- **⚡ Cryptocurrency Shock Detection**: Extreme Value Theory (EVT) and jump detection for flash crashes and sudden liquidity drains.
- **🔗 Cross-Asset Spillover Analysis**: Quantifies spillover magnitude, transmission delay (lead-lag steps), and decay persistence.
- **🧠 Temporal Graph Neural Network**: GATv2 spatial attention + Bi-directional GRU for multi-horizon price/return prediction.
- **🚨 Cybersecurity & Anomaly Detection**: Unsupervised Temporal VGAE to flag wash trading, oracle manipulation, and corrupted feeds.
- **🛡️ Adversarial Robustness Testing**: Benchmarks model degradation under feature (PGD/FGSM) and graph topology (DICE) attacks.
- **📱 Interactive Dashboard**: Real-time Streamlit interface with interactive network graphs and perturbation stress-testing.

---

## 📂 Project Structure

```text
CryptoGraph/
├── configs/             # Hydra / YAML experiment & model configurations
├── data/                # Raw, processed, and external dataset storage
├── dashboard/           # Streamlit interactive monitoring dashboard
├── notebooks/           # Research and exploratory Jupyter notebooks
├── src/cryptograph/     # Core Python library
│   ├── common/          # Types, logging, shared utilities
│   ├── data/            # CCXT data collector, feature & graph builders
│   ├── models/          # GATv2, GCN, GRU, TCN, TGNN, and VGAE
│   ├── spillover/       # Shock detector, propagation tracker, attribution
│   ├── security/        # Anomaly scoring, adversarial attacks & defenses
│   └── pipelines/       # Training and evaluation orchestration pipelines
└── tests/               # Pytest unit and integration test suite
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Run Tests
```bash
pytest tests/
```

### 3. Launch Monitoring Dashboard
```bash
streamlit run dashboard/app.py
```
