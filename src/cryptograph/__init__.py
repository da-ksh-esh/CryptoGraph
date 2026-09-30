"""CryptoGraph-Secure: Dynamic Spatio-Temporal GNN for Cryptocurrency Shock Spillover & Security."""

# Windows: pyarrow (pulled in by pandas) ships an older msvcp140.dll; if it is
# loaded before torch, torch's c10.dll fails with WinError 1114. Importing torch
# first makes every `import cryptograph...` safe regardless of caller order.
try:  # pragma: no cover - platform workaround
    import torch  # noqa: F401
except ImportError:  # torch-free installs can still use the pure-pandas modules
    pass

__version__ = "0.1.0"
