"""Models used in the results: ContagionTGN (main), DyGFormer, CryptoLSTM, SnapshotTGAE.

Legacy (not used in any result; kept for the older snapshot-model tests): `tgn.CryptoTGN`, `tgae.TemporalGAE`.
The VAR / Diebold-Yilmaz baseline lives in `cryptograph.analysis` (statistical, not trained); the gradient-boosting
baseline is trained in `cryptograph.training.gbm_task`.
"""

from .dygformer import DyGFormer
from .lstm import CryptoLSTM
from .tgae import SnapshotTGAE
from .tgn import ContagionTGN

__all__ = ["ContagionTGN", "DyGFormer", "CryptoLSTM", "SnapshotTGAE"]
