"""Robustness of the contagion models and anomaly scoring.

- stress: input noise, missing events and FGSM attacks on the final TGN vs gradient boosting (notebook 06)
- run: the tracked stress-test run (`model: robustness` experiments)
- perturbation.DualPerturbationTester, anomaly.AnomalyScorer: LEGACY helpers for the old snapshot model (not used in
  any result; the T-GAE behind notebook 06 is `models.tgae.SnapshotTGAE`)
"""

from .stress import auprc_by_k, tgn_test_pass

__all__ = ["tgn_test_pass", "auprc_by_k"]
