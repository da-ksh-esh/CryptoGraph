"""Robustness tests and anomaly scoring (optional stage).

- perturbation.DualPerturbationTester: feature/topology perturbation of predictions and explanations
- attacks.AdversarialAttacker: FGSM feature attack (TODO: adapt to models returning tuples)
- anomaly.AnomalyScorer: reconstruction-error anomaly flags (unclassified)
"""

from .anomaly import AnomalyScorer
from .attacks import AdversarialAttacker
from .perturbation import DualPerturbationTester

__all__ = ["AnomalyScorer", "AdversarialAttacker", "DualPerturbationTester"]
