"""Explanations of model predictions and their validation.

- tgn_capture: capture what the trained TGN attends to and re-run its prediction exactly with per-event weights
- tempme: TempME (Chen & Ying, NeurIPS 2023) ported to our TGN (temporal motifs, information-bottleneck explainer)
- fidelity: Fidelity+/-, characterization score, baseline explainers (random, recency, attention, optimised mask)
- attribution: integrated gradients on the TGN prediction head (feature-level explanations)
- run: the full explanation run behind notebook 04 (`model: tempme` experiments)
- edge_mask.EdgeMaskExplainer, validation.ExplanationValidator: LEGACY explainer / validator for the old snapshot
  model (not used in any result; the real TempME is `tempme.TempMEExplainer`)
"""

from .attribution import integrated_gradients
from .fidelity import fidelity_curve
from .tempme import TempMEExplainer, train_tempme
from .tgn_capture import Captured, ExplainableTGN, capture, load_trained

__all__ = ["TempMEExplainer", "train_tempme", "ExplainableTGN", "Captured", "capture", "load_trained",
           "fidelity_curve", "integrated_gradients"]
