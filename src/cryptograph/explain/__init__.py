"""Explanations of model predictions and their validation.

- tgn_capture: capture what the trained TGN attends to and re-run its prediction exactly with per-event weights
- tempme: TempME (Chen & Ying, NeurIPS 2023) ported to our TGN (temporal motifs, information-bottleneck explainer)
- fidelity: Fidelity+/-, characterization score, baseline explainers (random, recency, attention, optimised mask)
- attribution: integrated gradients on the TGN prediction head (feature-level explanations)
- run: the full explanation run behind notebook 04 (`model: tempme` experiments)
- edge_mask.EdgeMaskExplainer: older GNNExplainer-style edge mask for the snapshot model (was misnamed TempME)
- validation.ExplanationValidator: older snapshot-model validator
"""

from .edge_mask import EdgeMaskExplainer, TempME
from .validation import ExplanationValidator

__all__ = ["EdgeMaskExplainer", "TempME", "ExplanationValidator"]
