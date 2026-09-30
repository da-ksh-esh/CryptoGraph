"""Explanations of model predictions and their validation.

- edge_mask.EdgeMaskExplainer: GNNExplainer-style edge mask (was misnamed TempME)
- validation.ExplanationValidator: Fidelity+, characterization, rank correlation
  against statistical connectedness (Diebold-Yilmaz NET ranking)
"""

from .edge_mask import EdgeMaskExplainer, TempME
from .validation import ExplanationValidator

__all__ = ["EdgeMaskExplainer", "TempME", "ExplanationValidator"]
