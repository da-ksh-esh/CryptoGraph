"""Shared result types."""

from dataclasses import dataclass


@dataclass
class AnomalyAlert:
    timestamp: str
    asset: str
    anomaly_score: float
    threshold: float
    suspected_type: str  # currently always "unclassified" (the scorer does not identify causes)
    reconstruction_error: float
