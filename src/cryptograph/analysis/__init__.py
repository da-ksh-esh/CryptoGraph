"""EDA and statistical analysis. Reads pipeline outputs; never feeds model inputs."""

from .connectedness import dy_connectedness, generalized_fevd, rolling_total_spillover, spillover_table
from .var_spillover import VARSpilloverBaseline

__all__ = ["generalized_fevd", "spillover_table", "dy_connectedness", "rolling_total_spillover", "VARSpilloverBaseline"]
