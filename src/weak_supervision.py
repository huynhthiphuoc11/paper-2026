"""Shim tương thích — dùng `from src.weak import ...`."""
from src.weak.aggregator import ProbabilisticLabelAggregator, compute_lf_correlation_matrix
from src.weak.pipeline import WeakLabelPipeline

# Alias cũ trong run_experiments.py (API rút gọn)
WeakSupervisionFramework = WeakLabelPipeline

__all__ = [
    "ProbabilisticLabelAggregator",
    "compute_lf_correlation_matrix",
    "WeakLabelPipeline",
    "WeakSupervisionFramework",
]
