"""Shim tương thích — dùng `from src.data import ...`."""
from src.data.loader import *  # noqa: F401,F403
from src.data.loader import FEATURE_COLS, CVJobDatasetLoader, RealKaggleDatasetAdapter, load_dataset

__all__ = [
    "FEATURE_COLS",
    "CVJobDatasetLoader",
    "RealKaggleDatasetAdapter",
    "load_dataset",
]
