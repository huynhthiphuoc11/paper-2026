from __future__ import annotations

import torch
from torch import nn

from src.experiments import FEATURE_COLUMNS


class LinearScorer(nn.Module):
    """Shared scorer capacity for pointwise, pairwise, and listwise objectives."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(len(FEATURE_COLUMNS), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)
