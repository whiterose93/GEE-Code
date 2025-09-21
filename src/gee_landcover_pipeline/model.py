"""PyTorch model definitions for land-cover classification."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .config import EMBEDDING_BANDS, LANDCOVER_CLASSES, TrainingConfig


class LandcoverClassifier(nn.Module):
    """A simple multi-layer perceptron for embedding-based classification."""

    def __init__(
        self,
        hidden_dims: Sequence[int],
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        input_dim = len(EMBEDDING_BANDS)
        dims = [input_dim, *hidden_dims, num_classes]
        layers = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(embeddings)


def build_model(
    config: TrainingConfig,
    num_classes: int | None = None,
) -> LandcoverClassifier:
    """Instantiate a classifier from a :class:`TrainingConfig`."""

    if num_classes is None:
        num_classes = len(LANDCOVER_CLASSES)
    return LandcoverClassifier(
        hidden_dims=config.hidden_dims,
        num_classes=num_classes,
        dropout=config.dropout,
    )
