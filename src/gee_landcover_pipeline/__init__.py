"""Utilities to download Google Satellite Embedding data and classify land cover."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .config import DownloadConfig, InferenceConfig, LANDCOVER_CLASSES, TrainingConfig

__all__ = [
    "LANDCOVER_CLASSES",
    "TrainingConfig",
    "DownloadConfig",
    "InferenceConfig",
    "LandcoverPipeline",
]

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    from .pipeline import LandcoverPipeline


def __getattr__(name: str):
    if name == "LandcoverPipeline":
        from .pipeline import LandcoverPipeline as _Pipeline

        return _Pipeline
    raise AttributeError(name)
