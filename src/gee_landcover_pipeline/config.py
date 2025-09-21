"""Configuration and constants for the land-cover classification pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Sequence

DATASET_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

EMBEDDING_BANDS: Sequence[str] = tuple(f"A{i:02d}" for i in range(64))

LANDCOVER_CLASSES: Mapping[int, Mapping[str, str]] = {
    0: {"name": "Forest", "color": "#006400"},
    1: {"name": "Mangrove", "color": "#2ca25f"},
    2: {"name": "Built-up", "color": "#c82606"},
    3: {"name": "Bareland", "color": "#f7c9a9"},
    4: {"name": "Water", "color": "#0066ff"},
    5: {"name": "Vegetation", "color": "#bbd16a"},
    6: {"name": "Low Vegetation", "color": "#d1f0a3"},
    7: {"name": "Forest Plantation", "color": "#45c2a5"},
    8: {"name": "Palm Plantation", "color": "#f6e599"},
}

CLASS_NAME_TO_ID: Dict[str, int] = {
    meta["name"].lower(): class_id
    for class_id, meta in LANDCOVER_CLASSES.items()
}

SHAPEFILE_CLASS_MAP: Mapping[str, int] = {
    "forest": 0,
    "mangrove": 1,
    "built": 2,
    "bareland": 3,
    "water": 4,
    "vegetation": 5,
    "low vegetation": 6,
    "forest plantation": 7,
    "palm plantation": 8,
}


@dataclass
class DownloadConfig:
    """Parameters that control embedding downloads from Earth Engine."""

    years: Sequence[int] = (2023,)
    scale: float = 10.0
    target_property: str = "class_id"
    sample_per_polygon: int = 4000
    max_inference_pixels: int = 50000
    random_seed: int = 42
    tile_scale: int = 4


@dataclass
class TrainingConfig:
    """Hyper-parameters for the PyTorch land-cover classifier."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 50
    batch_size: int = 512
    validation_split: float = 0.2
    hidden_dims: Sequence[int] = (512, 256, 128)
    dropout: float = 0.2
    target_property: str = "class_id"
    random_state: int = 42
    device: str = "auto"
    model_output: Path = field(default_factory=lambda: Path("output") / "landcover_classifier.pt")


@dataclass
class InferenceConfig:
    """Configuration for inference exports."""

    output_dir: Path = field(default_factory=lambda: Path("output"))
    output_format: str = "geojson"
    probability: bool = True


DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("output")
