"""Inference utilities for the land-cover classifier."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

import geopandas as gpd
import pandas as pd
import torch

from .config import EMBEDDING_BANDS, InferenceConfig, LANDCOVER_CLASSES
from .model import LandcoverClassifier

LOGGER = logging.getLogger(__name__)


def load_inference_data(files: Sequence[Path]) -> gpd.GeoDataFrame:
    """Load GeoParquet inference files into a GeoDataFrame."""

    frames: List[gpd.GeoDataFrame] = []
    for path in files:
        table = gpd.read_parquet(path)
        missing = [band for band in EMBEDDING_BANDS if band not in table.columns]
        if missing:
            raise ValueError(f"Inference file {path} missing expected columns: {missing}")
        frames.append(table)
    if not frames:
        raise ValueError("No inference tables found. Run the downloader first.")
    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326")
    return combined


def run_inference(
    model: LandcoverClassifier,
    data: gpd.GeoDataFrame,
    class_order: Sequence[int],
    config: InferenceConfig,
) -> gpd.GeoDataFrame:
    """Run inference and append prediction columns to ``data``."""

    model.eval()
    device = next(model.parameters()).device
    embeddings = torch.from_numpy(data[EMBEDDING_BANDS].to_numpy(dtype=np.float32)).to(device)
    with torch.no_grad():
        logits = model(embeddings)
        probabilities = torch.softmax(logits, dim=1)
        confidence, pred_indices = probabilities.max(dim=1)

    class_ids = [class_order[idx] for idx in pred_indices.cpu().numpy()]
    names = [LANDCOVER_CLASSES.get(cid, {}).get("name", str(cid)) for cid in class_ids]
    colors = [LANDCOVER_CLASSES.get(cid, {}).get("color", "#000000") for cid in class_ids]

    result = data.copy()
    result["predicted_class"] = class_ids
    result["predicted_name"] = names
    result["predicted_color"] = colors
    result["confidence"] = confidence.cpu().numpy()

    if config.probability:
        probs = probabilities.cpu().numpy()
        for idx, class_id in enumerate(class_order):
            result[f"prob_{class_id}"] = probs[:, idx]

    return result


def export_predictions(data: gpd.GeoDataFrame, config: InferenceConfig, name: str) -> Path:
    """Save predictions to disk in the configured format."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.output_format == "geojson":
        output_path = config.output_dir / f"{name}.geojson"
        data.to_file(output_path, driver="GeoJSON")
    elif config.output_format == "gpkg":
        output_path = config.output_dir / f"{name}.gpkg"
        data.to_file(output_path, driver="GPKG")
    elif config.output_format == "parquet":
        output_path = config.output_dir / f"{name}.parquet"
        data.to_parquet(output_path)
    else:
        raise ValueError(f"Unsupported output format: {config.output_format}")
    LOGGER.info("Predictions written to %s", output_path)
    return output_path
