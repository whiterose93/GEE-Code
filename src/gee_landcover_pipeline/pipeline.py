"""High level orchestration for the end-to-end land-cover pipeline."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence

import torch

from .config import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    DownloadConfig,
    InferenceConfig,
    TrainingConfig,
)
from .data import download_embeddings, initialize_earth_engine
from .inference import export_predictions, load_inference_data, run_inference
from .trainer import load_training_data, save_training_history, train_classifier

LOGGER = logging.getLogger(__name__)


class LandcoverPipeline:
    """Orchestrates the three stages of the requested workflow."""

    def __init__(
        self,
        shapefile_dir: Path,
        data_dir: Path | None = None,
        download_config: DownloadConfig | None = None,
        training_config: TrainingConfig | None = None,
        inference_config: InferenceConfig | None = None,
    ) -> None:
        self.shapefile_dir = Path(shapefile_dir)
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.download_config = download_config or DownloadConfig()
        self.training_config = training_config or TrainingConfig()
        self.inference_config = inference_config or InferenceConfig(
            output_dir=DEFAULT_OUTPUT_DIR
        )

    def run(self) -> None:
        """Execute download, training, and inference sequentially."""

        initialize_earth_engine()
        saved_files = download_embeddings(
            self.shapefile_dir, self.data_dir, self.download_config
        )
        training_files = [path for path in saved_files if path.stem.endswith("_train")]
        inference_files = [path for path in saved_files if path.stem.endswith("_inference")]

        model = None
        class_order: List[int] = []
        label_to_index: dict[int, int] = {}
        history_path = self.training_config.model_output.with_suffix(".history.json")

        if training_files:
            LOGGER.info("Training classifier with %d tables", len(training_files))
            training_data = load_training_data(
                training_files, self.training_config.target_property
            )
            model, history, label_to_index, class_order = train_classifier(
                training_data, self.training_config
            )
            self._save_model(model, label_to_index, class_order)
            save_training_history(history, history_path)
            LOGGER.info("Training history written to %s", history_path)
        else:
            LOGGER.warning(
                "No training shapefiles containing '%s' were found. Skipping training.",
                self.training_config.target_property,
            )

        if model is None:
            LOGGER.warning("Inference skipped because no model was trained.")
            return

        if not inference_files:
            LOGGER.warning("No inference shapefiles were processed; skipping inference stage.")
            return

        for inference_path in inference_files:
            LOGGER.info("Running inference for %s", inference_path.name)
            inference_data = load_inference_data([inference_path])
            predictions = run_inference(
                model, inference_data, class_order, self.inference_config
            )
            output_name = inference_path.stem.replace("_inference", "_predictions")
            export_predictions(predictions, self.inference_config, output_name)

    def _save_model(
        self,
        model: torch.nn.Module,
        label_to_index: dict[int, int],
        class_order: Sequence[int],
    ) -> None:
        """Persist the trained model weights and metadata."""

        self.training_config.model_output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "class_order": list(class_order),
            "label_to_index": {int(k): int(v) for k, v in label_to_index.items()},
            "training_config": asdict(self.training_config),
            "download_config": asdict(self.download_config),
        }
        torch.save(
            {
                "state_dict": model.state_dict(),
                "metadata": metadata,
            },
            self.training_config.model_output,
        )
        metadata_path = self.training_config.model_output.with_suffix(".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2))
        LOGGER.info("Model weights written to %s", self.training_config.model_output)
        LOGGER.info("Model metadata written to %s", metadata_path)
