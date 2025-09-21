"""Utilities for training the land-cover classifier."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import EMBEDDING_BANDS, TrainingConfig
from .model import LandcoverClassifier, build_model

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingHistoryEntry:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float


class EmbeddingDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """A PyTorch dataset for satellite embeddings."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        if len(embeddings) != len(labels):
            raise ValueError("Embeddings and labels must have the same length.")
        self.embeddings = torch.from_numpy(embeddings.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - trivial
        return self.embeddings[idx], self.labels[idx]


def load_training_data(files: Sequence[Path], target_property: str) -> pd.DataFrame:
    """Load GeoParquet training files and return a clean Pandas DataFrame."""

    frames: List[pd.DataFrame] = []
    for path in files:
        table = gpd.read_parquet(path)
        if target_property not in table.columns:
            LOGGER.warning("File %s missing target column '%s'", path, target_property)
            continue
        columns = list(EMBEDDING_BANDS) + [target_property]
        missing = [col for col in columns if col not in table.columns]
        if missing:
            raise ValueError(f"File {path} lacks expected columns: {missing}")
        cleaned = table.dropna(subset=columns)
        frames.append(cleaned[columns])
    if not frames:
        raise ValueError("No valid training tables found. Ensure shapefiles contained class labels.")
    return pd.concat(frames, ignore_index=True)


def _encode_labels(labels: Iterable[int]) -> Tuple[np.ndarray, Dict[int, int], List[int]]:
    """Convert arbitrary class identifiers into a contiguous range of indices."""

    unique_classes = sorted(set(int(label) for label in labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_classes)}
    encoded = np.array([label_to_index[int(label)] for label in labels], dtype=np.int64)
    return encoded, label_to_index, unique_classes


def _select_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def train_classifier(
    data: pd.DataFrame,
    config: TrainingConfig,
) -> Tuple[LandcoverClassifier, List[TrainingHistoryEntry], Dict[int, int], List[int]]:
    """Train the classifier and return the trained model and metadata."""

    labels = data[config.target_property].astype(int).tolist()
    embeddings = data[list(EMBEDDING_BANDS)].to_numpy(dtype=np.float32)
    encoded_labels, label_to_index, class_order = _encode_labels(labels)
    num_classes = len(label_to_index)
    if num_classes == 0:
        raise ValueError("Training data does not contain any class labels.")

    indices = np.arange(len(data))
    if config.validation_split > 0 and len(data) > 1:
        test_size = max(1, int(len(data) * config.validation_split))
        if test_size >= len(data):
            test_size = len(data) - 1
        stratify = encoded_labels if len(set(encoded_labels)) > 1 else None
        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=config.random_state,
            stratify=stratify,
        )
    else:
        train_idx, val_idx = indices, np.array([], dtype=int)

    train_dataset = EmbeddingDataset(embeddings[train_idx], encoded_labels[train_idx])
    val_dataset = EmbeddingDataset(embeddings[val_idx], encoded_labels[val_idx]) if len(val_idx) else None

    device = _select_device(config.device)
    model = build_model(config, num_classes=num_classes).to(device)

    class_counts = np.bincount(encoded_labels, minlength=num_classes)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size) if val_dataset else None

    history: List[TrainingHistoryEntry] = []
    best_state: Dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_dataset)

        val_loss = 0.0
        val_accuracy = 0.0
        if val_loader:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    predictions = outputs.argmax(dim=1)
                    correct += (predictions == targets).sum().item()
                    total += targets.size(0)
            val_loss = val_loss / max(total, 1)
            val_accuracy = correct / max(total, 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
        else:
            best_state = model.state_dict()

        history.append(
            TrainingHistoryEntry(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, label_to_index, class_order


def save_training_history(history: Sequence[TrainingHistoryEntry], path: Path) -> None:
    """Persist the training history to a JSON file."""

    serializable = [entry.__dict__ for entry in history]
    path.write_text(json.dumps(serializable, indent=2))
