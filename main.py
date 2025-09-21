"""Command line entry-point for the Google Satellite Embedding land-cover pipeline."""
from __future__ import annotations

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gee_landcover_pipeline.config import DownloadConfig, InferenceConfig, TrainingConfig



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download Google Satellite Embedding vectors for local shapefiles, train a "
            "land-cover classifier, and export predictions."
        )
    )
    parser.add_argument(
        "--shapefile-dir",
        type=Path,
        default=Path("shp"),
        help="Directory that contains input shapefiles.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory where intermediate parquet files will be stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where the trained model and predictions will be saved.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2023],
        help="List of years to download from the embedding collection.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="Pixel resolution (in meters) used for sampling embeddings.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=4000,
        help="Number of samples per class to request during training data creation.",
    )
    parser.add_argument(
        "--max-inference-pixels",
        type=int,
        default=50000,
        help="Maximum number of inference samples per shapefile.",
    )
    parser.add_argument(
        "--target-property",
        type=str,
        default="class_id",
        help="Name of the attribute column that stores class identifiers in the training shapefiles.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of training epochs for the neural network.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of the training data reserved for validation.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help="Hidden layer sizes for the neural network.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability applied between hidden layers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device to use for training (e.g. 'cpu', 'cuda', or 'auto').",
    )
    parser.add_argument(
        "--tile-scale",
        type=int,
        default=4,
        help="Tile scale parameter passed to Earth Engine sampling operations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling and training reproducibility.",
    )
    parser.add_argument(
        "--output-format",
        choices=["geojson", "gpkg", "parquet"],
        default="geojson",
        help="File format for exported predictions.",
    )
    parser.add_argument(
        "--no-probabilities",
        action="store_true",
        help="If set, skip exporting per-class probabilities with the predictions.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=None,
        help="Custom path for the trained model weights (default: <output-dir>/landcover_classifier.pt).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (e.g. INFO, DEBUG).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        from gee_landcover_pipeline.pipeline import LandcoverPipeline
    except ModuleNotFoundError as exc:
        parser.error(f"Dependensi belum terpasang: {exc}")

    model_output = args.model_output or args.output_dir / "landcover_classifier.pt"

    download_config = DownloadConfig(
        years=tuple(args.years),
        scale=args.scale,
        target_property=args.target_property,
        sample_per_polygon=args.sample_per_class,
        max_inference_pixels=args.max_inference_pixels,
        random_seed=args.seed,
        tile_scale=args.tile_scale,
    )

    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        target_property=args.target_property,
        random_state=args.seed,
        device=args.device,
        model_output=model_output,
    )

    inference_config = InferenceConfig(
        output_dir=args.output_dir,
        output_format=args.output_format,
        probability=not args.no_probabilities,
    )

    pipeline = LandcoverPipeline(
        shapefile_dir=args.shapefile_dir,
        data_dir=args.data_dir,
        download_config=download_config,
        training_config=training_config,
        inference_config=inference_config,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
