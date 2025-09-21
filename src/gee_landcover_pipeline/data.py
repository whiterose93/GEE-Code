"""Functions for downloading Google Satellite Embedding vectors for training and inference."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

import ee
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, shape

from .config import DATASET_ID, EMBEDDING_BANDS, DownloadConfig

LOGGER = logging.getLogger(__name__)


def initialize_earth_engine() -> None:
    """Initialize the Earth Engine API if it has not been initialized."""

    try:
        ee.Initialize()
    except Exception as exc:  # pragma: no cover - executed when EE not yet initialized
        LOGGER.info("Authenticating Earth Engine client: %s", exc)
        ee.Authenticate()
        ee.Initialize()


def list_shapefiles(shapefile_dir: Path) -> List[Path]:
    """Return a sorted list of shapefile paths inside ``shapefile_dir``."""

    shapefile_dir = Path(shapefile_dir)
    return sorted(shp for shp in shapefile_dir.glob("*.shp"))


def load_geodataframe(path: Path) -> gpd.GeoDataFrame:
    """Load a shapefile and ensure it is in EPSG:4326."""

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Shapefile '{path}' does not contain any features.")
    if gdf.crs is None:
        raise ValueError(
            f"Shapefile '{path}' lacks a coordinate reference system."
            " Assign one before running the pipeline."
        )
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


def geodataframe_to_feature_collection(
    gdf: gpd.GeoDataFrame, properties: Sequence[str] | None = None
) -> ee.FeatureCollection:
    """Convert a GeoDataFrame into an Earth Engine FeatureCollection."""

    features: List[ee.Feature] = []
    columns = list(gdf.columns)
    if properties is not None:
        properties = [prop for prop in properties if prop in columns]
    for _, row in gdf.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        geo_json = geometry.__geo_interface__
        props = {
            col: row[col]
            for col in columns
            if col != "geometry" and (properties is None or col in properties)
        }
        features.append(ee.Feature(ee.Geometry(geo_json), props))
    if not features:
        raise ValueError("No valid geometries found to convert into a FeatureCollection.")
    return ee.FeatureCollection(features)


def _embedding_image_for_year(year: int, geometry: ee.Geometry) -> ee.Image:
    """Fetch the embedding image for ``year`` that intersects ``geometry``."""

    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    collection = (
        ee.ImageCollection(DATASET_ID)
        .filterDate(start, end)
        .filterBounds(geometry)
    )
    size = collection.size().getInfo()
    if size == 0:
        raise ValueError(
            f"No embedding imagery found for year {year}. "
            "Check that the region intersects the dataset extent."
        )
    return ee.Image(collection.first()).select(EMBEDDING_BANDS)


def _feature_collection_to_geodataframe(collection: ee.FeatureCollection) -> gpd.GeoDataFrame:
    """Convert an Earth Engine FeatureCollection to a GeoDataFrame."""

    info = collection.getInfo()
    features = info.get("features", [])
    if not features:
        return gpd.GeoDataFrame(columns=list(EMBEDDING_BANDS), geometry=[], crs="EPSG:4326")

    rows = []
    geometries = []
    for feature in features:
        props = feature.get("properties", {}).copy()
        geom = feature.get("geometry")
        if geom:
            geom_obj = shape(geom)
            geometries.append(geom_obj)
            centroid = geom_obj.representative_point()
            props.setdefault("longitude", centroid.x)
            props.setdefault("latitude", centroid.y)
        else:
            geometries.append(Point(float("nan"), float("nan")))
            props.setdefault("longitude", float("nan"))
            props.setdefault("latitude", float("nan"))
        rows.append(props)
    df = pd.DataFrame(rows)
    return gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")


def _sample_training_data(
    image: ee.Image,
    features: ee.FeatureCollection,
    gdf: gpd.GeoDataFrame,
    config: DownloadConfig,
) -> gpd.GeoDataFrame:
    """Sample embeddings for training using stratified sampling per class."""

    class_values = sorted({int(value) for value in gdf[config.target_property].dropna().unique()})
    if not class_values:
        raise ValueError(
            "Training shapefiles must contain the target class column "
            f"'{config.target_property}'."
        )
    label_image = ee.Image().int().paint(features, config.target_property)
    image_with_labels = image.updateMask(label_image.neq(0)).addBands(
        label_image.rename(config.target_property)
    )
    samples = image_with_labels.stratifiedSample(
        classBand=config.target_property,
        classValues=ee.List(class_values),
        classPoints=ee.List([config.sample_per_polygon] * len(class_values)),
        region=features.geometry(),
        scale=config.scale,
        seed=config.random_seed,
        geometries=True,
        tileScale=config.tile_scale,
    )
    sampled_gdf = _feature_collection_to_geodataframe(samples)
    return sampled_gdf


def _sample_inference_data(
    image: ee.Image,
    features: ee.FeatureCollection,
    config: DownloadConfig,
) -> gpd.GeoDataFrame:
    """Sample embeddings for inference across the provided geometry."""

    samples = image.sample(
        region=features.geometry(),
        scale=config.scale,
        numPixels=config.max_inference_pixels,
        seed=config.random_seed,
        geometries=True,
        tileScale=config.tile_scale,
    )
    return _feature_collection_to_geodataframe(samples)


def download_embeddings(
    shapefile_dir: Path,
    data_dir: Path,
    config: DownloadConfig,
) -> List[Path]:
    """Download embeddings for all shapefiles in ``shapefile_dir``.

    Returns a list of GeoParquet files containing sampled embeddings.
    """

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    saved_files: List[Path] = []
    for shapefile in list_shapefiles(shapefile_dir):
        LOGGER.info("Processing shapefile %s", shapefile.name)
        gdf = load_geodataframe(shapefile)
        target_is_available = config.target_property in gdf.columns
        properties = [config.target_property] if target_is_available else None
        feature_collection = geodataframe_to_feature_collection(gdf, properties=properties)
        for year in config.years:
            LOGGER.info("  Sampling embeddings for year %s", year)
            image = _embedding_image_for_year(year, feature_collection.geometry())
            if target_is_available:
                sampled = _sample_training_data(image, feature_collection, gdf, config)
            else:
                sampled = _sample_inference_data(image, feature_collection, config)
            sampled["source_shapefile"] = shapefile.stem
            sampled["year"] = year
            sampled["is_training"] = target_is_available
            if target_is_available and config.target_property not in sampled.columns:
                raise ValueError(
                    "Sampled training data missing the target property."
                )
            outfile = data_dir / f"{shapefile.stem}_{year}_{'train' if target_is_available else 'inference'}.parquet"
            sampled.to_parquet(outfile)
            LOGGER.info("  Wrote %s", outfile)
            saved_files.append(outfile)
    return saved_files
