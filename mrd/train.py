"""Train the land-cover model on the MRD training points and publish it.

Features come from mrd.data.load_data and sample_points, the same functions classification uses, so the model always
sees features computed the same way. Run with `mrd train`; see `mrd train --help`.
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from dep_tools.aws import object_exists
from dep_tools.grids import PACIFIC_GRID_10  # Same grid as v0-0-1
from sklearn.ensemble import RandomForestClassifier

from mrd import CLASSES, MODEL_PREFIX, OUTPUT_BUCKET, VERSION, tile_index
from mrd.aws import s3_client
from mrd.data import FEATURES, S2_REFLECTANCE, NoDataError, fiji_tiles, load_data, sample_points

logger = logging.getLogger(__name__)

# Relative to the current directory: run `mrd train` from the repo root.
TRAINING_DATA = Path("data/mred-tdata.geojson")
MODELS_DIR = Path("data/models")
# Sampled features per version, tile and year, reused by later runs. Delete them to resample, e.g. after changing
# the training data or the features.
FEATURES_DIR = Path("data/features")

CLASS_COLUMN = "lulc_code"
# Small chunks so only the chunks containing points are read. Values don't depend on chunk size: S2 and S1 load on
# their native grid and the DEM is reprojected with nearest resampling.
TRAINING_CHUNKS = {"x": 512, "y": 512}
# FR5. n_jobs isn't a hyperparameter: it's passed separately and cleared before saving, so classification's
# per-block predict_proba doesn't start its own threads inside dask.
HYPERPARAMETERS = {"n_estimators": 500, "class_weight": "balanced", "oob_score": True, "random_state": 42}
TABLE_COLUMNS = [CLASS_COLUMN, *FEATURES]


def read_points(path: Path) -> tuple[gpd.GeoDataFrame, int]:
    """Training points in EPSG:3832, and the number dropped for null or empty geometry.

    Raises:
        ValueError: If a class code isn't in CLASSES.
    """
    points = gpd.read_file(path, columns=[CLASS_COLUMN])
    has_geometry = points.geometry.notna() & ~points.geometry.is_empty
    points = points[has_geometry].to_crs(PACIFIC_GRID_10.crs)
    points[CLASS_COLUMN] = points[CLASS_COLUMN].astype(int)
    unknown = set(points[CLASS_COLUMN]) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unknown class codes in {path}: {sorted(unknown)}")
    return points, int((~has_geometry).sum())


def build_table(points: gpd.GeoDataFrame, year: int, version: str = VERSION) -> tuple[pd.DataFrame, int]:
    """Features at every point, loading the Fiji tiles (fiji_tiles) that contain points, and the number of points
    that couldn't be sampled.

    Points outside the Fiji tiles, in a tile without S2 or S1 data, or exactly on a tile edge, aren't sampled. The
    table keeps points with NaN features; see drop_unobserved.
    """
    tables = []
    tiles = fiji_tiles()
    for i, tile in enumerate(tiles, 1):
        logger.info(f"Tile {i}/{len(tiles)}: {tile}")
        # Indexed by the points' index, with the class, so cached rows match the points they came from.
        cache = FEATURES_DIR / f"features_{version}_{tile.replace(',', '_')}_{year}.csv"
        if cache.exists():
            tables.append(pd.read_csv(cache, index_col=0))
            logger.info(f"    Loaded cached features from {cache}")
            continue
        geobox = PACIFIC_GRID_10.tile_geobox(tile_index(tile))
        tile_points = points[points.within(geobox.extent.geom)]
        if tile_points.empty:
            continue
        try:
            ds = load_data(geobox, year, TRAINING_CHUNKS)
        except NoDataError as e:
            logger.warning(f"Skipping {len(tile_points)} points in tile {tile}: {e}")
            continue
        features = sample_points(ds, tile_points).assign(**{CLASS_COLUMN: tile_points[CLASS_COLUMN]})
        logger.info(f"    Sampled {len(features)} of {len(tile_points)} points")
        cache.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(cache)
        tables.append(features)
    table = pd.concat(tables).sort_index()[TABLE_COLUMNS] if tables else pd.DataFrame(columns=TABLE_COLUMNS)
    not_sampled = len(points) - len(table)
    # Same as predict: sklearn rejects inf.
    table[FEATURES] = table[FEATURES].replace([np.inf, -np.inf], np.nan)
    return table, not_sampled


def drop_unobserved(table: pd.DataFrame) -> pd.DataFrame:
    """Drop points without an S2 observation: predict makes those pixels nodata, so the model never sees them."""
    return table[table[S2_REFLECTANCE].notna().all(axis=1)]


def fit(table: pd.DataFrame) -> RandomForestClassifier:
    """Fit on all points. A DataFrame keeps the feature names on the model."""
    model = RandomForestClassifier(**HYPERPARAMETERS, n_jobs=-1).fit(table[FEATURES], table[CLASS_COLUMN])
    model.n_jobs = None
    return model


def _keys(version: str) -> dict[str, str]:
    return {name: f"{MODEL_PREFIX}/{version}/{name}" for name in ["metrics.json", "model.joblib"]}


def upload(local_dir: Path, version: str) -> None:
    """Upload metrics.json then model.joblib, so a model is never published without its metrics."""
    client = s3_client()
    content_types = {"metrics.json": "application/json", "model.joblib": "application/octet-stream"}
    for name, key in _keys(version).items():
        client.upload_file(str(local_dir / name), OUTPUT_BUCKET, key, ExtraArgs={"ContentType": content_types[name]})
        logger.info(f"Uploaded s3://{OUTPUT_BUCKET}/{key}")


def train(
    year: int = 2022,
    version: str = VERSION,
    training_data: Path = TRAINING_DATA,
    upload_model: bool = True,
    overwrite: bool = False,
) -> Path:
    """Train on all points, write model.joblib and metrics.json locally, and upload them unless upload_model is
    False. Returns the local directory.

    Raises:
        FileExistsError: If uploading without overwrite and the version is already published (checked first).
    """
    if upload_model and not overwrite:
        client = s3_client()
        existing = [key for key in _keys(version).values() if object_exists(OUTPUT_BUCKET, key, client)]
        if existing:
            raise FileExistsError(f"Already published (use --overwrite to replace): {existing}")

    points, null_geometry = read_points(training_data)
    raw, not_sampled = build_table(points, year, version)
    table = drop_unobserved(raw)
    counts = {
        "input": len(points) + null_geometry,
        "null_geometry": null_geometry,
        "not_sampled": not_sampled,
        "no_s2": len(raw) - len(table),
        "used": len(table),
    }
    logger.info(f"Training points: {counts}")
    missing = set(CLASSES) - set(table[CLASS_COLUMN])
    if missing:
        raise ValueError(f"No training points left for classes {sorted(missing)}")

    model = fit(table)
    metrics = {"oob_score": float(model.oob_score_)}

    local_dir = MODELS_DIR / version
    local_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, local_dir / "model.joblib", compress=3)
    (local_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"OOB score {metrics['oob_score']:.3f}. Wrote {local_dir}")

    if upload_model:
        upload(local_dir, version)
    return local_dir
