"""Train the land-cover model on the MRD training points and publish it.

Features come from mrd.data.load_data and sample_points, the same functions classification uses, so the model always
sees features computed the same way. Run with `mrd train`; see `mrd train --help`.
"""

import hashlib
import json
import logging
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import sklearn
from dep_tools.aws import object_exists
from dep_tools.grids import PACIFIC_GRID_10  # TODO: Validate that v0-0-1 used exactly this grid.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

from mrd import CLASSES, MODEL_PREFIX, OUTPUT_BUCKET, VERSION, tile_index
from mrd.aws import s3_client
from mrd.data import FEATURES, S2_REFLECTANCE, NoDataError, load_data, sample_points

logger = logging.getLogger(__name__)

# Relative to the current directory: run `mrd train` from the repo root.
TRAINING_DATA = Path("data/mred-tdata.geojson")
MODELS_DIR = Path("data/models")

CLASS_COLUMN = "lulc_code"
# Small chunks so only the chunks containing points are read. Values don't depend on chunk size: S2 and S1 load on
# their native grid and the DEM is reprojected with nearest resampling.
TRAINING_CHUNKS = {"x": 512, "y": 512}
# FR5. n_jobs isn't a hyperparameter: it's passed separately and cleared before saving, so classification's
# per-block predict_proba doesn't start its own threads inside dask.
HYPERPARAMETERS = {"n_estimators": 500, "class_weight": "balanced", "oob_score": True, "random_state": 42}
CV_FOLDS = 5
TABLE_COLUMNS = ["tile", CLASS_COLUMN, *FEATURES]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str | None:
    """Output of a git command, or None if it fails (e.g. not a git checkout)."""
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


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


# TODO: Get IDs from Fiji geom, not training data.
def tile_ids(points: gpd.GeoDataFrame) -> pd.Series:
    """The dep grid tile of each point, as "col,row"."""
    indices = (PACIFIC_GRID_10.pt2idx(x, y) for x, y in zip(points.geometry.x, points.geometry.y))
    return pd.Series([f"{i.x},{i.y}" for i in indices], index=points.index)


def build_table(points: gpd.GeoDataFrame, year: int) -> tuple[pd.DataFrame, int]:
    """Features at every point, tile by tile, and the number of points that couldn't be sampled.

    Points in a tile without S2 or S1 data, or exactly on a tile edge, aren't sampled. The table keeps points with
    NaN features; see drop_unobserved.
    """
    tiles = tile_ids(points)
    tables, not_sampled = [], 0
    for tile in sorted(tiles.unique(), key=tile_index):
        tile_points = points[tiles == tile]
        try:
            ds = load_data(PACIFIC_GRID_10.tile_geobox(tile_index(tile)), year, TRAINING_CHUNKS)
        except NoDataError as e:
            logger.warning(f"Skipping {len(tile_points)} points in tile {tile}: {e}")
            not_sampled += len(tile_points)
            continue
        features = sample_points(ds, tile_points)
        logger.info(f"Sampled {len(features)} of {len(tile_points)} points in tile {tile}")
        not_sampled += len(tile_points) - len(features)
        tables.append(features.assign(tile=tile, **{CLASS_COLUMN: points.loc[features.index, CLASS_COLUMN]}))
    table = pd.concat(tables).sort_index()[TABLE_COLUMNS] if tables else pd.DataFrame(columns=TABLE_COLUMNS)
    # Same as predict: sklearn rejects inf.
    table[FEATURES] = table[FEATURES].replace([np.inf, -np.inf], np.nan)
    return table, not_sampled


def drop_unobserved(table: pd.DataFrame) -> pd.DataFrame:
    """Drop points without an S2 observation: predict makes those pixels nodata, so the model never sees them."""
    return table[table[S2_REFLECTANCE].notna().all(axis=1)]


def _cv_metrics(table: pd.DataFrame, cv, groups: str | None) -> dict:
    x, y = table[FEATURES], table[CLASS_COLUMN]
    model = RandomForestClassifier(**{**HYPERPARAMETERS, "oob_score": False}, n_jobs=-1)
    predicted = cross_val_predict(model, x, y, cv=cv, groups=table[groups] if groups else None)
    labels = sorted(int(c) for c in y.unique())
    report = classification_report(y, predicted, labels=labels, output_dict=True, zero_division=0)
    return {
        "splitter": type(cv).__name__,
        "groups": groups,
        "n_splits": CV_FOLDS,
        "overall_accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            code: {
                "precision": float(report[str(code)]["precision"]),
                "recall": float(report[str(code)]["recall"]),
                "f1": float(report[str(code)]["f1-score"]),
                "support": int(report[str(code)]["support"]),
            }
            for code in labels
        },
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y, predicted, labels=labels).tolist()},
    }


def cross_validate(table: pd.DataFrame) -> dict:
    """Spatial (grouped by tile) and random stratified 5-fold cross-validation reports."""
    return {
        "spatial": _cv_metrics(table, StratifiedGroupKFold(CV_FOLDS, shuffle=True, random_state=42), "tile"),
        "random": _cv_metrics(table, StratifiedKFold(CV_FOLDS, shuffle=True, random_state=42), None),
    }


def fit(table: pd.DataFrame) -> RandomForestClassifier:
    """Fit on all points. A DataFrame keeps the feature names on the model."""
    model = RandomForestClassifier(**HYPERPARAMETERS, n_jobs=-1).fit(table[FEATURES], table[CLASS_COLUMN])
    model.n_jobs = None
    return model


# TODO: Needed? STAC processing instead?
def build_metadata(
    model: RandomForestClassifier,
    table: pd.DataFrame,
    cv: dict,
    year: int,
    version: str,
    training_data: Path,
    points: dict,
) -> dict:
    sha, status = _git("rev-parse", "HEAD"), _git("status", "--porcelain")
    return {
        "version": version,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": sha.strip() if sha else None,
        "git_dirty": bool(status.strip()) if status is not None else None,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "training_year": year,
        "training_data": str(training_data),
        "training_data_sha256": _sha256(training_data),
        "points": points,
        "class_counts": {int(k): int(v) for k, v in table[CLASS_COLUMN].value_counts().sort_index().items()},
        "tiles": {k: int(v) for k, v in table["tile"].value_counts().sort_index().items()},
        "feature_names": list(model.feature_names_in_),
        "classes": {code: name for code, (name, _) in CLASSES.items()},
        "hyperparameters": HYPERPARAMETERS,
        "oob_score": float(model.oob_score_),
        "cv": cv,
    }


def _keys(version: str) -> dict[str, str]:
    return {name: f"{MODEL_PREFIX}/{version}/{name}" for name in ["metadata.json", "model.joblib"]}


def upload(local_dir: Path, version: str) -> None:
    """Upload metadata.json then model.joblib, so a model is never published without its metadata."""
    client = s3_client()
    content_types = {"metadata.json": "application/json", "model.joblib": "application/octet-stream"}
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
    """Train on all points, write model.joblib and metadata.json locally, and upload them unless upload_model is
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
    raw, not_sampled = build_table(points, year)
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

    cv = cross_validate(table)
    model = fit(table)
    metadata = build_metadata(model, table, cv, year, version, training_data, counts)

    local_dir = MODELS_DIR / version
    local_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, local_dir / "model.joblib", compress=3)
    (local_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    for name in ["spatial", "random"]:
        logger.info(f"{name} CV: accuracy {cv[name]['overall_accuracy']:.3f}, macro F1 {cv[name]['macro_f1']:.3f}")
    logger.info(f"OOB score {metadata['oob_score']:.3f}. Wrote {local_dir}")

    if upload_model:
        upload(local_dir, version)
    return local_dir
