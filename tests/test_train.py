"""Offline checks for mrd.train: run with `pytest tests`."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from dep_tools.grids import PACIFIC_GRID_10
from shapely.geometry import Point

import mrd.train as t
from mrd.data import FEATURES


def _points(xy: list[tuple[float, float]], codes: list[int], crs=PACIFIC_GRID_10.crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({t.CLASS_COLUMN: codes}, geometry=[Point(x, y) for x, y in xy], crs=crs)


def test_read_points(tmp_path):
    path = tmp_path / "points.geojson"
    points = _points([(178.0, -17.5), (178.1, -17.6)], [1, 2], crs="EPSG:4326")
    points.loc[2] = [3, None]
    points.to_file(path)
    read, null_geometry = t.read_points(path)
    assert null_geometry == 1 and list(read.index) == [0, 1]
    assert read.crs == PACIFIC_GRID_10.crs and read[t.CLASS_COLUMN].dtype.kind == "i"
    _points([(178.0, -17.5)], [9], crs="EPSG:4326").to_file(path)
    with pytest.raises(ValueError, match="Unknown class"):
        t.read_points(path)


def test_drop_unobserved_keeps_s1_gaps():
    table = pd.DataFrame(0.5, index=[0, 1, 2], columns=t.TABLE_COLUMNS)
    table.loc[0, "red"] = np.nan
    table.loc[1, "mean_vv"] = np.nan
    assert list(t.drop_unobserved(table).index) == [1, 2]


def _table(n_per_class: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    codes = np.repeat([1, 2, 3], n_per_class)
    table = pd.DataFrame(rng.random((len(codes), len(FEATURES))), columns=FEATURES)
    table[FEATURES[0]] += codes  # separable enough to learn
    table.insert(0, t.CLASS_COLUMN, codes)
    return table


def test_fit():
    model = t.fit(_table())
    assert list(model.feature_names_in_) == FEATURES and model.n_jobs is None and 0 <= model.oob_score_ <= 1
