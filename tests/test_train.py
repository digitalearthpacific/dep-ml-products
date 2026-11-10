"""Offline checks for mrd.train: run with `pytest tests`."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from dep_tools.grids import PACIFIC_GRID_10
from odc.geo.xr import assign_crs, xr_coords
from shapely.geometry import Point

import mrd.train as t
from mrd.data import FEATURES, NoDataError

TILE = PACIFIC_GRID_10.tile_geobox((63, 20))


def _points(xy: list[tuple[float, float]], codes: list[int], crs=PACIFIC_GRID_10.crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({t.CLASS_COLUMN: codes}, geometry=[Point(x, y) for x, y in xy], crs=crs)


def _inside(fx: float, fy: float, geobox=TILE) -> tuple[float, float]:
    """A point at fractions (fx, fy) across a geobox."""
    b = geobox.extent.boundingbox
    return b.left + fx * (b.right - b.left), b.bottom + fy * (b.top - b.bottom)


def _features(geobox, value: float = 0.5):
    """A synthetic load_data result on a coarse version of geobox."""
    coarse = geobox.zoom_out(96)
    ds = assign_crs(xr_coords_dataset(coarse, value), coarse.crs)
    return ds


def xr_coords_dataset(geobox, value):
    import xarray as xr

    return xr.Dataset({f: (("y", "x"), np.full(geobox.shape, value)) for f in FEATURES}, coords=xr_coords(geobox))


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


def test_build_table(monkeypatch, tmp_path):
    points = _points(
        [
            _inside(0.2, 0.2),
            _inside(0.7, 0.7),
            _inside(0.0, 0.5),
            _inside(0.5, 0.5, TILE.translate_pix(9600, 0)),
            _inside(0.5, 0.5, TILE.translate_pix(19200, 0)),
        ],
        [1, 2, 3, 4, 5],
    )
    points.index = [10, 11, 12, 13, 14]
    loaded = []

    def load_data(geobox, year, chunks):
        loaded.append(geobox)
        if geobox == PACIFIC_GRID_10.tile_geobox((64, 20)):
            raise NoDataError("no S1")
        ds = _features(geobox)
        ds["slope"][:] = np.inf
        x, y = _inside(0.2, 0.2)  # point 10 has no S2
        ds["red"].loc[{"x": ds.x.sel(x=x, method="nearest"), "y": ds.y.sel(y=y, method="nearest")}] = np.nan
        return ds

    monkeypatch.setattr(t, "load_data", load_data)
    monkeypatch.setattr(t, "fiji_tiles", lambda: ["62,20", "63,20", "64,20"])
    monkeypatch.setattr(t, "FEATURES_DIR", tmp_path)
    table, not_sampled = t.build_table(points, 2022)
    # Point 12 is on the tile's west edge, 13 in a tile without data, 14 outside the Fiji tiles: none sampled.
    assert not_sampled == 3
    # 62,20 has no points, so it isn't loaded.
    assert loaded == [PACIFIC_GRID_10.tile_geobox((63, 20)), PACIFIC_GRID_10.tile_geobox((64, 20))]
    assert list(table.columns) == t.TABLE_COLUMNS and list(table.index) == [10, 11]
    assert table[t.CLASS_COLUMN].tolist() == [1, 2]
    assert table.slope.isna().all()  # inf -> NaN
    used = t.drop_unobserved(table)
    assert list(used.index) == [11]

    # A second run reuses the cached tile: same table, and only the tile without data is loaded again.
    loaded.clear()
    cached, _ = t.build_table(points, 2022)
    assert loaded == [PACIFIC_GRID_10.tile_geobox((64, 20))]
    pd.testing.assert_frame_equal(cached, table, check_dtype=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["features_0-0-2_63_20_2022.csv"]


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


def test_train_requires_every_class(monkeypatch, tmp_path):
    data = tmp_path / "points.geojson"
    data.write_text("{}")
    monkeypatch.setattr(t, "read_points", lambda path: (_points([(0, 0)], [1]), 0))
    monkeypatch.setattr(t, "build_table", lambda points, year, version: (_table(), 0))
    with pytest.raises(ValueError, match="No training points left for classes"):
        t.train(training_data=data, upload_model=False)


class _StubS3:
    def __init__(self):
        self.uploads = []

    def upload_file(self, filename, bucket, key, ExtraArgs):
        self.uploads.append((key, ExtraArgs["ContentType"]))


def test_upload(monkeypatch, tmp_path):
    stub = _StubS3()
    monkeypatch.setattr(t, "s3_client", lambda: stub)
    t.upload(tmp_path, "0-0-2")
    assert stub.uploads == [
        ("dep_s2s1_mrd/models/0-0-2/metrics.json", "application/json"),
        ("dep_s2s1_mrd/models/0-0-2/model.joblib", "application/octet-stream"),
    ]


def test_train_refuses_published_version(monkeypatch):
    monkeypatch.setattr(t, "s3_client", lambda: "client")
    monkeypatch.setattr(t, "object_exists", lambda bucket, key, client: key.endswith("model.joblib"))
    monkeypatch.setattr(t, "read_points", lambda path: pytest.fail("trained before checking"))
    with pytest.raises(FileExistsError, match="--overwrite"):
        t.train(upload_model=True, overwrite=False)
