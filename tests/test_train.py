"""Offline checks for mrd.train: run with `pytest tests`."""

import json
import subprocess
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from botocore.exceptions import ClientError
from dep_tools.grids import PACIFIC_GRID_10
from odc.geo.xr import assign_crs, xr_coords
from shapely.geometry import Point

import mrd.train as t
from mrd import CLASSES
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


def test_tile_ids():
    assert list(t.tile_ids(_points([_inside(0.5, 0.5)], [1]))) == ["63,20"]


def test_build_table(monkeypatch):
    points = _points(
        [_inside(0.2, 0.2), _inside(0.7, 0.7), _inside(0.0, 0.5), _inside(0.5, 0.5, TILE.translate_pix(9600, 0))],
        [1, 2, 3, 4],
    )
    points.index = [10, 11, 12, 13]

    def load_data(geobox, year, chunks):
        if geobox == PACIFIC_GRID_10.tile_geobox((64, 20)):
            raise NoDataError("no S1")
        ds = _features(geobox)
        ds["slope"][:] = np.inf
        x, y = _inside(0.2, 0.2)  # point 10 has no S2
        ds["red"].loc[{"x": ds.x.sel(x=x, method="nearest"), "y": ds.y.sel(y=y, method="nearest")}] = np.nan
        return ds

    monkeypatch.setattr(t, "load_data", load_data)
    table, not_sampled = t.build_table(points, 2022)
    # Point 12 is on the tile's west edge (not sampled); point 13 is in a tile without data.
    assert not_sampled == 2
    assert list(table.columns) == t.TABLE_COLUMNS and list(table.index) == [10, 11]
    assert table.tile.tolist() == ["63,20", "63,20"] and table[t.CLASS_COLUMN].tolist() == [1, 2]
    assert table.slope.isna().all()  # inf -> NaN
    used = t.drop_unobserved(table)
    assert list(used.index) == [11]


def test_tiles_in_numeric_order(monkeypatch):
    calls = []

    def load_data(geobox, year, chunks):
        calls.append(geobox)
        raise NoDataError("none")

    monkeypatch.setattr(t, "tile_ids", lambda points: pd.Series(["10,1", "9,1"], index=points.index))
    monkeypatch.setattr(t, "load_data", load_data)
    _, not_sampled = t.build_table(_points([(0, 0), (1, 1)], [1, 2]), 2022)
    assert calls == [PACIFIC_GRID_10.tile_geobox((9, 1)), PACIFIC_GRID_10.tile_geobox((10, 1))]
    assert not_sampled == 2


def test_drop_unobserved_keeps_s1_gaps():
    table = pd.DataFrame(0.5, index=[0, 1, 2], columns=t.TABLE_COLUMNS)
    table.loc[0, "red"] = np.nan
    table.loc[1, "mean_vv"] = np.nan
    assert list(t.drop_unobserved(table).index) == [1, 2]


def test_load_table_cache(monkeypatch, tmp_path):
    data = tmp_path / "points.geojson"
    data.write_text("{}")
    built = []

    def build_table(points, year):
        built.append(year)
        return pd.DataFrame(0.5, index=[0], columns=t.TABLE_COLUMNS).assign(tile="63,20"), 3

    monkeypatch.setattr(t, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(t, "build_table", build_table)
    assert t.load_table(None, 2022, data, use_cache=True)[1:] == (3, False)
    assert t.load_table(None, 2022, data, use_cache=True)[1:] == (3, True)
    assert t.load_table(None, 2022, data, use_cache=False)[2] is False
    monkeypatch.setattr(t, "_data_code_hash", lambda: "changed")
    assert t.load_table(None, 2022, data, use_cache=True)[2] is False
    assert len(built) == 3
    # A cache with other columns is rebuilt.
    cached = next((tmp_path / "cache").glob(f"*{'changed'[:12]}.parquet"))
    pd.DataFrame({"a": [1]}).to_parquet(cached)
    assert t.load_table(None, 2022, data, use_cache=True)[2] is False


def _table(n_per_class: int = 12, n_tiles: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    codes = np.repeat([1, 2, 3], n_per_class)
    table = pd.DataFrame(rng.random((len(codes), len(FEATURES))), columns=FEATURES)
    table[FEATURES[0]] += codes  # separable enough to learn
    table.insert(0, t.CLASS_COLUMN, codes)
    table.insert(0, "tile", [f"{i % n_tiles},1" for i in range(len(codes))])
    return table


def test_cross_validate_is_deterministic():
    table = _table()
    first, second = t.cross_validate(table), t.cross_validate(table)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    for name, groups in [("spatial", "tile"), ("random", None)]:
        report = first[name]
        assert report["groups"] == groups and report["n_splits"] == t.CV_FOLDS
        assert set(report) == {
            "splitter",
            "groups",
            "n_splits",
            "overall_accuracy",
            "macro_f1",
            "per_class",
            "confusion_matrix",
        }
        assert set(report["per_class"]) == {1, 2, 3} and report["confusion_matrix"]["labels"] == [1, 2, 3]
        assert sum(c["support"] for c in report["per_class"].values()) == len(table)


def test_fit_and_metadata(tmp_path):
    table = _table()
    model = t.fit(table)
    assert list(model.feature_names_in_) == FEATURES and model.n_jobs is None
    data = tmp_path / "points.geojson"
    data.write_text("{}")
    points = {"input": 40, "null_geometry": 1, "not_sampled": 2, "no_s2": 1, "used": len(table)}
    metadata = json.loads(
        json.dumps(t.build_metadata(model, table, t.cross_validate(table), 2022, "0-0-2", data, points, False))
    )
    for key in [
        "sklearn_version",
        "python_version",
        "feature_names",
        "classes",
        "training_year",
        "training_data_sha256",
        "hyperparameters",
        "oob_score",
        "cv",
        "git_sha",
        "created",
        "points",
    ]:
        assert key in metadata, key
    assert metadata["points"]["input"] == sum(v for k, v in points.items() if k != "input")
    assert metadata["training_data"] == str(data)  # outside the repo: absolute
    assert metadata["hyperparameters"] == t.HYPERPARAMETERS and "n_jobs" not in metadata["hyperparameters"]
    assert metadata["classes"] == {str(code): name for code, (name, _) in CLASSES.items()}


def test_train_requires_every_class(monkeypatch, tmp_path):
    data = tmp_path / "points.geojson"
    data.write_text("{}")
    monkeypatch.setattr(t, "read_points", lambda path: (_points([(0, 0)], [1]), 0))
    monkeypatch.setattr(t, "load_table", lambda *args, **kwargs: (_table(), 0, False))
    with pytest.raises(ValueError, match="No training points left for classes"):
        t.train(training_data=data, upload_model=False)


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code}}, "HeadObject")


def test_s3_exists():
    def client(error=None):
        def head_object(**kwargs):
            if error:
                raise _client_error(error)

        return SimpleNamespace(head_object=head_object)

    assert t._s3_exists(client(), "k") is True
    assert t._s3_exists(client("404"), "k") is False
    with pytest.raises(ClientError):
        t._s3_exists(client("403"), "k")


class _StubS3:
    def __init__(self, existing=()):
        self.existing, self.uploads = set(existing), []

    def head_object(self, Bucket, Key):
        if Key not in self.existing:
            raise _client_error("404")

    def upload_file(self, filename, bucket, key, ExtraArgs):
        self.uploads.append((key, ExtraArgs["ContentType"]))


def test_upload(monkeypatch, tmp_path):
    stub = _StubS3(existing={"dep_s2s1_mrd/models/0-0-2/model.joblib"})
    monkeypatch.setattr(t, "_s3_client", lambda: stub)
    with pytest.raises(FileExistsError):
        t.upload(tmp_path, "0-0-2", overwrite=False)
    assert stub.uploads == []
    t.upload(tmp_path, "0-0-2", overwrite=True)
    assert stub.uploads == [
        ("dep_s2s1_mrd/models/0-0-2/metadata.json", "application/json"),
        ("dep_s2s1_mrd/models/0-0-2/model.joblib", "application/octet-stream"),
    ]


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        check=True,
        capture_output=True,
    )


def test_check_committed_uses_repo_root(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / "mrd").mkdir(parents=True)
    (repo / "data").mkdir()
    (repo / "mrd" / "x.py").write_text("x = 1\n")
    (repo / ".gitignore").write_text("data\n")  # as in this repo: untracked data is hidden from git status
    (repo / "data" / "points.geojson").write_text("{}")
    _git(repo, "init", "-q")
    _git(repo, "add", "mrd", ".gitignore")
    _git(repo, "commit", "-q", "-m", "init")
    monkeypatch.setattr(t, "ROOT", repo)
    monkeypatch.chdir(tmp_path)  # not the repo root
    with pytest.raises(RuntimeError, match="isn't tracked"):
        t._check_committed(repo / "data" / "points.geojson")
    _git(repo, "add", "-f", "data")
    _git(repo, "commit", "-q", "-m", "data")
    t._check_committed(repo / "data" / "points.geojson")
    (repo / "mrd" / "x.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="Commit these"):
        t._check_committed(repo / "data" / "points.geojson")


def test_upload_run_refuses_data_outside_repo(tmp_path):
    with pytest.raises(ValueError, match="must be in the repo"):
        t.train(training_data=tmp_path / "points.geojson", upload_model=True)
