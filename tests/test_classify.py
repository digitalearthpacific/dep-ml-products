"""Offline checks for mrd.classify: run with `pytest tests`."""

import json
import logging
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from botocore.exceptions import ClientError
from dep_tools.grids import PACIFIC_GRID_10
from odc.geo.xr import assign_crs, to_cog, xr_coords
from rasterio.io import MemoryFile
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier

import mrd.classify as c
from mrd import MODEL_URL, NODATA, VERSION, collection_url_root, model_url
from mrd.data import FEATURES, NoDataError


def test_parse_years():
    assert c.parse_years("2022") == [2022]
    assert c.parse_years("2017-2020") == [2017, 2018, 2019, 2020]
    assert c.parse_years("2017,2020") == [2017, 2020]
    for bad in ["2024-2017", "20x", "2017-2020-2022", ""]:
        with pytest.raises(ValueError, match="Invalid years"):
            c.parse_years(bad)


def _grid(monkeypatch, rows: list[tuple[str, str]]):
    frame = gpd.GeoDataFrame(
        {"tile_id": [t for t, _ in rows], "country_code": [r for _, r in rows]},
        geometry=[box(0, 0, 1, 1)] * len(rows),
    )
    monkeypatch.setattr(c.gpd, "read_file", lambda url: frame)


def test_get_tiles(monkeypatch):
    _grid(monkeypatch, [("10,1", "FJ"), ("9,1", "FJ"), ("9,2", "TO")])
    assert c.get_tiles(["FJ"]) == ["9,1", "10,1"]
    assert c.get_tiles(None) == ["9,1", "9,2", "10,1"]
    with pytest.raises(ValueError, match="XX"):
        c.get_tiles(["XX"])


def test_print_tasks(monkeypatch):
    _grid(monkeypatch, [("65,20", "FJ"), ("66,20", "FJ"), ("1,1", "TO")])
    done = c.item_path("b", "0-0-2", 2022).stac_path("66,20")
    monkeypatch.setattr(c, "existing_stac_keys", lambda bucket, version: {done})
    tasks = json.loads(c.print_tasks("2022-2023", "FJ", "0-0-2", "b", overwrite=False, limit=None))
    assert {(t["tile-id"], t["year"]) for t in tasks} == {("65,20", 2022), ("65,20", 2023), ("66,20", 2023)}
    assert all(t["version"] == "0-0-2" for t in tasks)
    all_tasks = json.loads(c.print_tasks("2022-2023", "ALL", "0-0-2", "b", overwrite=True, limit=None))
    assert len(all_tasks) == 6
    output = c.print_tasks("2022", "FJ", "0-0-2", "b", overwrite=True, limit=1)
    assert json.loads(output) == [{"tile-id": "65,20", "year": 2022, "version": "0-0-2"}] and " " not in output


def test_existing_stac_keys(monkeypatch):
    pages = [{"Contents": [{"Key": "a/x.stac-item.json"}, {"Key": "a/x_class.tif"}]}, {}]
    paginator = SimpleNamespace(paginate=lambda **kwargs: pages)
    monkeypatch.setattr(c, "s3_client", lambda: SimpleNamespace(get_paginator=lambda name: paginator))
    assert c.existing_stac_keys("b", "0-0-2") == {"a/x.stac-item.json"}

    def forbidden(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "ListObjectsV2")

    paginator = SimpleNamespace(paginate=forbidden)
    assert c.existing_stac_keys("b", "0-0-2") == set()

    def no_bucket(**kwargs):
        raise ClientError({"Error": {"Code": "NoSuchBucket"}}, "ListObjectsV2")

    paginator = SimpleNamespace(paginate=no_bucket)
    with pytest.raises(ClientError):
        c.existing_stac_keys("b", "0-0-2")


def test_paths_and_urls():
    path = c.item_path("dep-public-staging", "0-0-2", 2022)
    assert path.stac_path("66,20") == "dep_s2s1_mrd/0-0-2/066/020/2022/dep_s2s1_mrd_066_020_2022.stac-item.json"
    assert path.path("66,20", "class") == "dep_s2s1_mrd/0-0-2/066/020/2022/dep_s2s1_mrd_066_020_2022_class.tif"
    assert model_url(VERSION) == MODEL_URL
    assert collection_url_root("dep-public-prod") == "https://stac.digitalearthpacific.org/collections"
    assert collection_url_root("dep-public-staging") == "https://stac.staging.digitalearthpacific.io/collections"


def test_antimeridian_safe_bbox_and_geometry():
    crossing, west = PACIFIC_GRID_10.tile_geobox((66, 20)), PACIFIC_GRID_10.tile_geobox((65, 20))
    bbox = c.antimeridian_safe_bbox(crossing, [-180, -19, 180, -17])
    assert bbox[0] > bbox[2]
    assert c.antimeridian_safe_bbox(west, [1, 2, 3, 4]) == [1, 2, 3, 4]
    ring = [[179.97, -18.48], [-179.17, -18.48], [-179.17, -17.65], [179.97, -17.65], [179.97, -18.48]]
    assert c.antimeridian_safe_geometry(crossing, {"type": "Polygon", "coordinates": [ring]})["type"] == "MultiPolygon"
    polygon = {"type": "Polygon", "coordinates": [[[179, -18], [179.5, -18], [179.5, -17], [179, -17], [179, -18]]]}
    assert c.antimeridian_safe_geometry(west, polygon) == polygon


def test_cog_nodata_from_attrs():
    geobox = PACIFIC_GRID_10.tile_geobox((65, 21)).zoom_out(960)
    da = assign_crs(
        xr.DataArray(np.full(geobox.shape, NODATA, "uint8"), coords=xr_coords(geobox), dims=geobox.dims), geobox.crs
    )
    da.attrs["nodata"] = NODATA
    with MemoryFile(to_cog(da)) as memfile, memfile.open() as src:
        assert src.nodata == NODATA and src.dtypes[0] == "uint8"


class _Calls(list):
    def __call__(self, name):
        return lambda *args, **kwargs: self.append((name, args, kwargs))


def _stub_run(monkeypatch, exists=False, no_data=False):
    calls = _Calls()
    monkeypatch.setattr(c, "s3_client", lambda: "client")
    monkeypatch.setattr(c, "object_exists", lambda bucket, key, client: exists)

    def load_data(geobox, year):
        if no_data:
            raise NoDataError(f"No dep_s1_geomad items for {year} over somewhere")
        coarse = geobox.zoom_out(96)
        rng = np.random.default_rng(0)
        ds = xr.Dataset({f: (("y", "x"), rng.random(coarse.shape)) for f in FEATURES}, coords=xr_coords(coarse))
        ds["red"][0, 0] = np.nan
        return assign_crs(ds, coarse.crs)

    def load_model(url):
        calls.append(("load_model", (url,), {}))
        x = pd.DataFrame(np.random.default_rng(1).random((30, len(FEATURES))), columns=FEATURES)
        return RandomForestClassifier(n_estimators=3, random_state=0).fit(x, np.resize([1, 2, 3], 30))

    def write(self, data, item_id):
        calls.append(("write", (data, item_id), {}))
        return ["class.tif", "proba.tif"]

    item = SimpleNamespace(bbox=[0, 0, 1, 1], geometry={"type": "Polygon"}, properties={}, links=[])
    item.add_link = item.links.append
    monkeypatch.setattr(c, "load_data", load_data)
    monkeypatch.setattr(c, "load_model", load_model)
    monkeypatch.setattr(c.AwsDsCogWriter, "write", write)
    monkeypatch.setattr(c, "get_stac_item", lambda *args, **kwargs: calls.append(("stac", args, kwargs)) or item)
    monkeypatch.setattr(c, "write_stac_s3", calls("write_stac"))
    return calls, item


def test_run_skips_existing(monkeypatch):
    calls, _ = _stub_run(monkeypatch, exists=True)
    assert c.run("65,21", 2022) is None and calls == []


def test_run_no_data_exits_quietly(monkeypatch, caplog):
    calls, _ = _stub_run(monkeypatch, no_data=True)
    with caplog.at_level(logging.WARNING, logger="mrd"):
        assert c.run("65,21", 2025) is None
    assert "dep_s1_geomad" in caplog.text
    assert [name for name, *_ in calls] == ["load_model"]


def test_run_writes_cogs_then_stac(monkeypatch):
    calls, item = _stub_run(monkeypatch)
    keys = c.run("65,21", 2022, bucket="dep-public-prod", image_tag="v1", threads=2)
    assert [name for name, *_ in calls] == ["load_model", "write", "stac", "write_stac"]
    out = calls[1][1][0]
    assert set(out.data_vars) == {"class", "proba"} and out["class"].dtype == "uint8"
    assert out["class"].attrs["nodata"] == NODATA and out["class"].values[0, 0] == NODATA
    assert out.attrs["stac_properties"]["datetime"] == "2022-01-01T00:00:00Z"
    assert calls[2][2]["collection_url_root"] == collection_url_root("dep-public-prod")
    assert calls[3][1][1] == keys[-1] == c.item_path("dep-public-prod", VERSION, 2022).stac_path("65,21")
    assert calls[3][2]["client"] == "client"
    assert item.properties == {"mrd:model": MODEL_URL, "mrd:image_tag": "v1"}
    assert item.links[0].rel == "related" and item.links[0].target == MODEL_URL
    assert rasterio  # the real COG writing is covered by test_cog_nodata_from_attrs


def test_run_with_local_model_records_path_without_link(monkeypatch, tmp_path):
    calls, item = _stub_run(monkeypatch)
    c.run("65,21", 2022, model_path=str(tmp_path / "model.joblib"))
    assert item.links == [] and item.properties["mrd:model"] == (tmp_path / "model.joblib").resolve().as_uri()


def test_cli(monkeypatch):
    from typer.testing import CliRunner

    from mrd.cli import app

    runner = CliRunner()
    monkeypatch.setattr(c, "print_tasks", lambda *args: '[{"tile-id":"1,1","year":2022,"version":"v"}]')
    result = runner.invoke(app, ["print-tasks", "--years", "2022"])
    assert result.exit_code == 0 and json.loads(result.stdout) == [{"tile-id": "1,1", "year": 2022, "version": "v"}]

    def fail(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(c, "run", fail)
    result = runner.invoke(app, ["run", "--tile-id", "65,21", "--year", "2022"])
    assert result.exit_code != 0 and isinstance(result.exception, RuntimeError)
