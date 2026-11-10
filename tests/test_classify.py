"""Offline checks for mrd.classify: run with `pytest tests`."""

import numpy as np
import pytest
import xarray as xr
from dep_tools.grids import PACIFIC_GRID_10
from odc.geo.xr import assign_crs, to_cog, xr_coords
from rasterio.io import MemoryFile

import mrd.classify as c
from mrd import MODEL_URL, NODATA, VERSION, collection_url_root, model_url


def test_parse_years():
    assert c.parse_years("2022") == [2022]
    assert c.parse_years("2017-2020") == [2017, 2018, 2019, 2020]
    assert c.parse_years("2017,2020") == [2017, 2020]
    for bad in ["2024-2017", "20x", "2017-2020-2022", ""]:
        with pytest.raises(ValueError, match="Invalid years"):
            c.parse_years(bad)


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


def test_versions_match():
    from pathlib import Path

    assert VERSION == model_url(VERSION).split("/")[-2]
    workflow = Path(__file__).parents[1] / "workflow" / "ml-mrd-annual.yaml"
    assert f'value: "{VERSION}"' in workflow.read_text(), "the workflow's version must match pyproject.toml's"
