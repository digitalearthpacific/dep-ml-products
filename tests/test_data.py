"""Checks for mrd.data. Offline tests run with `pytest tests` or `python tests/test_data.py`.

`python tests/test_data.py network` also runs check_network, which loads real data around the antimeridian.
"""

import sys

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pyproj
import pytest
import xarray as xr
from dep_tools.grids import PACIFIC_GRID_10
from odc.geo.geobox import GeoBox
from odc.geo.xr import assign_crs, xr_coords
from odc.stac import load as stac_load
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier

from mrd import NODATA
from mrd.data import (
    FEATURES,
    INDICES,
    S1_BANDS,
    S1_COLLECTION,
    S2_COLLECTION,
    S2_REFLECTANCE,
    _assign_geobox,
    _merge_geographic_halves,
    _nan_dataset,
    _overlaps,
    _search,
    _shift_to_pm180,
    add_indices,
    fiji_tiles,
    load_data,
    load_model,
    predict,
    sample_points,
    scale_reflectance,
    terrain_features,
)

# x of the antimeridian in EPSG:3832 (central meridian 150).
X_180 = pyproj.Transformer.from_crs(4326, 3832, always_xy=True).transform(180, -17)[0]


def _grid(geobox: GeoBox, **variables: np.ndarray) -> xr.Dataset:
    ds = xr.Dataset({k: (geobox.dims, v) for k, v in variables.items()}, coords=xr_coords(geobox))
    return assign_crs(ds, geobox.crs)


def _s2(values: dict[str, np.ndarray]) -> xr.Dataset:
    shape = next(iter(values.values())).shape
    ds = _grid(GeoBox.from_bbox((0, 0, shape[1] * 10, shape[0] * 10), "EPSG:3832", resolution=10), **values)
    for band in ds.data_vars:
        ds[band].attrs["nodata"] = 0
    return ds


def test_reflectance_nodata_and_zero_denominator_give_nan():
    bands = {b: np.array([[0, 1000]], "uint16") for b in S2_REFLECTANCE}  # pixel 0 is nodata
    ds = add_indices(scale_reflectance(_s2(bands)))
    for name in S2_REFLECTANCE + INDICES:
        assert np.isnan(ds[name].values[0, 0]), name
        assert not np.isinf(ds[name].values).any(), name
    # Valid pixels with zero red and nir: the ndvi denominator is 0.
    zero = scale_reflectance(_s2({b: np.array([[1]], "uint16") for b in S2_REFLECTANCE}))
    zero["red"], zero["nir"] = zero.red * 0, zero.nir * 0
    assert np.isnan(add_indices(zero).ndvi.values).all()


def test_indices_bounded():
    rng = np.random.default_rng(0)
    bands = {b: rng.integers(1, 10001, (20, 20)).astype("uint16") for b in S2_REFLECTANCE}
    ds = add_indices(scale_reflectance(_s2(bands)))
    for name in INDICES:
        values = ds[name].values
        assert np.isfinite(values).all(), name
        assert (values >= -1).all() and (values <= 1).all(), name


def test_missing_nodata_raises():
    ds = _s2({b: np.ones((1, 1), "uint16") for b in S2_REFLECTANCE})
    del ds["red"].attrs["nodata"]
    with pytest.raises(ValueError):
        scale_reflectance(ds)


def _dem(elevation: np.ndarray, lat0: float = -17.0) -> xr.DataArray:
    geobox = GeoBox.from_bbox(
        (179.0, lat0, 179.0 + elevation.shape[1] * 0.001, lat0 + elevation.shape[0] * 0.001),
        "EPSG:4326",
        resolution=0.001,
    )
    da = xr.DataArray(
        elevation,
        dims=("y", "x"),
        coords={"y": xr_coords(geobox)["latitude"].values, "x": xr_coords(geobox)["longitude"].values},
    )
    return assign_crs(da, "EPSG:4326")


def test_flat_terrain():
    t = terrain_features(_dem(np.full((10, 10), 50.0)))
    for name in ["slope", "northness", "eastness"]:
        assert np.allclose(t[name].values, 0), name


def test_plane_rising_south_faces_north():
    angle = 10.0
    dem = _dem(np.zeros((20, 20)))
    geod = pyproj.CRS(4326).get_geod()
    lat = np.radians(dem.y)
    north_radius = geod.a * (1 - geod.es) / (1 - geod.es * np.sin(lat) ** 2) ** 1.5
    north_metres = (dem.y - dem.y.mean()) * np.radians(1) * north_radius
    dem = dem + (-np.tan(np.radians(angle)) * north_metres).broadcast_like(dem)  # higher to the south
    t = terrain_features(dem)
    assert np.allclose(t.slope.values, angle, atol=0.5)
    assert (t.northness.values > 0).all()
    assert np.allclose(t.eastness.values, 0, atol=1e-3)
    assert len(FEATURES) == 28


def _half(left: float, values: np.ndarray) -> xr.Dataset:
    geobox = GeoBox.from_bbox((left, -17.05, left + 0.1, -17.0), "EPSG:4326", resolution=0.01)
    return assign_crs(xr.Dataset({"data": (("latitude", "longitude"), values)}, coords=xr_coords(geobox)), 4326)


def test_merge_halves_is_continuous():
    east = _half(179.9, np.tile(np.arange(10.0), (5, 1)))
    west = _half(-180.0, np.tile(np.arange(10.0, 20.0), (5, 1)))
    merged = _merge_geographic_halves(east, west)
    assert merged.data.shape == (5, 20)
    assert (merged.data.values == np.arange(20.0)).all()
    assert np.allclose(np.diff(merged.x.values), 0.01)


def test_one_sided_half_reprojects_to_its_side():
    east = _half(179.9, np.full((5, 10), 100.0))
    shifted = _assign_geobox(east, _shift_to_pm180(east, -180))
    y = pyproj.Transformer.from_crs(4326, 3832, always_xy=True).transform(180, -17.04)[1]
    target = GeoBox.from_bbox((X_180 - 5000, y, X_180 + 5000, y + 2000), "EPSG:3832", resolution=100)
    values = shifted.odc.reproject(target, resampling="nearest").data
    assert np.isfinite(values.sel(x=slice(None, X_180 - 200)).values).all()
    assert np.isnan(values.sel(x=slice(X_180 + 200, None)).values).all()


def test_footprint_filter_drops_edge_neighbours():
    geobox = PACIFIC_GRID_10.tile_geobox((65, 21))
    footprint = lambda gb: gb.extent.to_crs(4326).json  # noqa: E731
    assert _overlaps(footprint(geobox), geobox)
    assert not _overlaps(footprint(PACIFIC_GRID_10.tile_geobox((64, 21))), geobox)
    assert not _overlaps(footprint(PACIFIC_GRID_10.tile_geobox((65, 20))), geobox)
    assert _overlaps(footprint(geobox.translate_pix(4800, 0)), geobox)


def test_sample_points():
    geobox = GeoBox.from_bbox((0, 0, 40, 30), "EPSG:3832", resolution=10)
    ds = _grid(geobox, a=np.arange(12.0).reshape(3, 4), b=np.arange(12.0).reshape(3, 4) * 2)
    points = pd.DataFrame({"geometry": [Point(5, 25), Point(35, 5), Point(500, 500)]}, index=[7, 8, 9])
    points = gpd.GeoDataFrame(points, crs="EPSG:3832")
    sampled = sample_points(ds, points)
    assert list(sampled.columns) == ["a", "b"]
    assert list(sampled.index) == [7, 8]
    assert sampled.loc[7, "a"] == 0 and sampled.loc[8, "a"] == 11 and sampled.loc[8, "b"] == 22
    outside = sample_points(ds, points.loc[[9]])
    assert outside.empty and list(outside.columns) == ["a", "b"]


def _model(labels=(1, 2, 3)) -> RandomForestClassifier:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.random((60, len(FEATURES))), columns=FEATURES)
    return RandomForestClassifier(n_estimators=5, random_state=0).fit(x, np.resize(labels, 60))


def test_predict():
    model = _model()
    rng = np.random.default_rng(1)
    geobox = GeoBox.from_bbox((0, 0, 30, 20), "EPSG:3832", resolution=10)
    ds = _grid(geobox, **{f: rng.random((2, 3)) for f in FEATURES})
    ds["red"][0, 0] = np.nan  # no optical observation -> nodata
    ds["mean_vv"][0, 1] = np.nan  # S1 gap -> still classified
    ds["slope"][0, 2] = np.inf
    out = predict(model, ds).compute()
    assert out["class"].dtype == "uint8" and out["proba"].dtype == "uint8"
    assert out["class"].values[0, 0] == NODATA and out["proba"].values[0, 0] == NODATA
    assert out["class"].values[0, 1] in (1, 2, 3) and out["class"].values[0, 2] in (1, 2, 3)
    assert (out["proba"].values[out["class"].values != NODATA] <= 100).all()
    with pytest.raises(ValueError, match="Missing"):
        predict(model, ds.drop_vars("red"))
    with pytest.raises(ValueError, match="extra"):
        predict(model, ds.assign(foo=ds.red))


def test_load_model(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(_model(), path)
    assert list(load_model(str(path)).feature_names_in_) == FEATURES
    joblib.dump(RandomForestClassifier(n_estimators=2).fit(np.random.rand(10, 2), [1] * 5 + [2] * 5), path)
    with pytest.raises(ValueError, match="feature names"):
        load_model(str(path))
    joblib.dump(_model(labels=(1, 9)), path)
    with pytest.raises(ValueError, match="classes"):
        load_model(str(path))


def _assert_s2_matches_source(ds: xr.Dataset, window: GeoBox) -> None:
    from pystac_client import Client

    from mrd import DEP_CATALOG

    items = _search(Client.open(DEP_CATALOG), window, collections=[S2_COLLECTION], datetime="2022")
    direct = scale_reflectance(stac_load(items, geobox=window, bands=S2_REFLECTANCE).squeeze("time", drop=True))
    for band in S2_REFLECTANCE:
        np.testing.assert_array_equal(ds[band].values, direct[band].values)


def check_network() -> None:
    """Load real data across the antimeridian (tile 66,21) and with valid S2 over land (tile 65,21), for 2022."""
    from pystac_client import Client

    from mrd import DEP_CATALOG

    tiles = fiji_tiles()
    assert len(tiles) == 27 and {"62,27", "66,20", "67,16"} <= set(tiles), tiles  # Rotuma, across 180, Lau
    print(f"fiji_tiles: {len(tiles)} tiles")

    # Tile 66,21 around 16.9 S, columns 102-614 spanning 180 at column ~358. DEP's S2 geomad has no data here.
    window = PACIFIC_GRID_10.tile_geobox((66, 21))[812:1068, 102:614]
    client = Client.open(DEP_CATALOG)
    for collection in [S2_COLLECTION, S1_COLLECTION]:
        ids = [i.id for i in _search(client, window, collections=[collection], datetime="2022")]
        assert ids == [f"{collection}_066_021_2022"], ids
    ds = load_data(window, 2022).compute()
    assert list(ds.data_vars) == FEATURES
    assert ds.odc.geobox == window
    west, east = ds.sel(x=slice(None, X_180)), ds.sel(x=slice(X_180, None))
    for name in ["elevation", "slope", "northness", "eastness", *S1_BANDS, "mean_vv_vh"]:
        for side, data in [("west", west), ("east", east)]:
            assert not data[name].isnull().all("y").any(), f"{name} has all-NaN columns {side} of 180"
    _assert_s2_matches_source(ds, window)
    print(f"Window 1 (tile 66,21 across 180): S2 valid fraction {float(ds.red.notnull().mean()):.3f}")

    # Tile 65,21 near Taveuni's west coast, about 34% land, west of the S2 gap.
    window = PACIFIC_GRID_10.tile_geobox((65, 21))[1300:1556, 8500:9012]
    ds = load_data(window, 2022).compute()
    assert list(ds.data_vars) == FEATURES
    valid = ds.red.notnull()
    print(f"Window 2 (tile 65,21): S2 valid fraction {float(valid.mean()):.3f}")
    assert float(valid.mean()) >= 0.5
    for name in INDICES:
        values = ds[name].where(valid).values
        values = values[np.isfinite(values)]
        assert values.size and (values >= -1).all() and (values <= 1).all(), name
    assert not ds.elevation.isnull().all("y").any()
    _assert_s2_matches_source(ds, window)
    print("Network checks passed")


def test_nan_dataset():
    geobox = GeoBox.from_bbox((0, 0, 80, 60), "EPSG:3832", resolution=10)
    ds = _nan_dataset(geobox, S1_BANDS, {"x": 4, "y": 4})
    assert list(ds.data_vars) == S1_BANDS and ds.odc.geobox == geobox
    assert ds.mean_vv.chunks == ((4, 2), (4, 4)) and ds.mean_vv.isnull().all()


if __name__ == "__main__":
    if sys.argv[1:] == ["network"]:
        check_network()
    else:
        sys.exit(pytest.main([__file__, "-q"]))
