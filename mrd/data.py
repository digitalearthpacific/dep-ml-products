"""Feature loading, sampling and prediction shared by model training and whole-country classification.

Training (``load_data`` then ``sample_points``) and classification (``load_data`` then ``predict``) both build their
features here, so the model always sees features computed the same way.

Features were checked against auspatious/ldn-lulc (commit faaa212), which is the more robust implementation, and
aligned with it where it handles something better. Citations are to that repo.

=============================  ==========================================================================
Feature(s)                     Decision
=============================  ==========================================================================
S2 reflectance (9 bands)       Ported: scaled in place to float32 reflectance, clipped to [0, 1], STAC
                               nodata -> NaN, error if nodata isn't declared. Replaces the raw uint16
                               bands. ``scale_offset_landsat``, ldn/raster.py:57-92 (S2 scale 0.0001,
                               offset 0 instead of Landsat's).
safe_divide                    Ported: NaN where the denominator is 0 or an input is NaN.
                               ldn/raster.py:95-109.
ndvi, ndwi, bsi                Kept: same formulas as ``calculate_indices``, ldn/raster.py:136-147.
mndwi                          Changed to green and swir16 (Xu 2006); the prototype used swir22.
                               ldn/raster.py:140.
ndre21                         Renamed from the prototype's "ndbi", which was computed from red-edge 2 and
                               1. True NDBI is -NDMI and ldn doesn't store it either
                               (ldn/raster.py:146-147). Not in ldn (Landsat has no red-edge).
evi, savi, ndmi, ndre          Kept from the prototype, using the ported safe_divide. Not in ldn: S2
                               specific. evi stays clipped to [-1, 1] as its denominator nears 0 on
                               bright pixels.
ndti, mbi, baei, bui           Not added: new features are out of scope, and ldn's TODO flags them as
                               possibly redundant (ldn/raster.py:126-129).
emad, bcmad, smad              Kept raw, as ldn does (ldn/raster.py:46, 66-68).
mean_vv, mean_vh, mean_vv_vh   Kept (S1 linear power and their ratio). Not in ldn, which has no radar.
elevation, slope,              Ported ``_calculate_terrain_features`` (ldn/raster.py:159-192): computed on
northness, eastness            the DEM's native WGS84 grid with central differences and ellipsoid radii;
                               northness and eastness are weighted by sin(slope). Replaces the prototype's
                               Horn/Sobel in EPSG:3832, which had Mercator scale error, reflected tile
                               edges and full-strength aspect on flat ground.
DEM across the antimeridian    Ported ``_load_halves`` and ``_merge_geographic_halves``
                               (ldn/raster.py:214-325). Differs: a side with no DEM items (an all-ocean
                               1 degree cell) is NaN instead of an error, as is a tile with no DEM items.
Padding and crop               Only the DEM is padded (about 100 m), because only terrain uses neighbouring
                               pixels; S2 and S1 features are per pixel. Reprojecting onto the unpadded
                               geobox is the crop. ldn pads 2 px at 30 m (ldn/raster.py:369) but searches
                               unpadded; this searches with the padded area so no DEM tile is missed.
=============================  ==========================================================================
"""

import logging
import os
import tempfile
import warnings
from math import ceil
from pathlib import Path
from urllib.parse import urlparse

import dask.array as da
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import requests
import xarray as xr
from affine import Affine
from dep_tools.utils import bbox_across_180, search_across_180
from odc.geo import Geometry
from odc.geo.geobox import GeoBox, bounding_box_in_pixel_domain
from odc.geo.xr import assign_crs, xr_coords
from odc.stac import load as stac_load
from planetary_computer import sign_url
from pystac import Item
from pystac_client import Client
from shapely.geometry import box, shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning

from mrd import CLASSES, DEM_CATALOG, DEM_COLLECTION, DEP_CATALOG, NODATA, S1_COLLECTION, S2_COLLECTION

logger = logging.getLogger(__name__)

S2_REFLECTANCE = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "swir16", "swir22"]
S2_MAD = ["emad", "bcmad", "smad"]
S1_BANDS = ["mean_vv", "mean_vh"]
INDICES = ["ndvi", "ndwi", "mndwi", "bsi", "evi", "savi", "ndmi", "ndre", "ndre21"]
TERRAIN = ["elevation", "slope", "northness", "eastness"]
# Model features, in order.
FEATURES = S2_REFLECTANCE + S2_MAD + INDICES + S1_BANDS + ["mean_vv_vh"] + TERRAIN

# DEP S2 geomad uint16 -> surface reflectance (STAC raster:bands).
S2_SCALE = 0.0001
S2_OFFSET = 0.0

CHUNKS = {"x": 2048, "y": 2048}
WGS84 = "EPSG:4326"
# WGS84 with its prime meridian at 180, so a grid across the antimeridian is continuous.
PM180_CRS = "+proj=longlat +datum=WGS84 +pm=180"
# DEM margin around the target area, so terrain gradients at its edge use real neighbours (about 3 GLO-30 pixels).
DEM_PAD_METRES = 100


class NoDataError(Exception):
    """No S2 or S1 input exists for the area and year."""


def safe_divide(num: xr.DataArray, den: xr.DataArray) -> xr.DataArray:
    """Divide, giving NaN where the denominator is 0 or either input is NaN."""
    defined = num.notnull() & den.notnull() & (den != 0)
    return num.where(defined) / den.where(defined)


def scale_reflectance(ds: xr.Dataset) -> xr.Dataset:
    """Scale the S2 reflectance bands in place to float32 in [0, 1], with nodata as NaN.

    Raises:
        ValueError: If a band has no declared nodata (load with odc-stac so the STAC nodata is set).
    """
    for band in S2_REFLECTANCE:
        raw = ds[band]
        if raw.odc.nodata is None:
            raise ValueError(f"Band {band} has no declared nodata. Load it with odc-stac so the STAC nodata is set.")
        scaled = (raw * S2_SCALE + S2_OFFSET).clip(0, 1).astype("float32")
        ds[band] = scaled.where(raw != raw.odc.nodata)
    return ds


def add_indices(ds: xr.Dataset) -> xr.Dataset:
    """Add spectral indices to a dataset of scaled S2 reflectance (see scale_reflectance), in place."""
    blue, green, red, nir = ds.blue, ds.green, ds.red, ds.nir
    re1, re2, swir1 = ds.rededge1, ds.rededge2, ds.swir16
    # Normalised Difference Vegetation Index (Rouse 1974).
    ds["ndvi"] = safe_divide(nir - red, nir + red)
    # Normalised Difference Water Index (McFeeters 1996).
    ds["ndwi"] = safe_divide(green - nir, green + nir)
    # Modified Normalised Difference Water Index (Xu 2006).
    ds["mndwi"] = safe_divide(green - swir1, green + swir1)
    # Bare Soil Index (Rikimaru 2002).
    ds["bsi"] = safe_divide((swir1 + red) - (nir + blue), (swir1 + red) + (nir + blue))
    # Enhanced Vegetation Index. The denominator nears 0 on bright pixels, so clip to the meaningful range.
    ds["evi"] = safe_divide(2.5 * (nir - red), nir + 6 * red - 7.5 * blue + 1.0).clip(-1, 1)
    # Soil Adjusted Vegetation Index (L = 0.5).
    ds["savi"] = safe_divide(1.5 * (nir - red), nir + red + 0.5)
    # Normalised Difference Moisture Index.
    ds["ndmi"] = safe_divide(nir - swir1, nir + swir1)
    # Normalised Difference Red Edge index.
    ds["ndre"] = safe_divide(nir - re1, nir + re1)
    # Normalised difference of red-edge 2 and red-edge 1.
    ds["ndre21"] = safe_divide(re2 - re1, re2 + re1)
    return ds


def terrain_features(dem: xr.DataArray) -> xr.Dataset:
    """Elevation, slope (degrees), northness and eastness from elevation on a geographic (degree) grid.

    Central differences in degrees, converted to ground metres with the ellipsoid's radii of curvature at each
    latitude. Northness and eastness are cos/sin of aspect (the downslope bearing) weighted by sin(slope): 1 = steep
    north/east-facing, -1 = steep south/west-facing, 0 on flat ground.
    """
    geod = dem.odc.crs.proj.get_geod()
    lat = np.radians(dem.y)
    w = 1 - geod.es * np.sin(lat) ** 2
    east_radius = geod.a / np.sqrt(w)
    north_radius = geod.a * (1 - geod.es) / w**1.5
    dz_dx = dem.differentiate("x") / (np.radians(1) * east_radius * np.cos(lat))
    dz_dy = dem.differentiate("y") / (np.radians(1) * north_radius)
    grad = np.hypot(dz_dx, dz_dy)
    hypotenuse = np.sqrt(1 + grad**2)
    terrain = {"slope": np.degrees(np.arctan(grad)), "northness": -dz_dy / hypotenuse, "eastness": -dz_dx / hypotenuse}
    return xr.Dataset({name: v.astype("float32") for name, v in {"elevation": dem, **terrain}.items()})


def _overlaps(footprint: dict, geobox: GeoBox) -> bool:
    """Whether an EPSG:4326 item footprint overlaps geobox, not just touches its edge.

    The extent is shrunk by half a pixel so floating point slivers from the EPSG:4326 round trip don't keep
    neighbouring tiles.
    """
    inner = geobox.extent.buffer(-abs(geobox.resolution.x) / 2)
    return Geometry(footprint, WGS84).to_crs(geobox.crs).intersects(inner)


def _search(client: Client, geobox: GeoBox, **kwargs) -> list[Item]:
    """Search across the antimeridian, keeping only items that overlap geobox (see _overlaps)."""
    return [item for item in search_across_180(geobox, client=client, **kwargs) if _overlaps(item.geometry, geobox)]


def _shift_to_pm180(half: xr.Dataset, shift: float) -> GeoBox:
    """The geobox of a WGS84 half with longitudes relative to 180. Only the affine changes, so no pixels move."""
    return GeoBox(half.odc.geobox.shape, Affine.translation(shift, 0) * half.odc.geobox.affine, PM180_CRS)


def _assign_geobox(data: xr.Dataset, geobox: GeoBox) -> xr.Dataset:
    data = data.assign_coords(xr_coords(geobox)).rename(longitude="x", latitude="y")
    return assign_crs(data, geobox.crs)


def _merge_geographic_halves(east: xr.Dataset, west: xr.Dataset) -> xr.Dataset:
    """Join WGS84 halves either side of the antimeridian into one continuous +pm=180 grid, by pixel position.

    Ported from ldn/raster.py:295-325. Coordinates from the two loads can't be matched by value: odc gives
    -179.99999999999997 for -180, which would leave a duplicate column.
    """
    east_box, west_box = _shift_to_pm180(east, -180), _shift_to_pm180(west, 180)
    # East columns before the west half's first column, in whole pixels. Raises if not on the same pixel grid.
    west_start = bounding_box_in_pixel_domain(west_box, east_box).left
    merged_box = GeoBox((east_box.height, west_start + west_box.width), east_box.affine, PM180_CRS)
    merged = xr.concat([east.isel(longitude=slice(0, west_start)), west], dim="longitude", join="override")
    return _assign_geobox(merged, merged_box)


def _load_wgs84(items: list[Item], area: Geometry) -> xr.Dataset:
    return stac_load(items, geopolygon=area, patch_url=sign_url, chunks={}).squeeze(drop=True)


def _load_dem(items: list[Item], area: GeoBox) -> xr.DataArray:
    """Load DEM elevation on its native WGS84 grid over area, joining the sides if it crosses the antimeridian.

    A side with no items (an all-ocean 1 degree cell has no GLO-30 tile) is left out, so it is NaN after reprojection.
    """
    bbox = bbox_across_180(area)
    if not isinstance(bbox, tuple):
        dem = _load_wgs84(items, area.footprint(WGS84)).rename(longitude="x", latitude="y")
    else:
        sides = [box(*b) for b in bbox]
        # An EPSG:4326 item footprint can't cross 180, so it intersects only one side.
        east_items, west_items = ([i for i in items if shape(i.geometry).intersects(side)] for side in sides)
        east, west = (
            _load_wgs84(side_items, Geometry(side, WGS84)) if side_items else None
            for side_items, side in zip([east_items, west_items], sides)
        )
        if east is not None and west is not None:
            dem = _merge_geographic_halves(east, west)
        else:
            logger.warning("DEM has no items on one side of the antimeridian; that side's terrain is NaN")
            half, shift = (east, -180) if east is not None else (west, 180)
            dem = _assign_geobox(half, _shift_to_pm180(half, shift))
    # Compute once: the DEM is small at 30 m, and every use would otherwise download it again.
    return dem["data"].rename("elevation").compute()


def _dask_chunks(chunks: dict) -> dict | str:
    return chunks or "auto"


def load_dem_terrain(geobox: GeoBox, chunks: dict) -> xr.Dataset:
    """Copernicus DEM GLO-30 terrain features on geobox: computed on the DEM's native grid, then reprojected.

    The DEM itself is fetched eagerly; the reprojected features are dask arrays.
    """
    area = geobox.pad(max(2, ceil(DEM_PAD_METRES / abs(geobox.resolution.x))))
    items = _search(Client.open(DEM_CATALOG), area, collections=[DEM_COLLECTION])
    logger.info(f"Found {len(items)} DEM items")
    if len(items) >= 10:
        raise RuntimeError(f"Too many DEM items ({len(items)}). Expected ~4, the search may be world-spanning.")
    if not items:
        logger.warning("No DEM items; terrain is NaN")
        nan = da.full(tuple(geobox.shape), np.nan, dtype="float32")
        terrain = xr.Dataset({n: (("y", "x"), nan) for n in TERRAIN}, coords=xr_coords(geobox))
        return assign_crs(terrain, geobox.crs).chunk(_dask_chunks(chunks))
    terrain = terrain_features(_load_dem(items, area))
    return terrain.chunk().odc.reproject(geobox, resampling="nearest").chunk(_dask_chunks(chunks))


def load_data(geobox: GeoBox, year: int, chunks: dict | None = None) -> xr.Dataset:
    """Load the model features (FEATURES, in order) for one area and year, on exactly geobox.

    S2 and S1 geomads are on the dep grid (EPSG:3832), so they load onto geobox without reprojection, including
    across the antimeridian. The DEM is fetched eagerly (see load_dem_terrain); everything else is lazy.

    Raises:
        NoDataError: If no S2 or no S1 item overlaps geobox for the year.
    """
    chunks = CHUNKS if chunks is None else chunks
    client = Client.open(DEP_CATALOG)

    def load(collection: str, bands: list[str]) -> xr.Dataset:
        items = _search(client, geobox, collections=[collection], datetime=str(year))
        logger.info(f"Found {len(items)} {collection} items")
        if not items:
            raise NoDataError(f"No {collection} items for {year} over {bbox_across_180(geobox)}")
        return stac_load(items, geobox=geobox, bands=bands, chunks=chunks).squeeze("time", drop=True)

    s2 = add_indices(scale_reflectance(load(S2_COLLECTION, S2_REFLECTANCE + S2_MAD)))
    s1 = load(S1_COLLECTION, S1_BANDS)
    s1["mean_vv_vh"] = safe_divide(s1.mean_vv, s1.mean_vh)
    dem = load_dem_terrain(geobox, chunks)
    return xr.merge([s2, s1, dem], join="exact", compat="override")[FEATURES]


def sample_points(ds: xr.Dataset, points: gpd.GeoDataFrame) -> pd.DataFrame:
    """Feature values at the pixels nearest each point inside ds, one row per point, indexed like points.

    Points outside ds are dropped. NaN values are kept; the caller decides what to drop.
    """
    points = points.to_crs(ds.odc.crs)
    points = points[points.within(ds.odc.geobox.extent.geom)]
    selected = ds.sel(
        x=xr.DataArray(points.geometry.x.to_numpy(), dims="point"),
        y=xr.DataArray(points.geometry.y.to_numpy(), dims="point"),
        method="nearest",
    ).compute()
    columns = list(ds.data_vars)
    return pd.DataFrame({name: selected[name].to_numpy() for name in columns}, index=points.index, columns=columns)


def predict(model: RandomForestClassifier, ds: xr.Dataset) -> xr.Dataset:
    """Classify every pixel of ds, block by block.

    Returns ``class`` (uint8 class value) and ``proba`` (uint8 winning-class probability, 0-100), both with nodata
    NODATA. A pixel is nodata where any S2 reflectance band is NaN (no optical observation). Other NaN features, such
    as S1 gaps or the DEM over open ocean, are passed to the random forest, which supports missing values.

    Raises:
        ValueError: If ds's variables aren't exactly the model's features.
    """
    names = list(model.feature_names_in_)
    missing, extra = set(names) - set(ds.data_vars), set(ds.data_vars) - set(names)
    if missing or extra:
        raise ValueError(f"Dataset features don't match the model. Missing: {sorted(missing)}, extra: {sorted(extra)}")
    s2_columns = [names.index(band) for band in S2_REFLECTANCE]

    def classify(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # numpy block (..., feature)
        flat = x.reshape(-1, x.shape[-1]).astype("float32")
        flat[~np.isfinite(flat)] = np.nan  # sklearn rejects inf
        valid = ~np.isnan(flat[:, s2_columns]).any(axis=1)
        classes = np.full(len(flat), NODATA, "uint8")
        proba = np.full(len(flat), NODATA, "uint8")
        if valid.any():
            p = model.predict_proba(pd.DataFrame(flat[valid], columns=names))
            classes[valid] = model.classes_[p.argmax(axis=1)]
            proba[valid] = np.round(p.max(axis=1) * 100)
        return classes.reshape(x.shape[:-1]), proba.reshape(x.shape[:-1])

    classes, proba = xr.apply_ufunc(
        classify,
        ds[names].to_dataarray("feature").chunk(feature=-1),
        input_core_dims=[["feature"]],
        output_core_dims=[[], []],
        dask="parallelized",
        output_dtypes=["uint8", "uint8"],
    )
    out = xr.Dataset({"class": classes, "proba": proba})
    for name in out.data_vars:
        out[name].attrs["nodata"] = NODATA
    return out


def load_model(url: str) -> RandomForestClassifier:
    """Load the model from an HTTPS URL (downloaded once to ~/.cache/mrd) or a local path.

    The cache is keyed by URL, which includes the version, and is never revalidated: versions are immutable, so
    delete the cached file to pick up a model republished under the same version. s3:// URLs aren't supported; use
    the bucket's HTTPS URL.

    Raises:
        InconsistentVersionWarning: As an error, if the model was pickled with a different scikit-learn version.
        ValueError: If the model has no feature names or has class values outside CLASSES.
    """
    path = Path(url)
    if url.startswith("https://"):
        parsed = urlparse(url)
        path = Path.home() / ".cache" / "mrd" / parsed.netloc / parsed.path.lstrip("/")
        if not path.exists():
            logger.info(f"Downloading model from {url} to {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            # Write then rename, so an interrupted download never leaves a truncated model in the cache.
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                tmp.write(response.content)
            os.replace(tmp.name, path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        model = joblib.load(path)
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Model has no feature names. Fit it on a DataFrame with the FEATURES columns.")
    if not set(model.classes_.tolist()) <= set(CLASSES):
        raise ValueError(f"Model classes {model.classes_.tolist()} aren't all in {sorted(CLASSES)}")
    return model
