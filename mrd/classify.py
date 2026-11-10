"""Classify dep grid tiles with the published model and write the results to S3.

`mrd print-tasks` lists the tile and year tasks for the Argo workflow, and `mrd run` classifies one of them: features
from mrd.data.load_data, the model from mrd.data.load_model, then class and proba COGs and a STAC item.
"""

import json
import logging
from pathlib import Path

import dask
import geopandas as gpd
from antimeridian import fix_shape
from botocore.exceptions import ClientError
from dep_tools.aws import object_exists, write_stac_s3
from dep_tools.grids import PACIFIC_GRID_10
from dep_tools.namers import S3ItemPath
from dep_tools.stac_utils import get_stac_item
from dep_tools.utils import bbox_across_180
from dep_tools.writers import AwsDsCogWriter
from odc.geo.geobox import GeoBox
from pystac import Link

from mrd import DATASET_ID, OUTPUT_BUCKET, SENSOR, TILES_URL, VERSION, collection_url_root, model_url, tile_index
from mrd.aws import s3_client
from mrd.data import NoDataError, load_data, load_model, predict

logger = logging.getLogger(__name__)


def item_path(bucket: str, version: str, year: int) -> S3ItemPath:
    """Output paths, e.g. dep_s2s1_mrd/<version>/066/020/2022/dep_s2s1_mrd_066_020_2022_class.tif."""
    return S3ItemPath(
        bucket=bucket,
        sensor=SENSOR,
        dataset_id=DATASET_ID,
        version=version,
        time=str(year),
        prefix="dep",
        # Given explicitly, so dep-tools doesn't look up the bucket's region over the network.
        full_path_prefix=f"https://{bucket}.s3.us-west-2.amazonaws.com/",
    )


def get_tiles(regions: list[str] | None) -> list[str]:
    """dep grid tile ids ("col,row") for country codes such as ["FJ"], or every tile for None, sorted by (col, row)."""
    tiles = gpd.read_file(TILES_URL).astype({"tile_id": str, "country_code": str})
    if regions is not None:
        unknown = set(regions) - set(tiles.country_code)
        if unknown:
            raise ValueError(f"No dep grid tiles for regions {sorted(unknown)}")
        tiles = tiles[tiles.country_code.isin(regions)]
    return sorted(tiles.tile_id, key=tile_index)


def parse_years(years: str) -> list[int]:
    """ "2022", "2017-2024" (inclusive) or "2017,2020"."""
    try:
        if "-" in years:
            start, end = (int(y) for y in years.split("-"))
            if start > end:
                raise ValueError
            return list(range(start, end + 1))
        return [int(y) for y in years.split(",")]
    except ValueError:
        raise ValueError(f'Invalid years "{years}". Use e.g. "2022", "2017-2024" or "2017,2020".') from None


def existing_stac_keys(bucket: str, version: str) -> set[str]:
    """Keys of the STAC items already written for a version, from one listing of its prefix.

    Without permission to list the bucket, nothing is filtered: each `mrd run` still skips finished tiles itself.
    Any other error, e.g. a bucket that doesn't exist, is raised.
    """
    prefix = f"dep_{SENSOR}_{DATASET_ID}/{version}/"
    keys = set()
    try:
        for page in s3_client().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            keys.update(o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".stac-item.json"))
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") not in ("AccessDenied", "403"):
            raise
        logger.warning(f"Can't list existing items ({e}); not filtering, every task is included")
        return set()
    return keys


def print_tasks(
    years: str,
    regions: str | None = "FJ",
    version: str = VERSION,
    bucket: str = OUTPUT_BUCKET,
    overwrite: bool = False,
    limit: int | None = None,
) -> str:
    """The tile and year tasks as compact JSON for Argo's withParam: [{"tile-id", "year", "version"}].

    regions is a comma-separated list of country codes, or "ALL" (or None) for every tile. Without overwrite, tasks
    whose STAC item exists are left out.

    Raises:
        ValueError: For invalid years or regions.
    """
    region_codes = None if regions is None or regions.upper() == "ALL" else regions.split(",")
    tiles, year_list = get_tiles(region_codes), parse_years(years)
    tasks = [{"tile-id": tile, "year": year, "version": version} for tile in tiles for year in year_list]
    if not overwrite:
        existing = existing_stac_keys(bucket, version)
        tasks = [t for t in tasks if item_path(bucket, version, t["year"]).stac_path(t["tile-id"]) not in existing]
    if limit is not None:
        tasks = tasks[:limit]
    return json.dumps(tasks, separators=(",", ":"))


def antimeridian_safe_bbox(area: GeoBox, fallback_bbox: list[float]) -> list[float]:
    """STAC bbox for an area, as [east edge, ymin, west edge, ymax] (bbox[0] > bbox[2]) if it crosses 180.

    rio_stac derives the bbox from the raster's reprojected corners, which for a tile crossing 180 collapses to a
    near-global box. Ported from ldn-lulc (ldn/raster.py:554-573).
    """
    halves = bbox_across_180(area)
    if not isinstance(halves, tuple):
        return fallback_bbox
    east, west = halves
    return [east[0], min(east[1], west[1]), west[2], max(east[3], west[3])]


def antimeridian_safe_geometry(area: GeoBox, geometry: dict) -> dict:
    """STAC geometry split at 180 if the area crosses it, otherwise unchanged (ldn/raster.py:576-590)."""
    return fix_shape(geometry) if isinstance(bbox_across_180(area), tuple) else geometry


def run(
    tile_id: str,
    year: int,
    version: str = VERSION,
    model_path: str | None = None,
    bucket: str = OUTPUT_BUCKET,
    overwrite: bool = False,
    image_tag: str = "unknown",
    threads: int | None = None,
) -> list[str] | None:
    """Classify one tile and year, and write class and proba COGs and a STAC item. Returns the written keys, or None
    if the tile was skipped (already done) or has no input data.
    """
    geobox = PACIFIC_GRID_10.tile_geobox(tile_index(tile_id))
    itempath = item_path(bucket, version, year)
    stac_key = itempath.stac_path(tile_id)
    client = s3_client()
    # dep-tools' object_exists treats any error as missing, so without s3:ListBucket a finished tile is redone
    # (overwritten with the same output) rather than failing.
    if not overwrite and object_exists(bucket, stac_key, client):
        logger.info(f"Skipping {tile_id} {year}: {stac_key} already exists")
        return None

    model_source = model_path or model_url(version)
    model = load_model(model_source)
    try:
        ds = load_data(geobox, year)
    except NoDataError as e:
        logger.warning(f"No data for tile {tile_id} in {year}, nothing written: {e}")
        return None

    out = predict(model, ds)
    out.attrs["stac_properties"] = {
        "datetime": f"{year}-01-01T00:00:00Z",
        "start_datetime": f"{year}-01-01T00:00:00Z",
        "end_datetime": f"{year}-12-31T23:59:59Z",
    }
    # Threads, not dask.distributed: every block shares the one in-memory model instead of a pickled copy per worker.
    with dask.config.set(scheduler="threads", num_workers=threads):
        out = out.compute()
    paths = AwsDsCogWriter(itempath, write_multithreaded=True, client=client).write(out, tile_id)

    # Written last, so an item's existence means the tile is complete.
    item = get_stac_item(itempath, tile_id, out, collection_url_root=collection_url_root(bucket), with_raster=True)
    item.bbox = antimeridian_safe_bbox(geobox, item.bbox)
    item.geometry = antimeridian_safe_geometry(geobox, item.geometry)
    if model_source.startswith("https://"):
        item.add_link(Link(rel="related", target=model_source, media_type="application/octet-stream", title="model"))
        item.properties["mrd:model"] = model_source
    else:
        # A local model has no URL readers could follow, so record where it came from without a link.
        item.properties["mrd:model"] = Path(model_source).resolve().as_uri()
    item.properties["mrd:image_tag"] = image_tag
    write_stac_s3(item, stac_key, bucket, client=client)
    logger.info(f"Wrote {tile_id} {year}: {paths + [stac_key]}")
    return paths + [stac_key]
