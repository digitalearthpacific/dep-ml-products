"""Index a version's STAC items into one STAC-GeoParquet file next to them."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dep_tools.aws import object_exists
from rustac import write_sync

from mrd import DATASET_ID, OUTPUT_BUCKET, SENSOR, VERSION
from mrd.aws import s3_client
from mrd.data import NoDataError

logger = logging.getLogger(__name__)


def find_stac_keys(bucket: str, prefix: str) -> list[str]:
    """Keys of the STAC items under bucket/prefix."""
    pages = s3_client().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
    return [o["Key"] for page in pages for o in page.get("Contents", []) if o["Key"].endswith(".stac-item.json")]


def load_stac_items(bucket: str, keys: list[str]) -> list[dict]:
    """The STAC item documents at keys, in the same order, fetched in parallel."""
    client = s3_client()  # boto3 clients are thread safe

    def load(key: str) -> dict:
        return json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())

    with ThreadPoolExecutor(max_workers=32) as pool:
        return list(pool.map(load, keys))


def index_to_stac_geoparquet(
    bucket: str = OUTPUT_BUCKET, version: str = VERSION, upload: bool = True, overwrite: bool = False
) -> Path:
    """Write the version's STAC items to data/<prefix>/stac.parquet, and upload it to <prefix>/stac.parquet.

    rustac writes locally and boto3 uploads, so credentials work the same as everywhere else (see mrd.aws).

    Raises:
        FileExistsError: If uploading without overwrite and the index already exists (checked first).
        NoDataError: If the version has no STAC items.
    """
    prefix = f"dep_{SENSOR}_{DATASET_ID}/{version}/"
    key = f"{prefix}stac.parquet"
    client = s3_client()
    if upload and not overwrite and object_exists(bucket, key, client):
        raise FileExistsError(f"s3://{bucket}/{key} already exists (use --overwrite to replace)")

    keys = find_stac_keys(bucket, prefix)
    logger.info(f"Found {len(keys)} STAC items under s3://{bucket}/{prefix}")
    if not keys:
        raise NoDataError(f"No STAC items under s3://{bucket}/{prefix}")
    items = load_stac_items(bucket, keys)

    local = Path("data") / key
    local.parent.mkdir(parents=True, exist_ok=True)
    write_sync(str(local), items)
    logger.info(f"Wrote {len(items)} items to {local}")

    if upload:
        client.upload_file(str(local), bucket, key, ExtraArgs={"ContentType": "application/vnd.apache.parquet"})
        logger.info(f"Uploaded s3://{bucket}/{key}")
    return local
