"""MRD land-cover classification for Fiji.

MRD is Mineral Resource Detection: land-cover maps for finding and monitoring mineral resources such as quarries.
"""

# The one place the product version is set. CLIs default to it; the Argo workflow passes it as a parameter.
VERSION = "0-0-2"

OUTPUT_BUCKET = "dep-public-staging"
MODEL_PREFIX = "dep_s2s1_mrd/models"
SENSOR = "s2s1"
DATASET_ID = "mrd"

DEP_CATALOG = "https://stac.digitalearthpacific.org"
DEM_CATALOG = "https://planetarycomputer.microsoft.com/api/stac/v1/"
S2_COLLECTION = "dep_s2_geomad"
S1_COLLECTION = "dep_s1_geomad"
DEM_COLLECTION = "cop-dem-glo-30"

# Output nodata for both the class and proba bands.
NODATA = 255

# Class value: (name, colour).
CLASSES = {
    1: ("bare_land", "#968640"),
    2: ("forest", "#064a00"),
    3: ("crops", "#ffce33"),
    4: ("grassland", "#d7ffa0"),
    5: ("settlements", "#b3b2ae"),
    6: ("mangroves", "#07b28d"),
    7: ("water", "#71a8ff"),
    8: ("quarry", "#b03a2e"),
}


def tile_index(tile_id: str) -> tuple[int, int]:
    """A dep grid tile id "col,row" as (col, row)."""
    col, row = tile_id.split(",")
    return int(col), int(row)


def model_url(version: str) -> str:
    """Where `mrd train` publishes the model for a version. Always the staging bucket, whatever the output bucket."""
    return f"https://{OUTPUT_BUCKET}.s3.us-west-2.amazonaws.com/{MODEL_PREFIX}/{version}/model.joblib"


MODEL_URL = model_url(VERSION)


def collection_url_root(bucket: str) -> str:
    """The STAC collections endpoint that output items in a bucket belong to."""
    if bucket == "dep-public-prod":
        return "https://stac.digitalearthpacific.org/collections"
    return "https://stac.staging.digitalearthpacific.io/collections"
