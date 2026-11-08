"""MRD land-cover classification for Fiji.

MRD is Fiji's Mineral Resources Department, the partner this product is built for.
"""

# The one place the product version is set. CLIs default to it; the Argo workflow passes it as a parameter.
VERSION = "0-0-2"

OUTPUT_BUCKET = "dep-public-staging"
MODEL_PREFIX = "dep_s2s1_mrd/models"
MODEL_URL = f"https://{OUTPUT_BUCKET}.s3.us-west-2.amazonaws.com/{MODEL_PREFIX}/{VERSION}/model.joblib"

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
