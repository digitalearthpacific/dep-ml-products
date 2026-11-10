# Digital Earth Pacific ML Products: MRD land cover

Land-cover classification for Fiji. MRD is **Mineral Resource Detection**: the maps help find and monitor mineral
resources such as quarries. A random forest classifies every 10 m pixel of each dep grid tile, for each year, into:

| Value | Class |
|---|---|
| 1 | bare_land |
| 2 | forest |
| 3 | crops |
| 4 | grassland |
| 5 | settlements |
| 6 | mangroves |
| 7 | water |
| 8 | quarry |

The approach (tile grid, antimeridian handling, outputs) follows [auspatious/ldn-lulc](https://github.com/auspatious/ldn-lulc).
The full spec is [docs/prd/fiji-landcover-0-0-2.md](docs/prd/fiji-landcover-0-0-2.md).

## Outputs

Per tile and year, under `s3://dep-public-staging/dep_s2s1_mrd/<version>/<col>/<row>/<year>/`:

- `..._class.tif`: class value, uint8, nodata 255.
- `..._proba.tif`: probability of the winning class (0-100), uint8, nodata 255.
- `....stac-item.json`: written last, so it marks a finished tile. It links the model used and records the image tag.

Writing to `dep-public-prod` later only needs `--output-bucket dep-public-prod`. The output bucket must be public-read:
the STAC item is built by reading the written COGs back over HTTPS.

## Inputs and their limits

- DEP Sentinel-2 and Sentinel-1 geomads (`dep_s2_geomad`, `dep_s1_geomad`) from https://stac.digitalearthpacific.org.
- Copernicus DEM GLO-30 from Planetary Computer, for elevation, slope, northness and eastness.
- **Sentinel-1 2025 isn't available yet** (`dep_s1_geomad` has 2017-2024). The model needs S1, so 2025 tasks log
  "No data" and exit without writing. When DEP publishes it, run the workflow with `-p years=2025`.
- DEP's S2 geomad has no data in a band about 30 km wide around 180°, so those pixels (e.g. parts of Taveuni) are
  nodata in every year.

## Setup

```bash
uv sync
```

## Train

The model is trained once, by hand, on `data/mred-tdata.geojson` (2022 labels and imagery).

**Limitation: the training data isn't in this repository.** The labelled points are kept out of this public repo.
They live in [nicholasmetherall/minerals-model](https://github.com/nicholasmetherall/minerals-model), so you need
access to that repo to train. `make training-data` downloads them to `data/` (git-ignored); set `GITHUB_TOKEN` if the
repo is private. Classifying with a published model doesn't need them.


```bash
uv run mrd train --year 2022 --no-upload   # local only: data/models/<version>/
uv run mrd train --year 2022               # also publishes to s3://dep-public-staging/dep_s2s1_mrd/models/<version>/
```

Run it from the repo root. Publishing refuses to replace a model already published for that version unless you pass
`--overwrite`. `notebooks/1-ml-learning.ipynb` explores the same steps.

## Classify

One tile locally (needs AWS credentials that can write to the output bucket):

```bash
uv run mrd run --tile-id 64,20 --year 2022 --overwrite
uv run mrd print-tasks --years 2017-2024 --overwrite   # the task list the workflow uses
```

`notebooks/2-prediction.ipynb` classifies a small window without writing anything.

## Run the workflow

```bash
argo submit workflow/ml-mrd-annual.yaml                          # Fiji, 2017-2024
argo submit workflow/ml-mrd-annual.yaml -p years=2022            # one year
```

- Pods run the image tagged with the workflow's `version` (e.g. `ghcr.io/digitalearthpacific/dep-ml-products:0-0-2`).
  CI pushes that tag (plus `git describe` and `latest` tags) for `main`, `aws` and releases, so merge or release
  before running. The `version` tag is rebuilt on every push there; the `git describe` tag pins an exact build.
- The model for `version` must be published first (`mrd train`); otherwise each task fails loading it.
- Fiji for 8 years is 216 tasks (27 tiles).
- Pod resources (32Gi, 8 CPU) are an unmeasured estimate. Re-set them from a measured single-tile run.

## Versions

The version is `version` in `pyproject.toml` (`0.0.2`). With dashes (`0-0-2`) it is `mrd.VERSION`, the S3 output and
model folders, and the Docker image tag. To release a new version, bump it in `pyproject.toml` and the workflow's
`version` parameter together (a test checks they match), then train and publish a model for it.

## Docker

The image is `linux/amd64` only (`fiona`, needed by dep-tools, has no ARM Linux wheel). It's a plain Python image:
GDAL comes bundled in the `rasterio` and `fiona` wheels. On Apple Silicon, `make build` builds it under emulation.

## Development

```bash
make test                                    # offline tests
make lint
uv run python tests/test_data.py network    # loads real data across the antimeridian (slow)
```


## Training Data

Will is in comms with MRD for newer quarries and tenements data.
https://www.lands.gov.fj/
