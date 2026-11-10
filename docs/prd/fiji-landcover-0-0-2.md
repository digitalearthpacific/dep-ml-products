# PRD: Fiji land-cover classification v0-0-2

**Naming.** `mrd` (as in `dep_s2s1_mrd`) stands for **Mineral Resource Detection** (confirmed by the user): land-cover maps for finding and monitoring mineral resources such as quarries. Everything in this project uses `mrd`, never "gravel".

## Problem

The committed pipeline (`src/run_task.py`, `src/print_tasks.py`, `models/*.dump`, `workflow/ml-gravel-annual-0-0-1.yaml`) produces outputs that the README itself says "don't look good". The model files have no record of their features or sklearn version, and the feature set is implicit. A newer prototype in `notebooks/` (uncommitted) works, but:

- it only covers a 2022 dev subset of Fiji;
- it saves the model to a local file;
- its training and prediction code are separate notebook cells, not shared functions.

We need one production path: train a single model on all Fiji training data, publish it to S3, and classify every Fiji tile for many years via Argo. Training and classification must use the same code.

## Users

- **Pipeline operator (DEP team):** trains the model once by hand and submits the annual Argo workflow.
- **Fiji mineral resources stakeholders:** the client for the Mineral Resource Detection (MRD) land-cover maps.
- **Downstream consumers:** read `dep_s2s1_mrd` COGs and STAC items from `dep-public-staging`.

## Current state (observed)

### Committed production code

- **CLIs:** typer CLIs `src/run_task.py:138-150` and `src/print_tasks.py:12-20`.
  - Tiles come from dep-grid `grid_pacific.geojson` filtered by `country_code` (`src/run_task.py:30-37`, `src/print_tasks.py:37-38`).
  - Years are given as `"A-B"` ranges (`src/print_tasks.py:30-34`).
- **Model loading:** a local joblib path, reloaded on every process call (`src/run_task.py:106`). The image only has it because of `COPY . .` (`Dockerfile:20`).
- **Features:** whatever bands end up merged from `dep_s1_mosaic` and `dep_s2_geomad` on the **staging** STAC, plus elevation and 7 indices (`src/run_task.py:54-87, 214-244`). No explicit feature list.
- **Outputs:** written with dep-tools v0.5.3 as `class` and `proba` COGs plus a STAC item. The STAC item's existence is the skip check (`src/run_task.py:169-203, 256-259`). Existing layout: `s3://dep-public-staging/dep_s2s1_mrd/0-0-1/061/021/2017/dep_s2s1_mrd_061_021_2017_{class,proba}.tif`.
- **Error handling:**
  - Every `ValueError` is logged as "Failed to find Sentinel-1 data" and the task exits successfully (`src/run_task.py:267-269`).
  - `print_tasks.py:34` creates a `ValueError` but never raises it.
  - `--regions ALL` crashes (`src/print_tasks.py:24,38`).
- **Dependencies:** `scikit-learn==1.5.1` (`pyproject.toml:12`).
- **Workflow file:**
  - Version `"0.0.1"` (`workflow/ml-gravel-annual-0-0-1.yaml:27`).
  - Years `"2023"` and regions `FJ` are hardcoded in the DAG (`:49-54`), not parameters.
  - The memory value has a leading space (`:72`).
  - Retry is on the DAG, not on `process` (`:40-42`).
  - The label `app: s1_mosaic` is a leftover (`:23`).
  - Pods are sized 220Gi/50 CPU but run 64 threads (`:119-144`).

### Uncommitted prototype (`notebooks/`)

- **`utils.py`**
  - `load_data(bbox, year, chunks)`, `utils.py:148-250`:
    - S2 geomad: 12 bands plus 9 indices.
    - Copernicus DEM from Planetary Computer, plus slope, northness and eastness.
    - S1 geomad: `mean_vv`, `mean_vh`, `mean_vv_vh`.
    - Reads from **prod** STAC `stac.digitalearthpacific.org`.
  - `predict(model, ds)`, `utils.py:253-285`:
    - Selects features by `model.feature_names_in_`.
    - Runs blockwise `predict_proba`.
    - Returns `lulc` (int8, 0 = nodata) and `probabilities` (winning-class %).
  - Classes 1–8, `utils.py:61-70`: bare_land, forest, crops, grassland, settlements, mangroves, water, quarry.
- **Known defects**
  - The training data is read at import time from a path that doesn't exist (`training_data/mred-tdata.geojson`, `utils.py:57-59`).
  - Everything is limited to the dev `BBOX` (`utils.py:21-26`) and `DATETIME="2022"` (`:19`).
  - There's a TODO saying lon/lat bbox search breaks across the antimeridian (`:147`).
- **Training notebook** (`1-ml-learning.ipynb`):
  - Sampling: nearest-pixel feature sampling (cells 8-9).
  - Model: `RandomForestClassifier(n_estimators=500, class_weight="balanced", oob_score=True, random_state=42)` (cell 11).
  - Evaluation: OOB score only (cell 12).
  - Save: `joblib.dump(model, "test_model.dump")` (cell 13).
- **Prediction notebook** (`2-prediction.ipynb`): `.compute()` loads the whole area into memory before predicting (cell 7) and writes a local `lulc_2022.tif` (cell 13).
- **Training data** (`mred-tdata.geojson`):
  - 1,633 points, 3 with null geometry.
  - Lon 177.26–179.55, all west of 180°.
  - Class field `lulc_code`. Counts: bare_land 327, quarry 294, grassland 212, settlements 208, forest 194, crops 167, water 136, mangroves 95.
- **Dependencies:** `notebooks/pyproject.toml` requires Python ≥3.12 and scikit-learn ≥1.9,<2. No dep-tools, boto3 or CLI.

### Research findings

- **dep-tools v0.5.3 is still the latest tag.**
  - `S3ItemPath` turns `.` into `-` in the version (`dep_tools/namers.py:35`), so `0-0-2` is safe as-is.
  - It provides `grids.PACIFIC_GRID_10`, `utils.search_across_180` (`utils.py:185`), `AwsDsCogWriter` and `write_stac_s3`.
  - It doesn't constrain numpy or sklearn.
- **Data availability for Fiji on prod STAC:**
  - `dep_s2_geomad`: 2017–2025.
  - `dep_s1_geomad`: 2017–2024.
  - Staging has only 2022–2024 (S2) and 2024 (S1).
  - **Both inputs exist for 2017–2024.**
- **Tiles:** dep-grid has **54 FJ tiles**, each 96 km (9,600 px at 10 m) in EPSG:3832.
  - Longitudes are stored unwrapped (173.9–182.55).
  - 8 tiles straddle 180° (column 66). Lau is in columns 66–68.
- **Output size:** existing `proba` COGs are about 230 MB per tile.
- **S2 gap around 180° (upstream DEP data, found in T1).** `dep_s2_geomad` has no data in a band about 28 km wide around 180°, roughly x 3,330,000–3,358,000 in EPSG:3832 (about 179.92°E to 179.8°W).
  - In tiles 66,20 and 66,21 the first ~22 km east of the tile's west edge are empty. Tile 65,21's data ends ~6 km before its east edge.
  - The gap is the same in 2017, 2020, 2022, 2023 and 2024. `dep_s1_geomad` is fully valid there.
  - Item footprints are full tile bounds, so the items exist, but their S2 pixels are nodata.
  - Pixels in the gap become `class` nodata (FR8's S2 nodata rule). This covers parts of Taveuni and nearby islands.

### Sentinel-1 2025 availability

On prod STAC `stac.digitalearthpacific.org` (checked 2026-11-08, item searches over Fiji), `dep_s1_geomad` has items for 2017–2024 and **none for 2025**. `dep_s2_geomad` has 2017–2025. The collection-level temporal extent is open-ended and can't be relied on; only item searches show which years exist.

Because there is a single model that needs S1 features (D11), **2025 cannot be classified until DEP publishes S1 geomad 2025**. When it is published, run the workflow with `years=2025`; no code change is needed. Until then:
- the workflow's default range is `2017-2024`;
- a 2025 task exits 0 with a "no S1 data" log line (FR9).

## Decisions (from the user)

| # | Decision |
|---|---|
| D1 | Version is `0-0-2`, held in one variable. The model goes to `s3://dep-public-staging/dep_s2s1_mrd/models/0-0-2/`. |
| D2 | The new code replaces the committed code. |
| D3 | The workflow is renamed without the version in its filename. |
| D4 | The same functions are used for training and for whole-country classification. |
| D5 | Training year is configurable, default 2022. |
| D6 | Years to classify are a workflow parameter. |
| D7 | Training is a CLI script run by hand, not Argo. |
| D8 | Each tile writes `class` + `proba` COGs (same asset names as 0-0-1) plus a STAC item. |
| D9 | `proba` is stored as uint8, 0–100 (winning-class percent). |
| D10 | Naming uses `mrd`, not `gravel`: workflow `workflow/ml-mrd-annual.yaml`, `generateName: dep-ml-mrd-`. |
| D11 | Exactly one model (S2 + S1 + DEM features). There is no S2-only fallback model. |
| D12 | Training uses only 2022 labels and imagery for now (`mred-tdata.geojson`). |
| D13 | `auspatious/ldn-lulc` is the reference implementation for antimeridian handling and production processing. |
| D14 | All work is done on the `productionise` branch, which branches off `will-run`. |
| D15 | Tiles come from the dep grid, as in 0-0-1 (`src/run_task.py:30-37` reads dep-grid `grid_pacific.geojson`, 54 `FJ` tiles). |
| D16 | Inputs are read from prod STAC. Outputs are written to `dep-public-staging` as a test run; a later run writes to the prod bucket by changing only `--output-bucket`. |
| D17 | For T1, index and terrain logic is checked against ldn-lulc's implementation, which is more robust, and aligned with it. |

## Proposal

Replace `src/` with a small `mrd/` package. It follows `auspatious/ldn-lulc`'s layout (`ldn/` package, typer `ldn` entry point, `ldn/cli.py:59-63`) but is cut down to what this project needs:

| File | Holds |
|---|---|
| `mrd/__init__.py` | `VERSION = "0-0-2"`, bucket, catalog URLs, collection names, class map |
| `mrd/data.py` | Code shared by training and prediction: `load_data`, `add_indices`, terrain features, `sample_points`, `predict`, `load_model` |
| `mrd/train.py` | Builds the training table, runs CV, fits the model, uploads it |
| `mrd/classify.py` | Per-tile task: dep-tools writers and STAC item |
| `mrd/cli.py` | Typer app `mrd` with subcommands `train`, `print-tasks` and `run` |

Training and classification both get their features only from `mrd.data.load_data` (D4). This is ldn-lulc's pattern: one `make_features` is "used for both training data and prediction" (`ldn/raster.py:455-478`).

## Reference implementation: ldn-lulc

This plan follows <https://github.com/auspatious/ldn-lulc> (D13). Citations below are into that repo.

| Concern | ldn-lulc does | This plan |
|---|---|---|
| Tile grid | `dep_tools.grids.grid(..., crs=PACIFIC_EPSG, return_type="GridSpec")`, `grid.tile_geobox((x, y))` (`grids.py:99-101`, `lulc.py:564-565`) | Same, at 10 m |
| On-grid inputs (GeoMAD) | Loads its own EPSG:3832 GeoMAD by item ID, tile plus neighbours, onto the padded geobox. No reprojection, so tiles that cross 180° are safe (`lulc.py:100-134, 519-520`). | DEP `dep_s2_geomad` and `dep_s1_geomad` are on the same dep grid. Search with `search_across_180` on the **unpadded** geobox, keep only items that overlap it (not neighbours that just touch an edge), and load with `odc.stac.load(geobox=geobox)`. No pad: no S2 or S1 feature uses neighbouring pixels. |
| WGS84 inputs (DEM) | `load_across_antimeridian`: east and west halves are loaded separately, joined in `+pm=180`, then reprojected to the geobox. A plain `stac_load(geobox=)` left about **24% of Fiji tile 066_020 empty** (`raster.py:214-256`). | Port `load_across_antimeridian` for `cop-dem-glo-30` |
| Padding | `FOCAL_PAD` is half the largest focal window (25 px). Features are computed on the padded data, then `crop_to_geobox` (`raster.py:203-205, 438-452`). | DEM only, about 100 m (at least 3 GLO-30 pixels), searched and loaded on the padded area so terrain gradients at tile edges use real neighbours. Terrain is computed on the native DEM grid, then reprojected onto the unpadded geobox, which is the crop. No focal statistics. |
| Feature check | `do_prediction` fails if the dataset's features differ from `model.feature_names_in_` (`lulc.py:209-213`) | Same (FR4) |
| Model storage | One `.joblib` under `models/<v>/`, uploaded with boto3, downloaded over **HTTPS** with `requests` and cached (`lulc.py:418-454`). No metadata file. | Same download path, plus `metadata.json` (an addition) |
| Version | Several constants (`utils.py:93-100`) | One `VERSION` (D1) |
| Writing | dep-tools: `S3ItemPath`, `AwsDsCogWriter(write_multithreaded=True)`, `AwsStacWriter`, `StacCreator`, with an antimeridian-safe STAC bbox and geometry (`raster.py:483-590, 728-760`) | Same, including the antimeridian-safe bbox and geometry |
| Nodata | 255 for classification and probabilities. Probability is `round(p*100)` uint8 (`lulc.py:178-179, 395-398`). | Same encoding. A single winning-class `proba` band (D9) instead of one band per class. |
| Skip existing | Checked both in `print-tasks` (lists STAC items) and in the task (`object_exists`) (`cli.py:81-176`, `raster.py:532-535`) | Same |
| Errors | Each run logs and re-raises, so the pod exits 1 (`lulc.py:673-675`). There is a skip exit code for empty inputs (`cli_geomad.py:39-40`). | No data: exit 0 with a warning. Anything else: exit non-zero (FR9). |
| Dask and logging | `configure_s3_access_profile()`, `with DaskClient(...)`, 1024 chunks. One logging config set in the CLI; modules use `getLogger(__name__)` (`aws.py:27-29`, `lulc.py:637-642`, `cli.py:47-56`). | Same |
| Training | Fitting happens in a notebook. OOB score plus held-out test tiles, no CV (cells 16, 19). | `mrd train` CLI (D7) with spatial grouped CV |
| Dependencies and CI | uv, pinned `scikit-learn>=1.8,<1.9` with a comment that model and classification must be upgraded together, ruff (E, F, I, line length 120), Docker smoke test `ldn --help` (`pyproject.toml:37, 64-71`, `Dockerfile:1-30`) | Same tools. Exact sklearn pin; ruff replaces black. |
| Argo | `templates/ls-lulc-classify-annual.yaml`: `print-tasks` feeds `withParam`, parameters for years, version, image-tag and model-path, request 9Gi/1 CPU, no retries (`:32-178`) | Same shape. Keep a small retry on `run`; it is safe because no-data exits 0. |

Not adopted from ldn-lulc:
- gap-filling from neighbouring years (`lulc.py:285-320`);
- the 30% minimum-probability nodata rule (`lulc.py:256-259`); see Open question 5;
- per-class probability bands.

## Scope

**In scope**

- The `mrd/` package and the `mrd` CLI.
- Training and the S3 model artifact.
- Per-tile classification and print-tasks.
- Merged `pyproject.toml` and `uv.lock`, the Dockerfile, and CI switched to ruff.
- The renamed, parameterised workflow.
- Deleting outdated code: `src/`, `models/*.dump`, `notebooks/Load_ML_Mosaic.ipynb`, `notebooks/Test_ML_Mosaic.ipynb`, `notebooks/utils.py`, and the notebooks' own pyproject, Makefile, README and LICENSE.
- Prototype notebooks reduced to thin callers of `mrd`.

**Out of scope**

- Countries other than Fiji.
- New labels or other training years (D12).
- Hyperparameter tuning.
- Post-processing (smoothing, minimum mapping unit, probability threshold).
- Gap-filling.
- An S2-only model (D11).
- The prod-bucket run itself and prod STAC catalog publishing. The code supports it through `--output-bucket` (D16).
- Deleting `0-0-1` outputs.
- Training in Argo.

## Functional requirements

- **FR1: version constant.** `mrd.VERSION = "0-0-2"` is defined once. Every CLI uses it as the default `--version`. The model location is derived as `s3://dep-public-staging/dep_s2s1_mrd/models/{VERSION}/`, served over HTTPS as `https://dep-public-staging.s3.us-west-2.amazonaws.com/dep_s2s1_mrd/models/{VERSION}/` (region to confirm when implementing).
- **FR2: tile-based loading.** `load_data(geobox: GeoBox, year: int, chunks: dict | None = None) -> xr.Dataset` returns features on exactly `geobox`.
  - S2 and S1 geomad: search prod STAC with `dep_tools.utils.search_across_180(geobox, client=DEP client)`, keep items that overlap `geobox` (not those only touching an edge), then `odc.stac.load(geobox=geobox)`. No pad.
  - DEM: a ported `load_across_antimeridian` (ldn `raster.py:214-256`) on Planetary Computer `cop-dem-glo-30`, searched and loaded on a padded area. A side or tile with no DEM items is NaN, not an error.
  - Terrain (`elevation`, `slope`, `northness`, `eastness`) is computed on the native DEM grid, then reprojected onto `geobox`.
  - The lon/lat bbox path, the module-level `BBOX`, and the import-time training-data read are removed.
- **FR3: no edge artefacts.** Only the DEM is padded: about 100 m, at least 3 GLO-30 pixels, enough for the terrain gradient. There are no focal statistics, and S2/S1 features are per pixel.
- **FR4: one feature list.** Features are `load_data`'s data variables. `predict` selects by `model.feature_names_in_` and raises a clear error naming any missing or extra feature.
- **FR5: training.** `mrd train [--year 2022] [--version] [--training-data data/mred-tdata.geojson]`:
  - drops null-geometry points and reprojects to EPSG:3832;
  - groups points by tile; for each tile, runs `load_data(tile_geobox, year)` then `sample_points` (vectorised nearest-pixel `sel`);
  - reports `StratifiedGroupKFold` CV (5 folds, groups = tile) and random stratified 5-fold CV;
  - fits `RandomForestClassifier(n_estimators=500, class_weight="balanced", oob_score=True, random_state=42)` on all points, as a DataFrame so feature names are kept;
  - uploads `model.joblib` and `metadata.json` with boto3.
- **FR6: `metadata.json` contents.**
  - sklearn and python versions;
  - `feature_names` and the class map;
  - training year (2022) and a hash of the training data;
  - hyperparameters;
  - OOB score;
  - CV overall accuracy, macro-F1, per-class P/R/F1 with support, and the confusion matrix;
  - git SHA and creation time.
- **FR7: task list.** `mrd print-tasks --years 2017-2024 --regions FJ [--overwrite/--no-overwrite]` emits `[{"tile-id","year","version"}]`.
  - Invalid `--years` raises an error.
  - Without `--overwrite`, tiles that already have a STAC item are dropped.
- **FR8: per-tile classification.** `mrd run --tile-id X --year Y [--version] [--model-path] [--output-bucket dep-public-staging]`:
  - downloads and caches the model over HTTPS, and asserts its sklearn version matches the runtime;
  - skips the tile if its STAC item exists, unless `--overwrite`;
  - runs `load_data` then `predict`;
  - writes, via dep-tools to `dep_s2s1_mrd/<version>/<tile>/<year>/`:
    - `class`: uint8 class 1–8, nodata 255;
    - `proba`: uint8 winning-class percent 0–100, nodata 255;
    - a STAC item with an antimeridian-safe bbox and geometry, the model URI as a `related` link, and the image tag.
- **FR9: failure behaviour.**
  - If no S2 or S1 items exist for the tile and year (ocean-only, or S1 2025), log a "no data" warning naming the missing collection and exit 0 without writing.
  - Any other exception is logged and re-raised, so the pod exits non-zero.
- **FR10: workflow.** `workflow/ml-mrd-annual.yaml`, `generateName: dep-ml-mrd-`, label `app: ml-mrd`.
  - Workflow parameters: `version` (default `0-0-2`), `years` (default `2017-2024`), `regions` (`FJ`), `image-tag`, `overwrite`.
  - The model path is derived from `version`.
  - Steps call `mrd print-tasks` and `mrd run`.
  - A small `retryStrategy` sits on the `run` template.
  - The leading-space bug is fixed.
- **FR11: one environment.**
  - A single root `pyproject.toml` and `uv.lock`, with an `mrd` script entry point.
  - Drop `dask-ml` (T1) and `dea-tools` (T3, with `src/`, which imports it).
  - Pin scikit-learn exactly, with ldn's "upgrade model and classification together" comment.
  - The Dockerfile smoke-tests `mrd --help`.

## Non-functional requirements

- **NFR1: reproducibility.** `mrd train` is deterministic: `random_state=42`, sorted tile order.
- **NFR2: memory.** Prediction streams blockwise through dask and never computes the full tile stack.
- **NFR3: pod sizing.** Set from a measured single-tile run (peak RSS ×1.3, threads equal to CPU). For reference, ldn-lulc runs 9–12Gi and 1–2 CPU per tile at 30 m (`ls-lulc-classify-annual.yaml:136-178`). Our tiles are 10 m, about 9× the pixels.
- **NFR4: lint.** ruff (E, F, I, line length 120) over `mrd/` replaces `black --check src` in CI.

## Acceptance criteria

- **AC1:** `models/0-0-2/model.joblib` and `metadata.json` exist in `dep-public-staging`. The metadata covers every training point that has features (about 1,630) and includes the spatial-CV metrics.
- **AC2:** In the Docker image, `mrd.data.load_model(...)` gives no sklearn version warning, and `feature_names_in_` equals `list(load_data(...).data_vars)` for any FJ tile.
- **AC3:** `mrd run --tile-id 66,20 --year 2022` (tile crosses 180°) writes `class` and `proba`. The DEM-derived features have no empty stripe east or west of 180° (ldn-lulc's failure mode was about 24% of this tile empty). Output pixels are nodata (255) only where S2 reflectance is NaN, which includes the upstream DEP S2 gap around 180°.
- **AC4:** The shared edge of tiles `65,20` and `66,20` shows no seam in slope.
- **AC5:** `mrd print-tasks --years 2017-2024 --regions FJ --overwrite` returns 54 × 8 = 432 tasks.
- **AC6:** The workflow with `years=2017-2024` completes. No-data tiles exit 0 with a warning and don't count as failed.
- **AC7:** `rg -n "0-0-1|0\.0\.1|gravel|test_model|dea_tools|dep_s1_mosaic"` over `mrd/` and `workflow/` returns nothing. `0-0-2` appears only in `mrd/__init__.py` and the workflow default.
- **AC8:** `mrd/train.py` and `mrd/classify.py` get features only through `mrd.data.load_data` (checked by grep).

## Success measures

- All 54 FJ tiles × 2017–2024 are written under `0-0-2`.
- Spatial-CV macro-F1 is reported alongside OOB.
- The team accepts a visual QA of known quarry and mangrove sites.

## Risks

| Risk | Mitigation |
|---|---|
| sklearn pickle mismatch | One lock file and a version assert at load time (FR8, FR11) |
| Empty DEM regions across 180° | Port `load_across_antimeridian` (FR2); AC3 |
| Missing DEM tiles in the pad at tile edges | The DEM is searched and loaded on the padded area (FR2, FR3); AC4 |
| DEP S2 geomad gap around 180° makes those pixels `class` nodata | Upstream DEP data issue; recorded in Research findings, reported to DEP (Open question 4) |
| A 2022-trained model drifts on 2017–2024 | Accepted for v0-0-2 (D12); training year recorded in metadata |
| Spatial CV comes out much lower than OOB | Report both; don't block |
| No S1 geomad for 2025 | Default years 2017–2024; FR9; rerun with `years=2025` once it is published |

## Implementation roadmap

All tasks are done on the `productionise` branch, which branches off `will-run`. PRs target `will-run`, not `main`.

### T1: mrd package, shared loader, single environment
**Status:** Done
**Depends on:** None
**In scope:**
- `mrd/__init__.py` with `VERSION = "0-0-2"`, constants and the class map.
- `mrd/data.py`, ported from `notebooks/utils.py`, containing:
  - `load_data(geobox, year, chunks)` per FR2 and FR3, including a port of ldn's `load_across_antimeridian` for the DEM;
  - `add_indices` and the terrain features;
  - `sample_points`, `predict` (FR4) and `load_model` (HTTPS download, cache, sklearn assert).
- Check `add_indices` and the terrain features against ldn-lulc, which is the more robust implementation (D17). Port ldn's version wherever it handles something better, including:
  - reflectance scaling and clipping, nodata and NaN/inf handling, and safe division;
  - index formulas, especially the prototype's `mndwi` using swir22 (`notebooks/utils.py:115`) and its "ndbi", which is really a red-edge index (`:136-137`);
  - slope/aspect method and units, northness/eastness, and the focal-window statistics;
  - focal padding and crop (`ldn/raster.py:203-205, 438-478`).
  Record each kept or changed feature, with its ldn-lulc citation, in a docstring or a short table in `mrd/data.py`.
- Merge dependencies into the root `pyproject.toml` (add an `mrd` script, drop `dask-ml`; keep `dea-tools` until T3, pin scikit-learn exactly) and regenerate `uv.lock`.
- Switch the CI lint to ruff.
- Delete `notebooks/utils.py` and the notebooks' pyproject, Makefile, README and LICENSE.
**Out of scope:**
- The CLI subcommands.
- Training a model.
- The workflow.
**Surfaces touched:** `mrd/__init__.py`, `mrd/data.py`, `mrd/cli.py`, `tests/test_data.py`, `pyproject.toml`, `uv.lock`, `.github/workflows/build-test-docker.yml`, `notebooks/utils.py`, `notebooks/pyproject.toml`, `notebooks/Makefile`, `notebooks/README.md`, `notebooks/LICENSE`, `docs/prd/fiji-landcover-0-0-2.md`
**Interfaces:**
- `load_data(geobox: odc.geo.GeoBox, year: int, chunks: dict | None = None) -> xr.Dataset`
- `sample_points(ds: xr.Dataset, points: gpd.GeoDataFrame) -> pd.DataFrame`
- `predict(model, ds: xr.Dataset) -> xr.Dataset` (`class`, `proba`, uint8, nodata 255)
- `load_model(url: str) -> RandomForestClassifier`
- `VERSION: str = "0-0-2"`
**Verification:**
- `uv sync` succeeds.
- An assert-based check loads a small window of tile `66,21` across 180° for 2022 and finds all S2, S1 and DEM variables present. The DEM and S1 features have no all-NaN columns either side of 180°, and S2 matches the source item (empty there: the upstream S2 gap). A second window (tile 65,21, over land) checks S2 and the indices on real data.
- `ruff check mrd` passes.
- Every index and terrain feature in `mrd/data.py` cites the ldn-lulc function it was checked against, or gives a reason it differs.
- An assert-based check on a small synthetic array covers: nodata gives NaN (not inf), a zero denominator gives NaN, indices fall in [-1, 1], and slope is 0 on flat terrain.
**Next command:** `/eg-new-feature implement T1 from docs/prd/fiji-landcover-0-0-2.md`

### T2: Training and production model upload
**Status:** Not started
**Depends on:** T1
**In scope:**
- `mrd/train.py`, plus `mrd train` in `mrd/cli.py` (FR5, FR6).
- Move `notebooks/mred-tdata.geojson` to `data/mred-tdata.geojson`.
- Train with 2022 imagery and upload `model.joblib` and `metadata.json` to `models/0-0-2/`.
- Make `notebooks/1-ml-learning.ipynb` a thin caller.
**Out of scope:**
- Hyperparameter search.
- Other training years.
- New labels.
**Surfaces touched:** `mrd/train.py`, `mrd/cli.py`, `mrd/__init__.py`, `tests/test_train.py`, `pyproject.toml`, `uv.lock`, `data/mred-tdata.geojson`, `notebooks/1-ml-learning.ipynb`
**Interfaces:** `mrd train --year 2022 --version 0-0-2` writes `s3://dep-public-staging/dep_s2s1_mrd/models/0-0-2/{model.joblib,metadata.json}`.
**Verification:**
- AC1 and AC2.
- A rerun gives an identical CV report (NFR1).
**Next command:** `/eg-new-feature implement T2 from docs/prd/fiji-landcover-0-0-2.md`

### T3: Per-tile classification and task listing
**Status:** Not started
**Depends on:** T1
**In scope:**
- `mrd/classify.py`, plus `mrd run` and `mrd print-tasks` in `mrd/cli.py` (FR7, FR8, FR9).
- dep-tools writers with the antimeridian-safe STAC bbox and geometry (ldn `raster.py:554-590`).
- Logging and dask setup as in ldn.
- Point the Dockerfile at `mrd --help`.
- Delete `src/`, `models/`, `notebooks/Load_ML_Mosaic.ipynb` and `notebooks/Test_ML_Mosaic.ipynb`, and drop `dea-tools` from `pyproject.toml` (and `src` from setuptools `packages`).
- Make `notebooks/2-prediction.ipynb` a thin caller.
**Out of scope:**
- Workflow YAML.
- Post-processing.
- Gap-filling.
**Surfaces touched:** `mrd/classify.py`, `mrd/cli.py`, `mrd/aws.py`, `mrd/train.py`, `mrd/__init__.py`, `tests/test_classify.py`, `tests/test_train.py`, `pyproject.toml`, `uv.lock`, `Dockerfile`, `.dockerignore`, `src/`, `models/`, `notebooks/Load_ML_Mosaic.ipynb`, `notebooks/Test_ML_Mosaic.ipynb`, `notebooks/2-prediction.ipynb`, `docs/prd/fiji-landcover-0-0-2.md`
**Interfaces:**
- `mrd run --tile-id 66,20 --year 2022 [--version 0-0-2] [--model-path <url>] [--overwrite]` writes `dep_s2s1_mrd/0-0-2/066/020/2022/dep_s2s1_mrd_066_020_2022_{class,proba}.tif` plus `.stac-item.json`.
- `mrd print-tasks --years 2017-2024 --regions FJ` prints a JSON list.
**Verification:**
- AC3, AC4, AC5, AC7 and AC8.
- `mrd run` for a 2025 tile exits 0 with a "no S1 data" warning.
**Next command:** `/eg-new-feature implement T3 from docs/prd/fiji-landcover-0-0-2.md`

### T4: Renamed, parameterised Argo workflow and docs
**Status:** Not started
**Depends on:** T2, T3
**In scope:**
- `git mv workflow/ml-gravel-annual-0-0-1.yaml workflow/ml-mrd-annual.yaml`, then apply FR10.
- Set resources from a measured single-tile run (NFR3). Initial estimate in the workflow: 32Gi/8 CPU (unmeasured; re-measure).
- Rewrite the README:
  - what MRD stands for;
  - how to train;
  - how to submit;
  - S1 2025 availability;
  - the link to ldn-lulc.
**Out of scope:**
- A training workflow.
- Cron scheduling.
**Surfaces touched:** `workflow/ml-mrd-annual.yaml`, `README.md`, `Makefile`
**Interfaces:** `argo submit workflow/ml-mrd-annual.yaml -p years=2017-2024 -p image-tag=<tag>` fans out `mrd print-tasks` into `mrd run` for each tile and year.
**Verification:**
- `argo lint` passes.
- A `years=2022` run limited to a few tiles completes, then the full run meets AC6.
**Next command:** `/eg-new-feature implement T4 from docs/prd/fiji-landcover-0-0-2.md`

## Open questions

1. **Training-data swap:** should `MRD_dissagregated_32.geojson` (TODO at `notebooks/utils.py:57`) replace `mred-tdata.geojson` in a later version?
2. **Minimum probability:** adopt ldn's rule that `class` is nodata where `proba` < 30%? Not in scope for 0-0-2 unless decided.
4. **DEP S2 gap around 180°:** ask DEP to fix `dep_s2_geomad` coverage around the antimeridian (see Research findings).

Resolved:
- `proba` is uint8 (D9);
- naming is `mrd` (D10);
- one model, with 2025 skipped until S1 exists (D11);
- 2022 training only (D12);
- asset names `class`/`proba` (D8);
- dep grid, as in 0-0-1 (D15);
- read from prod, write to staging as a test, then prod (D16).
- MRD = Mineral Resource Detection (confirmed by the user).

## Sources

- **Repo code:** `src/run_task.py`, `src/print_tasks.py`, `workflow/ml-gravel-annual-0-0-1.yaml`, `Dockerfile`, `pyproject.toml`, `.github/workflows/build-test-docker.yml`, `notebooks/utils.py`, `notebooks/1-ml-learning.ipynb`, `notebooks/2-prediction.ipynb`, `notebooks/mred-tdata.geojson`.
- **Reference implementation:** <https://github.com/auspatious/ldn-lulc>. Files: `ldn/raster.py`, `ldn/lulc.py`, `ldn/cli.py`, `ldn/utils.py`, `ldn/grids.py`, `ldn/training_data.py`, `templates/ls-lulc-classify-annual.yaml`, `pyproject.toml`, `Dockerfile`, `notebooks/1_Train_Model.ipynb`.
- **dep-tools v0.5.3:** <https://github.com/digitalearthpacific/dep-tools/tree/v0.5.3>. Files: `dep_tools/namers.py`, `grids.py`, `utils.py`, `writers.py`, `aws.py`.
- **dep-grid:** <https://github.com/digitalearthpacific/dep-grid>, `grid_pacific.geojson`.
- **STAC APIs:** <https://stac.digitalearthpacific.org>, <https://stac.staging.digitalearthpacific.io> (item searches over Fiji, 2026-11-08).
- **Existing outputs:** `s3://dep-public-staging/dep_s2s1_mrd/0-0-1/`.
- **Library guidance from model knowledge:** the `StratifiedGroupKFold` and joblib guidance came from model knowledge, not a fetched source.
