# Digital Earth Pacific ML Products

## Tile-based workflows

This is a work in progress.

TODO: Add info here.


This is a land cover classification.

Mineral Resource Detection in Fiji. Output is here: https://stac-browser.staging.digitalearthpacific.io/collections/dep_s2s1_mrd
2017-2023 already exist here but outputs don't look good.

stac.prod.digitalearthpacific.io currently doesn't have this collection.



## Models

What type of model is used?

Where were they made? What features do they need?

## Quickstart

To run locally, install:

```bash
brew upgrade gdal
brew install uv
uv sync
```
Now you can run notebooks, make commands, or src directly.

The make commands are dockerised so no need to have installed locally:
```bash
uv run make build
uv run make run
```


