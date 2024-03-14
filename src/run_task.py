from pathlib import Path

import boto3
import geopandas as gpd
import joblib
import numpy as np
import typer
import xarray as xr
from dask.distributed import Client
from dep_tools.aws import object_exists
from dep_tools.azure import blob_exists
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.loaders import OdcLoader
from dep_tools.namers import DepItemPath
from dep_tools.processors import Processor
from dep_tools.searchers import PystacSearcher
from dep_tools.stac_utils import set_stac_properties
from dep_tools.utils import get_logger

# from dep_tools.task import SimpleLoggingAreaTask
from dep_tools.writers import AwsDsCogWriter, AzureDsWriter
from typing_extensions import Annotated
from xarray import DataArray, Dataset

from utils import predict_xr


def get_tiles() -> gpd.GeoDataFrame:
    return (
        gpd.read_file(
            "https://raw.githubusercontent.com/digitalearthpacific/dep-grid/master/grid_pacific.geojson"
        )
        .astype({"tile_id": str, "country_code": str})
        .set_index(["tile_id", "country_code"], drop=False)
    )


def get_item_path(
    base_product: str, model: str, version: str, year: int, prefix: str
) -> DepItemPath:
    return DepItemPath(
        base_product,
        model,
        version,
        year,
        zero_pad_numbers=True,
        prefix=prefix,
    )


def add_indices(data: Dataset) -> Dataset:
    # Incorporate NDVI (Normalised Difference Vegetation Index) = (NIR-red)/(NIR+red)
    data["ndvi"] = (data["B08"] - data["B04"]) / (data["B08"] + data["B04"])

    # Incorporate MNDWI (Mean Normalised Difference Water Index) = (Green – SWIR) / (Green + SWIR)
    data["mndwi"] = (data["B03"] - data["B12"]) / (data["B03"] + data["B12"])

    # Incorporate EVI (Enhanced Vegetation Index) = 2.5NIR−RED(NIR+6RED−7.5BLUE)+1
    data["evi"] = (2.5 * (data["B08"] - data["B04"])) * (
        (data["B08"] + (6 * (data["B04"]) - (7.5 * (data["B02"]))))
    ) + 1

    # Incorporate SAVI (Standard Vegetation Index) = (800nm−670nm) / (800nm+670nm+L(1+L)) # where L = 0.5
    data["savi"] = (data["B07"] - data["B04"]) / (
        data["B07"] + data["B04"] + 0.5 * (1 + 0.5)
    )

    # Incorporate BSI (Bare Soil Index) = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2)) # https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/barren_soil/
    data["bsi"] = ((data["B11"] + data["B04"]) - (data["B08"] + data["B02"])) / (
        (data["B11"] + data["B04"]) + (data["B08"] + data["B02"])
    )

    # Incorporate NDMI (Normalised Difference Moisture Index) # https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndmi/
    data["ndmi"] = ((data["B08"]) - (data["B11"])) / ((data["B08"]) + (data["B11"]))

    # Incorporate NDBI (Normalised Difference Built-up Index) (B06 - B05) / (B06 + B05); # - built up ratio of vegetation to paved surface - let BU = (ndvi - ndbi) - https://custom-scripts.sentinel-hub.com/custom-scripts/landsat-8/built_up_index/
    data["ndbi"] = ((data["B06"]) - (data["B05"])) / ((data["B06"]) + (data["B05"]))

    return data


class MLProcessor(Processor):
    def __init__(
        self,
        send_area_to_processor: bool = False,
        load_data: bool = False,
        model_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            send_area_to_processor,
        )
        self.load_data = load_data
        self.model_path = model_path

    def process(self, input_data: DataArray) -> Dataset:
        loaded_model = joblib.load(self.model_path)

        filled = input_data.fillna(-9999.0)

        predicted = predict_xr(loaded_model, filled, proba=True)

        # Convert to int
        cleaned_predictions = predicted.copy(deep=True)
        cleaned_predictions["class"].data = predicted["class"].data.astype(np.int8)
        cleaned_predictions["proba"].data = predicted["proba"].data.astype(np.float32)

        output = set_stac_properties(input_data, cleaned_predictions)

        if self.load_data:
            output = output.compute()

        return output


def main(
    tile_id: Annotated[str, typer.Option()],
    year: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    model_path: str = "models/test_model_20240312.dump",
    output_bucket: str = None,
    output_resolution: int = 10,
    memory_limit_per_worker: str = "16GB",
    n_workers: int = 1,
    threads_per_worker: int = 32,
    xy_chunk_size: int = 4096,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    base_product = "s2s1"
    tiles = get_tiles()
    area = tiles.loc[[tile_id]]

    log = get_logger(tile_id, "Gravel")
    log.info(f"Starting processing version {version} for {year}")

    itempath = get_item_path(base_product, "mrd", version, year, prefix="dep")

    stac_document = itempath.stac_path(tile_id)

    # If we don't want to overwrite, and the destination file already exists, skip it
    if not overwrite:
        already_done = False
        if output_bucket is None:
            # The Azure case
            already_done = blob_exists(stac_document)
        else:
            # The AWS case
            already_done = object_exists(output_bucket, stac_document)

        if already_done:
            log.info(f"Item already exists at {stac_document}")
            # This is an exit with success
            raise typer.Exit()

    # A searcher to find the data
    # Set up later...

    # A loader to load them
    loader = OdcLoader(
        crs=3832,
        resolution=output_resolution,
        groupby="solar_day",
        chunks=dict(time=1, x=xy_chunk_size, y=xy_chunk_size),
        fail_on_error=False,
        overwrite=overwrite,
    )

    # A processor to process them
    processor = MLProcessor(model_path=model_path)

    # And a writer to bind them
    if output_bucket is None:
        log.info("Writing with Azure writer")
        writer = AzureDsWriter(
            itempath=itempath,
            overwrite=overwrite,
            convert_to_int16=False,
            extra_attrs=dict(dep_version=version),
            write_multithreaded=True,
            load_before_write=True,
        )
    else:
        log.info("Writing with AWS writer")
        client = boto3.client("s3")
        writer = AwsDsCogWriter(
            itempath=itempath,
            overwrite=overwrite,
            convert_to_int16=False,
            extra_attrs=dict(dep_version=version),
            write_multithreaded=True,
            bucket=output_bucket,
            client=client,
        )

    with Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit_per_worker,
    ):
        try:
            # Run the task
            searcher = PystacSearcher(
                catalog="https://stac.staging.digitalearthpacific.org",
                collections=["dep_s1_mosaic", "dep_s2_geomad"],
                datetime=year,
            )

            items = searcher.search(area)

            items_by_collection = {}
            for item in items:
                items_by_collection.setdefault(item.collection_id, []).append(item)

            dem_searcher = PystacSearcher(
                catalog="https://planetarycomputer.microsoft.com/api/stac/v1",
                collections=["cop-dem-glo-30"],
            )
            items_by_collection["cop-dem-glo-30"] = dem_searcher.search(area)

            results = {k: len(v) for k, v in items_by_collection.items()}
            log.info(",".join([f"{k}:{v}" for k, v in results.items()]))

            all_data = [
                loader.load(items, area).squeeze("time")
                for items in items_by_collection.values()
            ]

            data = xr.merge(all_data, compat="override")
            data = data.rename({"data": "elevation"})
            data = data.drop_vars(["median_vv", "median_vh", "std_vv", "std_vh"])

            log.info(f"Found {len(data.time)} timesteps to load")

            output_data = processor.process(data)
            log.info(
                f"Processed data to shape {[output_data.sizes[d] for d in ['x', 'y']]}"
            )

            paths = writer.write(output_data, tile_id)
            if paths is not None:
                log.info(f"Completed writing to {paths[-1]}")
            else:
                log.warning("No paths returned from writer")

        except EmptyCollectionError:
            log.warning("No data found for this tile.")
        except Exception as e:
            log.exception(f"Failed to process {tile_id} with error: {e}")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
