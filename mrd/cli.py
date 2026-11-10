import logging
import os
from pathlib import Path

import typer

from mrd import OUTPUT_BUCKET, VERSION

app = typer.Typer()


@app.callback()
def main() -> None:
    """MRD (Mineral Resource Detection) land-cover classification for Fiji."""
    # Logs go to stderr, so print-tasks' stdout is only its JSON.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("mrd").setLevel(logging.INFO)


@app.command()
def train(
    year: int = typer.Option(2022, help="Year of imagery to sample at the training points."),
    version: str = typer.Option(VERSION, help="Model version, the S3 folder under the model prefix."),
    training_data: Path = typer.Option(None, help="Training points (GeoJSON). Default: data/mred-tdata.geojson."),
    upload: bool = typer.Option(True, help="Upload model.joblib and metrics.json to S3."),
    overwrite: bool = typer.Option(False, help="Replace an already published model version."),
) -> None:
    """Train the model on all training points and publish it."""
    # Imported here so `mrd --help` doesn't load the training stack.
    from mrd.train import TRAINING_DATA
    from mrd.train import train as run

    run(year, version, training_data or TRAINING_DATA, upload, overwrite)


@app.command("print-tasks")
def print_tasks(
    years: str = typer.Option(..., help='"2022", "2017-2024" or "2017,2020".'),
    version: str = typer.Option(VERSION, help="Output version."),
    output_bucket: str = typer.Option(OUTPUT_BUCKET),
    overwrite: bool = typer.Option(False, help="Include tasks whose output already exists."),
    limit: int = typer.Option(None, help="Only the first N tasks, e.g. for a test run."),
) -> None:
    """Print the tile and year tasks as JSON for the Argo workflow."""
    from mrd.classify import print_tasks as tasks

    typer.echo(tasks(years, version, output_bucket, overwrite, limit))


@app.command()
def run(
    tile_id: str = typer.Option(..., help='dep grid tile id, "col,row", e.g. "66,20".'),
    year: int = typer.Option(...),
    version: str = typer.Option(VERSION, help="Output version; the model is the one published for it."),
    model_path: str = typer.Option(None, help="Model URL or local path. Default: the model published for --version."),
    output_bucket: str = typer.Option(OUTPUT_BUCKET),
    overwrite: bool = typer.Option(False, help="Redo the tile even if its output exists."),
    image_tag: str = typer.Option(os.environ.get("IMAGE_TAG", "unknown"), help="Recorded in the STAC item."),
    threads: int = typer.Option(None, help="Dask threads. Default: all CPUs."),
) -> None:
    """Classify one tile and year, and write class and proba COGs and a STAC item."""
    from mrd.classify import run as classify

    try:
        classify(tile_id, year, version, model_path, output_bucket, overwrite, image_tag, threads)
    except Exception:
        logging.getLogger("mrd").exception(f"Failed to classify {tile_id} {year}")
        raise
