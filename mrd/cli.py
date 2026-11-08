import logging
from pathlib import Path

import typer

from mrd import VERSION

app = typer.Typer()


@app.callback()
def main() -> None:
    """MRD land-cover classification for Fiji."""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("mrd").setLevel(logging.INFO)


@app.command()
def train(
    year: int = typer.Option(2022, help="Year of imagery to sample at the training points."),
    version: str = typer.Option(VERSION, help="Model version, the S3 folder under the model prefix."),
    training_data: Path = typer.Option(None, help="Training points (GeoJSON). Default: data/mred-tdata.geojson."),
    upload: bool = typer.Option(True, help="Upload model.joblib and metadata.json to S3."),
    overwrite: bool = typer.Option(False, help="Replace an already published model version."),
    allow_dirty: bool = typer.Option(False, help="Upload even if the model code or training data are uncommitted."),
) -> None:
    """Train the model on all training points and publish it."""
    # Imported here so `mrd --help` doesn't load the training stack.
    from mrd.train import TRAINING_DATA
    from mrd.train import train as run

    run(year, version, training_data or TRAINING_DATA, upload, overwrite, allow_dirty)
