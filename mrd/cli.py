import typer

app = typer.Typer()


@app.callback()
def main() -> None:
    """MRD land-cover classification for Fiji."""
