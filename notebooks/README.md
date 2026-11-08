# minerals-model

Preliminary attempts on updated minerals workflow output
<img width="422" height="333" alt="Predicted LULC map (EPSG:3832)" src="https://github.com/user-attachments/assets/eadfc49d-feca-4b41-8984-78d52d1124d0" />

## Quickstart

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync              # create .venv and install deps from uv.lock
source .venv/bin/activate
uv run prek install  # enable pre-commit hooks (ruff lint + format)
uv run jupyter lab   # then open 1-ml-learning.ipynb, then 2-prediction.ipynb
```

Add a dependency with `uv add <package>` and commit both `pyproject.toml` and `uv.lock`.

Commits run [ruff](https://docs.astral.sh/ruff/) lint and format via [prek](https://github.com/j178/prek) (`.pre-commit-config.yaml`). Or use `make lint`, `make format`, `make check` (all hooks on all files).
