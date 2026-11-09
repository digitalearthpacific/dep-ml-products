# linux/amd64 only: fiona (from dep-tools) has no ARM Linux wheel. GDAL comes bundled in the rasterio and fiona wheels.
FROM --platform=linux/amd64 python:3.13-slim AS base

# git: uv fetches dep-tools from GitHub.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /usr/local/bin/
ENV UV_PROJECT_ENVIRONMENT=/code/.venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1
# The CA bundle for HTTPS reads by the GDAL bundled in the rasterio wheel.
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
WORKDIR /code

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev
ENV PATH="/code/.venv/bin:$PATH"

FROM base AS final
RUN mrd --help
RUN python -c "import mrd.classify, mrd.train, rasterio, fiona, pyogrio, geopandas, sklearn"
