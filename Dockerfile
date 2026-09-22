FROM ghcr.io/osgeo/gdal:ubuntu-small-3.13.2 AS base

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_PROJECT_ENVIRONMENT=/code/.venv UV_LINK_MODE=copy
WORKDIR /code

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev
 
COPY . .
RUN uv sync --frozen --no-dev
ENV PATH="/code/.venv/bin:$PATH"
 
FROM base AS final
RUN python src/print_tasks.py --help
