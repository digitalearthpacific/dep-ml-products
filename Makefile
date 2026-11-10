.DEFAULT_GOAL := build
IMAGE = dep-ml-products
# The training data isn't in this repo (see README). Set GITHUB_TOKEN if the source repo is private.
TRAINING_DATA_URL = https://raw.githubusercontent.com/nicholasmetherall/minerals-model/refs/heads/main/training_data/mred-tdata.geojson

training-data:
	mkdir -p data
	curl -fsSL $(if $(GITHUB_TOKEN),-H "Authorization: Bearer $(GITHUB_TOKEN)") -o data/mred-tdata.geojson $(TRAINING_DATA_URL)

build:
	docker build --platform linux/amd64 --tag $(IMAGE) .

print-tasks:
	uv run mrd print-tasks --years 2022

run:
	docker run --platform linux/amd64 --rm \
		-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN \
		$(IMAGE) \
		mrd run --tile-id 64,20 --year 2022 --overwrite

test-run:
	uv run mrd run --tile-id 64,20 --year 2022 --no-overwrite --model-path "data/models/0-0-2/model.joblib" --image-tag v0-0-2

# Classify every task locally, one tile at a time (each uses all cores and ~20 GB), skipping finished tiles.
# make run-all YEARS=2017-2024
YEARS ?= 2022
MODEL ?= data/models/0-0-2/model.joblib
run-all:
	uv run mrd print-tasks --years $(YEARS) | jq -r '.[] | "\(.["tile-id"]) \(.year)"' | \
	while read tile year; do \
		uv run mrd run --tile-id $$tile --year $$year --model-path $(MODEL) --image-tag v0-0-2 < /dev/null \
			|| echo "FAILED $$tile $$year" >> run-all-failures.txt; \
	done; \
	if [ -f run-all-failures.txt ]; then cat run-all-failures.txt; fi

# Prerequisites the training data
train: data/mred-tdata.geojson
	uv run mrd train --year 2022

test:
	uv run pytest tests -q

lint:
	uv run ruff check mrd tests
	uv run ruff format --check mrd tests
