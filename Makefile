.DEFAULT_GOAL := build
IMAGE = dep-ml-products
# The training data isn't in this repo (see README). Set GITHUB_TOKEN if the source repo is private.
TRAINING_DATA_URL = https://raw.githubusercontent.com/nicholasmetherall/minerals-model/refs/heads/main/training_data/mred-tdata.geojson

training-data:
	mkdir -p data
	curl -fsSL $(if $(GITHUB_TOKEN),-H "Authorization: Bearer $(GITHUB_TOKEN)") -o data/mred-tdata.geojson $(TRAINING_DATA_URL)

build:
	docker build --platform linux/amd64 --tag $(IMAGE) .

run:
	docker run --platform linux/amd64 --rm \
		-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN \
		$(IMAGE) \
		mrd run --tile-id 64,20 --year 2022 --overwrite

train: data/mred-tdata.geojson
	uv run mrd train --year 2022

data/mred-tdata.geojson:
	$(MAKE) training-data

test:
	uv run pytest tests -q

lint:
	uv run ruff check mrd tests
	uv run ruff format --check mrd tests
