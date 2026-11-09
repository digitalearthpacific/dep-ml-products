IMAGE = dep-ml-products

build:
	docker build --platform linux/amd64 --tag $(IMAGE) .

run:
	docker run --platform linux/amd64 --rm \
		-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN \
		$(IMAGE) \
		mrd run --tile-id 64,20 --year 2022 --overwrite

train:
	uv run mrd train --year 2022

test:
	uv run pytest tests -q

lint:
	uv run ruff check mrd tests
	uv run ruff format --check mrd tests
