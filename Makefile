build:
	docker build --tag dep/ml .

run:
	docker run -it --rm \
		-e AZURE_STORAGE_SAS_TOKEN=${AZURE_STORAGE_SAS_TOKEN} \
	 	dep/ml \
		python src/run_task.py \
		--tile-id "64,20" \
		--year 2023 \
		--version "0.0.0b1" \
		--output-resolution 100 \
		--overwrite
