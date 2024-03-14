build:
	docker build --tag dep/ml .

run:
	docker run -it --rm \
		-e PC_SDK_SUBSCRIPTION_KEY=${PC_SDK_SUBSCRIPTION_KEY} \
		-e AZURE_STORAGE_SAS_TOKEN=${AZURE_STORAGE_SAS_TOKEN} \
	 	dep/ml \
		python src/run_task.py \
		--region-code "64,20" \
		--datetime 2023 \
		--version "0.0.0b1" \
		--resolution 100 \
		--overwrite
