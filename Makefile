.PHONY: install test lint run image deploy clean

IMAGE ?= quay.io/example/gpu-usage-calculator:latest

install:
	python -m pip install -e '.[dev]'

test:
	pytest -q

lint:
	ruff check src tests

run:
	uvicorn gpu_usage_calculator.app:app --reload --port 8080

image:
	podman build -t $(IMAGE) -f Containerfile .

deploy:
	kubectl apply -k deploy/kubernetes/base

clean:
	rm -rf .pytest_cache .ruff_cache dist build

