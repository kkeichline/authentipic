# Makefile for AuthentiPic

.PHONY: setup test lint format run-train run-predict clean

# Setup the project
setup:
	poetry install

# Run tests
test:
	poetry run pytest

# Run linter
lint:
	poetry run ruff check src/authentipic

# Format code
format:
	poetry run black src/authentipic

# Run training pipeline
run-train:
	poetry run python src/authentipic/main.py --mode train

# Run prediction on an image
run-predict:
	poetry run python src/authentipic/main.py --mode predict --image_path $(IMAGE_PATH)

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf dist
	rm -rf build

# Run all quality checks
quality-check: lint test

# update packages with poetry
update:
	poetry update

# Run annotation web app
run-annotation-app:
    poetry run python src/authentipic/annotation/app.py

# Export annotations to CSV
export-annotations:
    poetry run python -m authentipic.annotation.manager --export

# Generate training dataset from annotations
generate-annotation-dataset:
    poetry run python -m authentipic.annotation.manager --generate-dataset data/processed/annotation_dataset.csv

# Run the API server
run-api:
    poetry run python src/authentipic/run_api.py

# Run the API server in development mode with auto-reload
run-api-dev:
    poetry run python src/authentipic/run_api.py --reload

# Convert model for browser/app
convert-tfjs:
    poetry run python scripts/convert_model.py --format tfjs --output-dir ./public/models

# Convert model to ONNX
convert-onnx:
    poetry run python scripts/convert_model.py --format onnx --output-dir ./exported_models