# Makefile for AuthentiPic

.PHONY: setup test lint format run-train run-predict clean export-annotations generate-annotation-dataset run-annotation-app run-api run-api-dev convert-tfjs convert-onnx

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
annotations-export:
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

# Convert model to all formats
convert-model-all:
	poetry run python scripts/convert_model.py --format all --output-dir ./models/converted

# Convert model to TensorFlow.js format for web deployment
convert-tfjs:
	poetry run python scripts/convert_model.py --format tfjs --output-dir ./public/models

# Convert model to ONNX format
convert-onnx:
	poetry run python scripts/convert_model.py --format onnx --output-dir ./models/converted

# Convert model to TensorFlow Lite format for Android/mobile
convert-tflite:
	poetry run python scripts/convert_model.py --format tflite --output-dir ./models/converted

# Convert model to CoreML format for iOS
convert-coreml:
	poetry run python scripts/convert_model.py --format coreml --output-dir ./models/converted

# Convert model with custom options
# Usage: make convert-custom CHECKPOINT=path/to/model.pth FORMAT=tfjs OUTPUT_DIR=custom/output/dir
convert-custom:
	poetry run python scripts/convert_model.py --checkpoint $(CHECKPOINT) --format $(FORMAT) --output-dir $(OUTPUT_DIR)

# Convert model to ONNX only (useful for debugging conversion issues)
convert-onnx-only:
	poetry run python scripts/convert_model.py --format onnx --output-dir ./models/debug

# Test ONNX model validity
test-onnx:
	poetry run python -c "import onnx; model = onnx.load('./models/debug/model.onnx'); onnx.checker.check_model(model); print('ONNX model is valid')"

# Install required conversion dependencies
install-conversion-deps:
	poetry add tf2onnx tensorflow==2.12.0 tensorflowjs==4.10.0 onnx==1.14.0