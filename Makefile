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