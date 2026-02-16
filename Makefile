.PHONY: help install install-dev test lint format typecheck clean download-wlasl upload-data preprocess-data train-baseline train-multimodal evaluate mlflow-ui

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := mouna
SRC_DIR := src/$(PROJECT_NAME)
TESTS_DIR := tests
CONFIG_DEV := configs/dev.yaml
CONFIG_PROD := configs/prod.yaml

# Default target
help:
	@echo "Mouna - Sign Language Recognition System"
	@echo ""
	@echo "Available targets:"
	@echo "  install           - Install package and dependencies"
	@echo "  install-dev       - Install with development dependencies"
	@echo "  test              - Run pytest tests"
	@echo "  test-setup        - Validate installation and imports"
	@echo "  test-e2e          - End-to-end test with synthetic data"
	@echo "  lint              - Run linting"
	@echo "  format            - Format code with black"
	@echo "  typecheck         - Run type checking with mypy"
	@echo "  clean             - Remove build artifacts and cache"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  download-wlasl    - Download WLASL dataset"
	@echo "  upload-data       - Upload data to Azure Blob Storage"
	@echo "  preprocess-data   - Preprocess videos and extract keypoints"
	@echo ""
	@echo "Training:"
	@echo "  train-baseline    - Train baseline model (keypoints + BiLSTM)"
	@echo "  train-multimodal  - Train multimodal model"
	@echo "  evaluate          - Evaluate trained model"
	@echo ""
	@echo "MLflow:"
	@echo "  mlflow-ui         - Launch MLflow UI"
	@echo ""

# Installation
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

install-all:
	$(PIP) install -e ".[all]"

# Testing
test:
	pytest $(TESTS_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing

test-fast:
	pytest $(TESTS_DIR) -v -x

test-setup:
	@echo "Validating installation and setup..."
	$(PYTHON) scripts/test_setup.py

test-e2e:
	@echo "Running end-to-end test with synthetic data..."
	$(PYTHON) scripts/test_e2e_mini.py

# Code quality
lint:
	ruff check $(SRC_DIR) $(TESTS_DIR)

format:
	black $(SRC_DIR) $(TESTS_DIR)
	ruff check --fix $(SRC_DIR) $(TESTS_DIR)

typecheck:
	mypy $(SRC_DIR)

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean
	rm -rf venv/
	rm -rf mlruns/
	rm -rf data/

# Data pipeline
download-wlasl:
	@echo "Downloading WLASL dataset..."
	$(PYTHON) -m $(PROJECT_NAME).data.ingestion \
		--action download \
		--output-dir data/raw/wlasl \
		--max-videos 100

download-wlasl-full:
	@echo "Downloading full WLASL dataset..."
	$(PYTHON) -m $(PROJECT_NAME).data.ingestion \
		--action download \
		--output-dir data/raw/wlasl

upload-data:
	@echo "Uploading data to Azure Blob Storage..."
	$(PYTHON) -m $(PROJECT_NAME).data.ingestion \
		--action upload \
		--input-dir data/raw/wlasl \
		--container sign-videos-bronze

preprocess-data:
	@echo "Preprocessing videos and extracting keypoints..."
	$(PYTHON) -m $(PROJECT_NAME).data.preprocessing \
		--config $(CONFIG_DEV) \
		--input-container sign-videos-bronze \
		--output-container sign-videos-silver

# Training
train-baseline:
	@echo "Training baseline model (keypoints + BiLSTM)..."
	$(PYTHON) -m $(PROJECT_NAME).models.baseline \
		--config $(CONFIG_DEV) \
		--data-dir data/raw/wlasl \
		--checkpoint-dir checkpoints/baseline

train-baseline-prod:
	@echo "Training baseline model (production)..."
	$(PYTHON) -m $(PROJECT_NAME).models.baseline \
		--config $(CONFIG_PROD) \
		--data-dir data/raw/wlasl \
		--checkpoint-dir checkpoints/baseline-prod

train-multimodal:
	@echo "Training multimodal model..."
	$(PYTHON) -m $(PROJECT_NAME).models.multimodal \
		--config $(CONFIG_DEV) \
		--data-dir data/raw/wlasl \
		--checkpoint-dir checkpoints/multimodal

train-multimodal-prod:
	@echo "Training multimodal model (production)..."
	$(PYTHON) -m $(PROJECT_NAME).models.multimodal \
		--config $(CONFIG_PROD) \
		--data-dir data/raw/wlasl \
		--checkpoint-dir checkpoints/multimodal-prod

# Evaluation
evaluate:
	@echo "Evaluating model..."
	$(PYTHON) -m $(PROJECT_NAME).evaluation.metrics \
		--config $(CONFIG_DEV) \
		--checkpoint checkpoints/baseline/best_model.pth \
		--data-dir data/raw/wlasl

evaluate-with-gemini:
	@echo "Evaluating model with Gemini QA layer..."
	$(PYTHON) -m $(PROJECT_NAME).evaluation.metrics \
		--config $(CONFIG_DEV) \
		--checkpoint checkpoints/baseline/best_model.pth \
		--data-dir data/raw/wlasl \
		--use-gemini

# MLflow
mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns

# Development helpers
jupyter:
	jupyter lab

notebook:
	jupyter notebook

# Setup new environment
setup:
	@echo "Setting up development environment..."
	$(PYTHON) -m venv venv
	@echo "Activate with: source venv/bin/activate"
	@echo "Then run: make install-dev"

# Quick start
quickstart: install-dev
	@echo "Quick start: downloading sample data..."
	make download-wlasl
	@echo ""
	@echo "Next steps:"
	@echo "1. Update .env with your Azure credentials"
	@echo "2. Run: make upload-data"
	@echo "3. Run: make preprocess-data"
	@echo "4. Run: make train-baseline"

# Docker (future)
docker-build:
	docker build -t $(PROJECT_NAME):latest .

docker-run:
	docker run -it --rm --gpus all -v $(PWD):/workspace $(PROJECT_NAME):latest
