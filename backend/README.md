# AuthentiPic Backend

## Overview

AuthentiPic is an AI-powered image detection system that identifies AI-generated or manipulated images. The backend provides the core functionality for training detection models, performing inference, and managing annotations.

## Table of Contents

- Project Structure
- Core Functionalities
- Components
- Setup and Installation
- Usage
- Development

## Project Structure

```
backend/
│
├── config/
│   ├── data_config.yaml     # Configuration for data sources
│   └── model_config.yaml    # Configuration for model architecture
│
├── docs/
│   ├── api_reference.md     # API documentation
│   ├── data_dictionary.md   # Data dictionary
│   └── model_architecture.md # Model architecture documentation
│
├── src/
│   └── authentipic/
│       ├── annotation/      # Annotation tools for collecting training data
│       │   ├── app.py       # Web application for annotation
│       │   ├── manager.py   # Annotation management system
│       │   └── utils.py     # Utility functions for annotation
│       │
│       ├── converter/       # Model conversion tools
│       │   └── converter.py # Conversion from PyTorch to TensorFlow.js
│       │
│       ├── inference/       # Inference functionality
│       │   └── predictor.py # Predictor class for model inference
│       │
│       ├── models/          # Model definitions
│       │   ├── model_factory.py # Factory for creating models
│       │   └── __init__.py
│       │
│       ├── training/        # Training functionality
│       │   ├── trainer.py   # Trainer class for model training
│       │   └── __init__.py
│       │
│       ├── visualizations/  # Visualization tools
│       │   ├── visualize.py # Visualization functions
│       │   └── __init__.py
│       │
│       ├── config.py        # Configuration management
│       ├── main.py          # Main entry point
│       └── __init__.py
│
├── tests/                   # Unit tests
│   ├── conftest.py          # Test configuration
│   ├── test_data.py         # Tests for data processing
│   ├── test_features.py     # Tests for feature extraction
│   └── test_models.py       # Tests for models
│
├── .gitignore               # Git ignore file
├── Makefile                 # Project automation
├── pyproject.toml           # Poetry dependency management
└── README.md                # This file
```

## Core Functionalities

1. **Image Detection**:
   - Detects AI-generated or manipulated images using deep learning models
   - Supports multiple model architectures (ResNet50, EfficientNet)
   - Provides confidence scores and visualization of results

2. **Annotation Pipeline**:
   - Web-based annotation tool for collecting human feedback
   - Annotation management system for tracking and quality control
   - Dataset generation for model training

3. **Model Training**:
   - Training pipeline for deep learning models
   - Support for transfer learning with pre-trained models
   - Early stopping, learning rate scheduling, and checkpointing

4. **Model Conversion**:
   - Conversion from PyTorch models to ONNX format
   - Conversion from ONNX to TensorFlow.js for web deployment

5. **Inference Engine**:
   - Fast inference on new images
   - Comprehensive metrics and error analysis
   - Visualization of model outputs and confidence

## Components

### 1. Annotation System

The annotation system allows for collecting human feedback on images, which is crucial for training and evaluating models. Key components:

- **AnnotationManager**: Manages the annotation workflow and storage
- **WebAnnotationApp**: Flask-based web interface for manual annotation
- **Dataset Generation**: Tools for creating training datasets from annotations

### 2. Model Architecture

The system supports multiple model architectures through a factory pattern:

- **BaseModel**: Abstract base class for all models
- **ResNetModel**: ResNet50-based architecture
- **EfficientNetModel**: EfficientNet-based architecture
- **ModelFactory**: Factory class for creating model instances

### 3. Training Pipeline

The training pipeline handles model training with advanced features:

- **Trainer**: Core class for training models with PyTorch
- **EarlyStopping**: Prevents overfitting by stopping training when validation loss plateaus
- **Learning Rate Schedulers**: Various schedulers for dynamically adjusting learning rates

### 4. Inference Engine

The inference engine provides tools for making predictions:

- **Predictor**: Makes predictions on new images
- **Metrics**: Calculates accuracy, precision, recall, F1, ROC curve, and more
- **Visualization**: Creates visualizations of model outputs and confidence

### 5. Model Conversion

The conversion module allows deployment to different platforms:

- **ModelConverter**: Converts PyTorch models to ONNX and TensorFlow.js formats
- **Export Pipeline**: Complete pipeline for exporting models for web deployment

## Setup and Installation

### Prerequisites

- Python 3.10+
- Poetry for dependency management

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/authentipic.git
   cd authentipic
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Configure data paths in data_config.yaml

## Usage

### Training a Model

```bash
# Using Poetry
poetry run python src/authentipic/main.py --mode train

# Using the Makefile
make run-train
```

### Running Inference

```bash
# Using Poetry
poetry run python src/authentipic/main.py --mode predict --image_path /path/to/image.jpg

# Using the Makefile
make run-predict IMAGE_PATH=/path/to/image.jpg
```

### Running the Annotation Tool

```bash
# Using Poetry
poetry run python src/authentipic/annotation/app.py

# Using the Makefile
make run-annotation-app
```

### Exporting Annotations to CSV

```bash
# Using Poetry
poetry run python -m authentipic.annotation.manager --export

# Using the Makefile
make export-annotations
```

## Development

### Running Tests

```bash
# Using Poetry
poetry run pytest

# Using the Makefile
make test
```

### Code Quality

```bash
# Linting
make lint

# Formatting
make format

# Full quality check
make quality-check
```

### Updating Dependencies

```bash
make update
```