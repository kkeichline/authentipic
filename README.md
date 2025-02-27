# AuthentiPic

AuthentiPic is an AI-powered image detection system designed to identify AI-generated or manipulated images. This repository contains both the backend and frontend components of the project.

## Table of Contents

- [Project Structure](#project-structure)
- [Backend Components](#backend-components)
- [Frontend Components](#frontend-components)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Development](#development)

## Project Structure

```
authentipic/
├── backend/
│   ├── config/
│   ├── docs/
│   ├── src/
│   │   ├── authentipic/
│   │   │   ├── annotation/
│   │   │   ├── converter/
│   │   │   ├── data/
│   │   │   ├── inference/
│   │   │   ├── models/
│   │   │   ├── training/
│   │   │   ├── visualizations/
│   │   │   ├── config.py
│   │   │   ├── main.py
│   │   │   ├── pipeline.py
│   │   │   ├── run_api.py
│   │   ├── tests/
│   ├── .gitignore
│   ├── Makefile
│   ├── pyproject.toml
│   ├── README.md
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   ├── pages/
│   ├── .gitignore
│   ├── next.config.ts
│   ├── package.json
│   ├── README.md
│   ├── tailwind.config.ts
│   ├── tsconfig.json
├── .gitignore
├── README.md
```

## Backend Components

### 1. Annotation System

- **AnnotationManager**: Manages the annotation workflow and storage.
- **WebAnnotationApp**: Flask-based web interface for manual annotation.
- **Dataset Generation**: Tools for creating training datasets from annotations.

### 2. Model Architecture

- **BaseModel**: Abstract base class for all models.
- **ResNetModel**: ResNet50-based architecture.
- **EfficientNetModel**: EfficientNet-based architecture.
- **ModelFactory**: Factory class for creating model instances.

### 3. Training Pipeline

- **Trainer**: Core class for training models with PyTorch.
- **EarlyStopping**: Prevents overfitting by stopping training when validation loss plateaus.
- **Learning Rate Schedulers**: Various schedulers for dynamically adjusting learning rates.

### 4. Inference Engine

- **Predictor**: Makes predictions on new images.
- **Metrics**: Calculates accuracy, precision, recall, F1, ROC curve, and more.
- **Visualization**: Creates visualizations of model outputs and confidence.

### 5. Model Conversion

- **ModelConverter**: Converts PyTorch models to ONNX and TensorFlow.js formats.
- **Export Pipeline**: Complete pipeline for exporting models for web deployment.

## Frontend Components

### 1. Next.js Application

- **Pages**: Contains the main pages of the application.
- **Components**: Reusable UI components.
- **App**: Main application layout and configuration.

### 2. Tailwind CSS

- **Configuration**: Tailwind CSS configuration for styling the application.

## Setup and Installation

### Prerequisites

- Python 3.10+
- Node.js
- Poetry for dependency management

### Installation

1. Clone the repository:
	```bash
	git clone https://github.com/yourusername/authentipic.git
	cd authentipic
	```

2. Install backend dependencies using Poetry:
	```bash
	cd backend
	poetry install
	```

3. Install frontend dependencies using npm:
	```bash
	cd ../frontend
	npm install
	```

## Usage

### Running the Backend

```bash
# Using Poetry
cd backend
poetry run python src/authentipic/main.py --mode train

# Using the Makefile
make run-train
```

### Running the Frontend

```bash
cd frontend
npm run dev
```

### Running the Annotation Tool

```bash
# Using Poetry
cd backend
poetry run python src/authentipic/annotation/app.py

# Using the Makefile
make run-annotation-app
```

## Development

### Running Tests

```bash
# Using Poetry
cd backend
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
