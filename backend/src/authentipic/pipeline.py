import argparse
import logging
import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

from authentipic.annotation.manager import AnnotationManager
from authentipic.config import config
from authentipic.main import train as train_model
from authentipic.models.model_factory import ModelFactory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("authentipic.pipeline")


class AnnotationDataset(Dataset):
    """Dataset for loading images from annotation CSV files."""

    def __init__(self, csv_path, transform=None):
        """
        Initialize dataset from annotation CSV file.

        Args:
            csv_path: Path to CSV file with columns 'image_path' and 'label'
            transform: Transform to apply to images
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        # Map string labels to integers
        label_map = {"real": 0, "ai_generated": 1, "manipulated": 1}
        self.labels = self.data["label"].map(lambda x: label_map.get(x.lower(), 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]["image_path"]
        label = self.labels.iloc[idx]

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            # Handle missing/corrupted images
            logger.warning(f"Could not read image at {image_path}")
            # Return a black image as placeholder
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


def setup_annotation_manager(config_path=None):
    """Set up the annotation manager."""
    annotation_config = {}
    if config_path and os.path.exists(config_path):
        try:
            import json

            with open(config_path, "r") as f:
                annotation_config = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading config: {e}. Using defaults.")

    annotation_dir = annotation_config.get("annotation_dir", "data/annotations")
    image_dir = annotation_config.get("image_dir", "data/raw")
    output_dir = annotation_config.get("output_dir", "data/processed/annotations")

    return AnnotationManager(
        annotation_dir=annotation_dir, image_dir=image_dir, output_dir=output_dir
    )


def generate_dataset(
    annotation_manager, output_path, min_confidence=3, require_agreement=True
):
    """Generate a training dataset from annotations."""
    logger.info(f"Generating training dataset at {output_path}")
    included, excluded = annotation_manager.generate_training_dataset(
        output_path=output_path,
        min_confidence=min_confidence,
        require_agreement=require_agreement,
    )

    logger.info(
        f"Generated dataset with {included} samples (excluded {excluded} samples)"
    )
    return included, excluded


def train_from_annotations(dataset_path, model_config=None):
    """
    Train model using annotation dataset.

    Args:
        dataset_path: Path to annotation dataset CSV
        model_config: Optional model configuration override
    """
    from authentipic.data.dataset_factory import DatasetFactory

    logger.info(f"Training model from annotation dataset: {dataset_path}")

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return False

    try:
        # Import here to avoid circular imports
        from authentipic.training.trainer import Trainer
        import torch.nn as nn
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        from torch.utils.data import random_split, DataLoader

        # Get device
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Create transforms
        transform = A.Compose(
            [
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        val_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        # Create dataset from annotations
        dataset = AnnotationDataset(dataset_path, transform=transform)

        # Split into train/val sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Override val transform
        val_dataset.dataset.transform = val_transform

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.hardware.num_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.hardware.num_workers,
        )

        # Create model
        model_cfg = model_config or config.model
        model = ModelFactory.get_model(model_cfg)
        model = model.to(device)

        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.training.learning_rate
        )

        # Create loss function
        criterion = nn.CrossEntropyLoss()

        # Create scheduler and early stopping
        scheduler = Trainer.create_scheduler(
            config.training.scheduler_type,
            optimizer,
            **config.training.scheduler_params,
        )
        early_stopping = Trainer.create_early_stopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_delta,
        )

        # Initialize trainer
        trainer = Trainer(
            model,
            optimizer,
            criterion,
            device,
            scheduler=scheduler,
            early_stopping=early_stopping,
            checkpoint_dir=config.training.checkpoint_dir,
        )

        # Start training
        trainer.fit(
            train_loader,
            val_loader,
            config.training.num_epochs,
            resume_from=config.training.resume_from,
        )

        logger.info("Training completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error training model: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def run_pipeline(args):
    """Run the complete annotation-training pipeline."""
    # Set up the annotation manager
    annotation_manager = setup_annotation_manager(args.annotation_config)

    # Step 1: Generate a dataset from annotations if specified
    dataset_path = None
    if args.generate_dataset:
        output_path = args.dataset_output or "data/processed/annotation_dataset.csv"
        included, excluded = generate_dataset(
            annotation_manager,
            output_path,
            min_confidence=args.min_confidence,
            require_agreement=not args.no_agreement_required,
        )
        dataset_path = output_path

    # Step 2: Train or retrain the model if specified
    if args.train:
        if dataset_path or args.dataset_path:
            train_path = dataset_path or args.dataset_path
            logger.info(f"Training model using annotations from: {train_path}")
            train_from_annotations(train_path)
        else:
            logger.info("Training model using standard pipeline")
            train_model()

    logger.info("Pipeline completed successfully")


def main():
    parser = argparse.ArgumentParser(
        description="AuthentiPic Annotation-Training Pipeline"
    )

    # Annotation options
    parser.add_argument(
        "--annotation-config", type=str, help="Path to annotation configuration file"
    )
    parser.add_argument(
        "--generate-dataset",
        action="store_true",
        help="Generate dataset from annotations",
    )
    parser.add_argument(
        "--dataset-output", type=str, help="Path to save the generated dataset"
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=3,
        help="Minimum confidence level for annotations (1-5)",
    )
    parser.add_argument(
        "--no-agreement-required",
        action="store_true",
        help="Don't require agreement between annotators",
    )

    # Training options
    parser.add_argument("--train", action="store_true", help="Train/retrain the model")
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to existing annotation dataset for training",
    )

    args = parser.parse_args()

    # Run the pipeline
    run_pipeline(args)


if __name__ == "__main__":
    main()
