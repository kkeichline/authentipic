import argparse
import logging
import torch
import torch.nn as nn

# from authentipic.data.downloader import DataDownloader
from authentipic.data.dataset_factory import DatasetFactory
from authentipic.models.model_factory import create_model
from authentipic.training.trainer import Trainer
from authentipic.inference.predictor import Predictor
from authentipic.config import config
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)


def get_device():
    if config.hardware.device == "mps" and torch.backends.mps.is_available():
        logger.info("Using MPS device.")
        return torch.device("mps")
    logger.info("Using CPU device.")
    return torch.device("cpu")


def get_transform(train=True):
    if train:
        return A.Compose(
            [
                A.RandomResizedCrop(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


# def download_data(config):
#    logger.info("Downloading data...")
#     downloader = DataDownloader(config.data_dir)
#     downloader.download_all()


def train():
    # download_data(config)

    device = get_device()

    # Create datasets
    dataset_configs = {
        "faceforensics": {"root_dir": config.data.faceforensics_dir},
        "dfdc": {"root_dir": config.data.dfdc_dir},
    }

    full_dataset = DatasetFactory.get_combined_dataset(
        dataset_configs,
        transform=get_transform(train=True),
        exclude=config.data.exclude_datasets,
    )

    # Split dataset
    train_size = int(config.data.train_split * len(full_dataset))
    val_size = int(config.data.val_split * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.hardware.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.hardware.num_workers,
    )

    # Create model
    model = create_model(config.model)
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Create scheduler and early stopping
    scheduler = Trainer.create_scheduler(
        config.training.scheduler_type, optimizer, **config.training.scheduler_params
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


def infer(config):
    # Load model for inference
    device = get_device()

    # Create and load model
    model = create_model(config.model)
    checkpoint = torch.load(config.inference.best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    predictor = Predictor(model, device)

    # Create test dataset
    dataset_configs = {
        "faceforensics": {"root_dir": config.faceforensics_dir},
        "dfdc": {"root_dir": config.dfdc_dir},
    }

    full_dataset = DatasetFactory.get_combined_dataset(
        dataset_configs,
        transform=get_transform(train=False),
        exclude=config.exclude_datasets,
    )

    # Split dataset and get test set
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    _, _, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4)

    # Run inference and evaluation on test set
    predictor.evaluate(
        test_loader,
        threshold=config.inference_threshold,
        save_plots=config.save_plots,
        output_dir=config.output_dir,
    )
    error_analysis = predictor.analyze_errors(test_loader)
    print(f"Number of false positives: {len(error_analysis['false_positives'])}")
    print(f"Number of false negatives: {len(error_analysis['false_negatives'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AuthentiPic: Deepfake Detection")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "infer"],
        required=True,
        help="Mode of operation",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "infer":
        infer()
    else:
        print("Invalid mode. Use 'train' or 'infer'.")
