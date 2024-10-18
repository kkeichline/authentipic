import argparse
import torch
import torch.nn as nn
from authentipic.data.downloader import DataDownloader
from authentipic.data.dataset_factory import DatasetFactory
from authentipic.models.model_factory import create_model
from authentipic.training.trainer import Trainer
from authentipic.inference.predictor import Predictor
from authentipic.config.config import load_config
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_device(config):
    if config.mac_m2.use_mps_if_available and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


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


def download_data(config):
    downloader = DataDownloader(config.data_dir)
    downloader.download_all()


def train(config):
    # Download data if not already present
    download_data(config)

    # Create datasets
    dataset_configs = {
        "faceforensics": {"root_dir": config.faceforensics_dir},
        "dfdc": {"root_dir": config.dfdc_dir},
    }

    full_dataset = DatasetFactory.get_combined_dataset(
        dataset_configs,
        transform=get_transform(train=True),
        exclude=config.exclude_datasets,
    )

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)

    # Create model, optimizer, and loss function
    model = create_model()

    device = get_device(config)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Create scheduler and early stopping
    scheduler = Trainer.create_scheduler(
        config.scheduler_type, optimizer, **config.scheduler_params
    )
    early_stopping = Trainer.create_early_stopping(**config.early_stopping_params)

    # Initialize trainer
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        config.device,
        scheduler=scheduler,
        early_stopping=early_stopping,
        checkpoint_dir=config.checkpoint_dir,
    )

    # Start training
    trainer.fit(
        train_loader, val_loader, config.num_epochs, resume_from=config.resume_from
    )


def infer(config):
    # Load model for inference
    model = create_model()
    checkpoint = torch.load(config.best_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)

    predictor = Predictor(model, config.device)

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
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "infer"],
        required=True,
        help="Mode of operation",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "train":
        train(config)
    elif args.mode == "infer":
        infer(config)
    else:
        print("Invalid mode. Use 'train' or 'infer'.")
