import torch
import torch.nn as nn
from authentipic.data.downloader import DataDownloader
from authentipic.data.dataset_factory import DatasetFactory
from authentipic.models.authentipic_model import create_model
from authentipic.training.trainer import Trainer
from authentipic.inference.predictor import Predictor
from authentipic.config.config import load_config
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def get_transform(train=True):
    if train:
        return A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def download_data(config):
    downloader = DataDownloader(config.data_dir)
    downloader.download_all()

def train(config):
    # Download data if not already present
    download_data(config)

    # Create datasets
    dataset_configs = {
        "faceforensics": {"root_dir": config.faceforensics_dir},
        "dfdc": {"root_dir": config.dfdc_dir}
    }
    full_dataset = DatasetFactory.get_combined_dataset(dataset_configs, transform=get_transform(train=True))
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)

    # Create model, optimizer, and loss function
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Initialize trainer and start training
    trainer = Trainer(model, optimizer, criterion, config.device)
    trainer.fit(train_loader, val_loader, config.num_epochs)

def infer(config):
    # Load model and run inference
    model = create_model()
    model.load_state_dict(torch.load(config.model_path))
    predictor = Predictor(model, config.device)
    
    # Create test dataset
    dataset_configs = {
        "faceforensics": {"root_dir": config.faceforensics_dir},
        "dfdc": {"root_dir": config.dfdc_dir}
    }
    full_dataset = DatasetFactory.get_combined_dataset(dataset_configs, transform=get_transform(train=False))
    
    # Split dataset and get test set
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    _, _, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4)
    
    # Run inference on test set
    predictor.predict_dataset(test_loader)

if __name__ == "__main__":
    config = load_config("config.yaml")
    
    if config.mode == "train":
        train(config)
    elif config.mode == "infer":
        infer(config)
    else:
        print("Invalid mode. Use 'train' or 'infer'.")