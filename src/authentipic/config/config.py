# src/authentipic/config/config.py

import yaml
from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    data_dir: str
    faceforensics_dir: str
    dfdc_dir: str
    train_split: float
    val_split: float
    test_split: float


@dataclass
class ModelConfig:
    model_path: str
    architecture: str


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    num_epochs: int
    optimizer: str
    scheduler: str
    early_stopping_patience: int


@dataclass
class AugmentationConfig:
    random_crop_size: int
    horizontal_flip_prob: float
    brightness_contrast_prob: float


@dataclass
class InferenceConfig:
    threshold: float
    batch_size: int


@dataclass
class HardwareConfig:
    device: str
    num_workers: int


@dataclass
class LoggingConfig:
    log_dir: str
    tensorboard_dir: str
    save_frequency: int


@dataclass
class ExperimentConfig:
    name: str
    tags: List[str]


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    inference: InferenceConfig
    hardware: HardwareConfig
    logging: LoggingConfig
    experiment: ExperimentConfig
    mode: str
    seed: int


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    return Config(
        data=DataConfig(**config_dict["data"]),
        model=ModelConfig(**config_dict["model"]),
        training=TrainingConfig(**config_dict["training"]),
        augmentation=AugmentationConfig(**config_dict["augmentation"]),
        inference=InferenceConfig(**config_dict["inference"]),
        hardware=HardwareConfig(**config_dict["hardware"]),
        logging=LoggingConfig(**config_dict["logging"]),
        experiment=ExperimentConfig(**config_dict["experiment"]),
        mode=config_dict["mode"],
        seed=config_dict["seed"],
    )
