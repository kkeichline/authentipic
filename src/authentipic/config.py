from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class DataConfig:
    data_dir: str = "./data/raw"
    faceforensics_dir: str = "./data/raw/faceforensics"
    dfdc_dir: str = "./data/raw/dfdc/train_sample_videos"
    exclude_datasets: List[str] = field(default_factory=lambda: ["faceforensics"])
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class ModelConfig:
    architecture: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 2


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 50
    checkpoint_dir: str = "./checkpoints"
    resume_from: str = None
    scheduler_type: str = "step"
    scheduler_params: dict = field(default_factory=lambda: {"step_size": 10, "gamma": 0.1})
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.001


@dataclass
class InferenceConfig:
    best_model_path: str = "./checkpoints/best_model.pth"
    batch_size: int = 64
    threshold: float = 0.5


@dataclass
class HardwareConfig:
    device: str = "mps"
    num_workers: int = 4
    pin_memory: bool = True
    use_mixed_precision: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    

config = Config()
