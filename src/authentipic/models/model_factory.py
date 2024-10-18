# src/authentipic/models/model_factory.py

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any


class BaseModel(nn.Module):
    def __init__(self, num_classes: int):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes

    def get_classifier(self, in_features: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
        )


class ResNetModel(BaseModel):
    def __init__(self, num_classes: int, pretrained: bool):
        super(ResNetModel, self).__init__(num_classes)
        self.base_model = models.resnet50(pretrained=pretrained)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.classifier = self.get_classifier(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        return self.classifier(features)


class EfficientNetModel(BaseModel):
    def __init__(self, num_classes: int, pretrained: bool):
        super(EfficientNetModel, self).__init__(num_classes)
        self.base_model = models.efficientnet_b0(pretrained=pretrained)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()
        self.classifier = self.get_classifier(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        return self.classifier(features)


class ModelFactory:
    @staticmethod
    def get_model(config: Dict[str, Any]) -> nn.Module:
        architecture = config["model"]["architecture"]
        num_classes = config["model"]["num_classes"]
        pretrained = config["model"]["pretrained"]

        if architecture == "resnet50":
            return ResNetModel(num_classes, pretrained)
        elif architecture == "efficientnet_b0":
            return EfficientNetModel(num_classes, pretrained)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")


def create_model(config: Dict[str, Any]) -> nn.Module:
    return ModelFactory.get_model(config)
