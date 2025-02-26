from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset
from typing import Tuple, Any


class BaseDataset(Dataset, ABC):
    @abstractmethod
    def __init__(self, root_dir: str, transform: Any = None):
        self.root_dir = root_dir
        self.transform = transform

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        pass

    @abstractmethod
    def load_data(self):
        pass
