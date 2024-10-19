from typing import Dict, Any, List
from authentipic.data.faceforensics_dataset import FaceForensicsDataset
from authentipic.data.dfdc_dataset import DFDCDataset


class DatasetFactory:
    @staticmethod
    def get_dataset(dataset_name: str, **kwargs: Any) -> Any:
        if dataset_name == "faceforensics":
            return FaceForensicsDataset(**kwargs)
        elif dataset_name == "dfdc":
            return DFDCDataset(**kwargs)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    @staticmethod
    def get_combined_dataset(
        dataset_configs: Dict[str, Dict[str, Any]],
        transform: Any = None,
        exclude: List[str] = None,
    ) -> Any:
        if exclude is None:
            exclude = []

        datasets = []
        for dataset_name, config in dataset_configs.items():
            if dataset_name not in exclude:
                config["transform"] = transform
                datasets.append(DatasetFactory.get_dataset(dataset_name, **config))

        from torch.utils.data import ConcatDataset

        return ConcatDataset(datasets)
