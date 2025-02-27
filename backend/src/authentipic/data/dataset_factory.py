from typing import Dict, Any, List
from torch.utils.data import ConcatDataset
from authentipic.data.faceforensics_dataset import FaceForensicsDataset
from authentipic.data.dfdc_dataset import DFDCDataset


class DatasetFactory:
    DATASET_REGISTRY = {
		"faceforensics": FaceForensicsDataset,
		"dfdc": DFDCDataset,
	}
    @staticmethod
    def get_dataset(dataset_name: str, **kwargs: Any) -> Any:
        if dataset_name not in DatasetFactory.DATASET_REGISTRY:
              raise ValueError(f"Dataset {dataset_name} not supported")
        return DatasetFactory.DATASET_REGISTRY[dataset_name](**kwargs)

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

        return ConcatDataset(datasets)
