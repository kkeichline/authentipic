import os
import cv2
import json
from typing import Any
from authentipic.data.base_dataset import BaseDataset


class DFDCDataset(BaseDataset):
    def __init__(self, root_dir: str, transform: Any = None):
        super().__init__(root_dir, transform)
        self.images, self.labels = self.load_data()

    def load_data(self):
        images, labels = [], []
        metadata_file = os.path.join(self.root_dir, "metadata.json")

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        for video_file, info in metadata.items():
            video_path = os.path.join(self.root_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                images.append(frame)
                labels.append(0 if info["label"] == "REAL" else 1)
            cap.release()

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
