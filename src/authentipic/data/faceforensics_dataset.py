import os
import cv2
from typing import Any
from authentipic.data.base_dataset import BaseDataset


class FaceForensicsDataset(BaseDataset):
    def __init__(self, root_dir: str, transform: Any = None):
        super().__init__(root_dir, transform)
        self.images, self.labels = self.load_data()

    def load_data(self):
        images, labels = [], []
        real_dir = os.path.join(
            self.root_dir, "original_sequences", "youtube", "c23", "videos"
        )
        fake_dir = os.path.join(
            self.root_dir, "manipulated_sequences", "Deepfakes", "c23", "videos"
        )

        for directory, label in [(real_dir, 0), (fake_dir, 1)]:
            for video_file in os.listdir(directory):
                if video_file.endswith(".mp4"):
                    video_path = os.path.join(directory, video_file)
                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()
                    if ret:
                        images.append(frame)
                        labels.append(label)
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
