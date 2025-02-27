"""
Annotation Pipeline for AuthentiPic

This module provides functionality for annotating images to create training data
and collect human feedback for the AuthentiPic system.
"""

import json
import logging
import hashlib
from typing import Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AnnotationPipeline")


class AnnotationManager:
    """Manages the annotation process for images"""

    def __init__(
        self,
        annotation_dir: str = "data/annotations",
        image_dir: str = "data/raw",
        output_dir: str = "data/processed/annotations",
        annotation_schema: Optional[Dict] = None,
    ):
        """
        Initialize the annotation manager

        Args:
            annotation_dir: Directory to store annotation files
            image_dir: Directory containing images to annotate
            output_dir: Directory to save processed annotations
            annotation_schema: Schema for annotations (labels, types, etc.)
        """
        self.annotation_dir = Path(annotation_dir)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)

        # Create directories if they don't exist
        self.annotation_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set default annotation schema if none provided
        if annotation_schema is None:
            self.annotation_schema = {
                "labels": {
                    "real": "Authentic image with minimal editing",
                    "ai_generated": "AI-generated image",
                    "manipulated": "Significantly manipulated image",
                    "unsure": "Cannot determine",
                },
                "confidence_levels": [1, 2, 3, 4, 5],  # 1=low, 5=high
                "metadata_fields": [
                    "annotator_id",
                    "annotation_timestamp",
                    "annotation_duration_sec",
                    "device_info",
                ],
            }
        else:
            self.annotation_schema = annotation_schema

        # Load existing annotations if available
        self.annotations = self._load_annotations()

    def _load_annotations(self) -> Dict:
        """
        Load existing annotations from disk

        Returns:
            Dictionary of existing annotations
        """
        annotation_file = self.annotation_dir / "annotations.json"
        if annotation_file.exists():
            try:
                with open(annotation_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(
                    f"Error loading annotation file: {annotation_file}. Starting fresh."
                )
                return {"images": {}, "metadata": {"schema": self.annotation_schema}}
        return {"images": {}, "metadata": {"schema": self.annotation_schema}}

    def _save_annotations(self) -> None:
        """Save annotations to disk"""
        annotation_file = self.annotation_dir / "annotations.json"
        with open(annotation_file, "w") as f:
            json.dump(self.annotations, f, indent=2)
        logger.info(f"Annotations saved to {annotation_file}")

    def get_next_image_to_annotate(self, annotator_id: str) -> Optional[str]:
        """
        Get the next image that needs annotation

        Args:
            annotator_id: ID of the current annotator

        Returns:
            Path to the next image, or None if no images need annotation
        """
        # Get all images from the image directory
        all_images = [
            str(img.relative_to(self.image_dir))
            for img in self.image_dir.glob("**/*.jpg")
        ]
        all_images.extend(
            [
                str(img.relative_to(self.image_dir))
                for img in self.image_dir.glob("**/*.png")
            ]
        )

        # Filter images that need annotation (not annotated or need more annotators)
        images_needing_annotation = []

        for img_path in all_images:
            # Check if image exists in annotations
            if img_path not in self.annotations["images"]:
                images_needing_annotation.append(img_path)
                continue

            # Check if this annotator has already annotated the image
            existing_annotators = [
                a["annotator_id"]
                for a in self.annotations["images"][img_path]["annotations"]
            ]
            if annotator_id not in existing_annotators:
                images_needing_annotation.append(img_path)

        if not images_needing_annotation:
            return None

        # For simplicity, return the first image needing annotation
        # In a production system, you might want to prioritize images based on certain criteria
        return images_needing_annotation[0]

    def add_annotation(
        self,
        image_path: str,
        label: str,
        annotator_id: str,
        confidence: int,
        annotation_time_sec: float,
        notes: Optional[str] = None,
        device_info: Optional[Dict] = None,
    ) -> bool:
        """
        Add an annotation for an image

        Args:
            image_path: Path to the image being annotated
            label: Annotation label (must be in schema)
            annotator_id: ID of the annotator
            confidence: Confidence level (must be in schema)
            annotation_time_sec: Time taken to annotate in seconds
            notes: Optional notes from the annotator
            device_info: Optional information about the device used

        Returns:
            True if annotation was successful, False otherwise
        """
        # Validate inputs
        if label not in self.annotation_schema["labels"]:
            logger.error(
                f"Invalid label: {label}. Must be one of {list(self.annotation_schema['labels'].keys())}"
            )
            return False

        if confidence not in self.annotation_schema["confidence_levels"]:
            logger.error(
                f"Invalid confidence level: {confidence}. Must be one of {self.annotation_schema['confidence_levels']}"
            )
            return False

        # Normalize image path
        rel_image_path = (
            str(Path(image_path).relative_to(self.image_dir))
            if self.image_dir in Path(image_path).parents
            else image_path
        )

        # Create image entry if it doesn't exist
        if rel_image_path not in self.annotations["images"]:
            self.annotations["images"][rel_image_path] = {
                "file_path": rel_image_path,
                "annotations": [],
                "metadata": self._extract_image_metadata(
                    self.image_dir / rel_image_path
                ),
            }

        # Create annotation
        annotation = {
            "annotator_id": annotator_id,
            "label": label,
            "confidence": confidence,
            "annotation_timestamp": datetime.now().isoformat(),
            "annotation_duration_sec": annotation_time_sec,
        }

        if notes:
            annotation["notes"] = notes

        if device_info:
            annotation["device_info"] = device_info

        # Add annotation to image
        self.annotations["images"][rel_image_path]["annotations"].append(annotation)

        # Save annotations
        self._save_annotations()

        logger.info(f"Added annotation for {rel_image_path} by {annotator_id}")
        return True

    def _extract_image_metadata(self, image_path: Path) -> Dict:
        """
        Extract metadata from an image

        Args:
            image_path: Path to the image

        Returns:
            Dictionary of image metadata
        """
        metadata = {
            "filename": image_path.name,
            "added_timestamp": datetime.now().isoformat(),
        }

        try:
            # Calculate image hash for identification
            with open(image_path, "rb") as f:
                metadata["hash"] = hashlib.md5(f.read()).hexdigest()

            # Extract basic image properties
            with Image.open(image_path) as img:
                metadata["width"] = img.width
                metadata["height"] = img.height
                metadata["format"] = img.format
                metadata["mode"] = img.mode

                # Extract EXIF data if available
                if hasattr(img, "_getexif") and img._getexif():
                    exif = img._getexif()
                    if exif:
                        # Only extract non-binary EXIF data
                        metadata["exif"] = {
                            str(k): str(v)
                            for k, v in exif.items()
                            if isinstance(v, (str, int, float))
                        }
        except Exception as e:
            logger.warning(f"Error extracting metadata from {image_path}: {e}")

        return metadata

    def export_annotations(self, format: str = "csv") -> str:
        """
        Export annotations to a specified format

        Args:
            format: Export format ("csv", "json", or "parquet")

        Returns:
            Path to the exported file
        """
        # Create flattened dataframe of annotations
        records = []

        for img_path, img_data in self.annotations["images"].items():
            for annotation in img_data["annotations"]:
                record = {
                    "image_path": img_path,
                    "width": img_data["metadata"].get("width"),
                    "height": img_data["metadata"].get("height"),
                    "format": img_data["metadata"].get("format"),
                    "image_hash": img_data["metadata"].get("hash"),
                }
                record.update(annotation)
                records.append(record)

        if not records:
            logger.warning("No annotations to export")
            return ""

        # Convert to DataFrame
        df = pd.DataFrame(records)

        # Export in requested format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format.lower() == "csv":
            output_path = self.output_dir / f"annotations_{timestamp}.csv"
            df.to_csv(output_path, index=False)
        elif format.lower() == "json":
            output_path = self.output_dir / f"annotations_{timestamp}.json"
            df.to_json(output_path, orient="records", indent=2)
        elif format.lower() == "parquet":
            output_path = self.output_dir / f"annotations_{timestamp}.parquet"
            df.to_parquet(output_path, index=False)
        else:
            logger.error(f"Unsupported export format: {format}")
            return ""

        logger.info(f"Exported {len(records)} annotations to {output_path}")
        return str(output_path)

    def get_annotation_statistics(self) -> Dict:
        """
        Calculate statistics on the current annotations

        Returns:
            Dictionary of annotation statistics
        """
        if not self.annotations["images"]:
            return {"total_images": 0, "total_annotations": 0}

        # Count total annotations
        total_annotations = sum(
            len(img_data["annotations"])
            for img_data in self.annotations["images"].values()
        )

        # Count annotations by label
        labels_count = {}
        for img_data in self.annotations["images"].values():
            for annotation in img_data["annotations"]:
                label = annotation["label"]
                if label not in labels_count:
                    labels_count[label] = 0
                labels_count[label] += 1

        # Calculate agreement statistics
        agreement_stats = self._calculate_agreement_statistics()

        return {
            "total_images": len(self.annotations["images"]),
            "total_annotations": total_annotations,
            "annotations_per_label": labels_count,
            "agreement_statistics": agreement_stats,
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_agreement_statistics(self) -> Dict:
        """
        Calculate inter-annotator agreement statistics

        Returns:
            Dictionary of agreement statistics
        """
        # Get images with multiple annotations
        multi_annotated_images = {
            img_path: img_data
            for img_path, img_data in self.annotations["images"].items()
            if len(img_data["annotations"]) > 1
        }

        if not multi_annotated_images:
            return {"multi_annotated_images": 0}

        # Calculate agreement percentages
        total_agreement = 0
        full_agreement_count = 0

        for img_path, img_data in multi_annotated_images.items():
            annotations = img_data["annotations"]
            labels = [a["label"] for a in annotations]

            # Check if all labels are the same
            if len(set(labels)) == 1:
                full_agreement_count += 1
                total_agreement += 1
            else:
                # Calculate partial agreement as percentage of most common label
                most_common_label = max(set(labels), key=labels.count)
                agreement_ratio = labels.count(most_common_label) / len(labels)
                total_agreement += agreement_ratio

        # Calculate average agreement across all multi-annotated images
        avg_agreement = (
            total_agreement / len(multi_annotated_images)
            if multi_annotated_images
            else 0
        )

        return {
            "multi_annotated_images": len(multi_annotated_images),
            "full_agreement_count": full_agreement_count,
            "full_agreement_percentage": (
                full_agreement_count / len(multi_annotated_images) * 100
            )
            if multi_annotated_images
            else 0,
            "average_agreement": avg_agreement * 100,  # as percentage
        }

    def get_image_requiring_review(self) -> Optional[str]:
        """
        Get an image that has conflicting annotations and requires review

        Returns:
            Path to an image requiring review, or None if no reviews needed
        """
        review_candidates = []

        for img_path, img_data in self.annotations["images"].items():
            annotations = img_data["annotations"]

            # Skip images with less than 2 annotations
            if len(annotations) < 2:
                continue

            # Check if all annotators agree
            labels = [a["label"] for a in annotations]
            if len(set(labels)) > 1:
                review_candidates.append(img_path)

        if not review_candidates:
            return None

        # Return the first candidate for review
        return review_candidates[0]

    def add_reviewed_annotation(
        self,
        image_path: str,
        reviewed_label: str,
        reviewer_id: str,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Add a reviewed annotation that resolves conflicting annotations

        Args:
            image_path: Path to the image being reviewed
            reviewed_label: Final label after review
            reviewer_id: ID of the reviewer
            notes: Optional notes from the reviewer

        Returns:
            True if review was successful, False otherwise
        """
        # Validate inputs
        if reviewed_label not in self.annotation_schema["labels"]:
            logger.error(
                f"Invalid label: {reviewed_label}. Must be one of {list(self.annotation_schema['labels'].keys())}"
            )
            return False

        # Normalize image path
        rel_image_path = (
            str(Path(image_path).relative_to(self.image_dir))
            if self.image_dir in Path(image_path).parents
            else image_path
        )

        # Ensure image exists in annotations
        if rel_image_path not in self.annotations["images"]:
            logger.error(f"Image {rel_image_path} not found in annotations")
            return False

        # Add reviewed annotation
        self.annotations["images"][rel_image_path]["reviewed"] = {
            "reviewer_id": reviewer_id,
            "reviewed_label": reviewed_label,
            "review_timestamp": datetime.now().isoformat(),
        }

        if notes:
            self.annotations["images"][rel_image_path]["reviewed"]["notes"] = notes

        # Save annotations
        self._save_annotations()

        logger.info(f"Added reviewed annotation for {rel_image_path} by {reviewer_id}")
        return True

    def generate_training_dataset(
        self,
        output_path: str,
        min_confidence: int = 3,
        require_agreement: bool = True,
        prefer_reviewed: bool = True,
    ) -> Tuple[int, int]:
        """
        Generate a training dataset CSV with high-quality annotations

        Args:
            output_path: Path to save the dataset
            min_confidence: Minimum confidence level to include
            require_agreement: Whether to require annotator agreement
            prefer_reviewed: Whether to prefer reviewed annotations

        Returns:
            Tuple of (number of included images, number of excluded images)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        excluded = 0

        for img_path, img_data in self.annotations["images"].items():
            full_path = str(self.image_dir / img_path)

            # If image has a reviewed annotation and we prefer it
            if prefer_reviewed and "reviewed" in img_data:
                label = img_data["reviewed"]["reviewed_label"]
                records.append(
                    {"image_path": full_path, "label": label, "source": "reviewed"}
                )
                continue

            # Get annotations with minimum confidence
            high_conf_annotations = [
                a for a in img_data["annotations"] if a["confidence"] >= min_confidence
            ]

            if not high_conf_annotations:
                excluded += 1
                continue

            # Check agreement if required
            if require_agreement and len(high_conf_annotations) > 1:
                labels = [a["label"] for a in high_conf_annotations]
                if len(set(labels)) > 1:
                    excluded += 1
                    continue

            # Use the highest confidence annotation or most common label
            if len(high_conf_annotations) == 1:
                chosen_annotation = high_conf_annotations[0]
            else:
                # Get the most common label
                labels = [a["label"] for a in high_conf_annotations]
                most_common_label = max(set(labels), key=labels.count)

                # Find the highest confidence annotation with this label
                chosen_annotation = max(
                    [
                        a
                        for a in high_conf_annotations
                        if a["label"] == most_common_label
                    ],
                    key=lambda x: x["confidence"],
                )

            records.append(
                {
                    "image_path": full_path,
                    "label": chosen_annotation["label"],
                    "confidence": chosen_annotation["confidence"],
                    "annotator_id": chosen_annotation["annotator_id"],
                    "source": "annotation",
                }
            )

        # Save to CSV
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)

        logger.info(
            f"Generated training dataset with {len(records)} images ({excluded} excluded) at {output_path}"
        )
        return len(records), excluded
