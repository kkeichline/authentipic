import torch
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class Predictor:
    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model
        self.device = device
        self.logger = logging.getLogger(__name__)

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            input = input.to(self.device)
            output = self.model(input)
            return torch.softmax(output, dim=1)

    def predict_batch(self, inputs: torch.Tensor) -> np.ndarray:
        predictions = self.predict(inputs)
        return predictions.cpu().numpy()

    def predict_dataset(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Predicting"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions = torch.softmax(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())

        return np.array(all_predictions), np.array(all_labels)

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for binary classification.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for the positive class
            threshold: Classification threshold
        """
        y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "auc_roc": roc_auc,
            "average_precision": average_precision_score(y_true, y_pred_proba[:, 1]),
        }

        return metrics

    def plot_roc_curve(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: Path = None
    ) -> None:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path / "roc_curve.png")
            plt.close()
        else:
            plt.show()

    def plot_precision_recall_curve(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: Path = None
    ) -> None:
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])

        plt.figure()
        plt.plot(
            recall,
            precision,
            color="darkorange",
            lw=2,
            label=f"PR curve (AP = {avg_precision:.2f})",
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path / "pr_curve.png")
            plt.close()
        else:
            plt.show()

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Path = None
    ) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        if save_path:
            plt.savefig(save_path / "confusion_matrix.png")
            plt.close()
        else:
            plt.show()

    def analyze_errors(
        self, dataloader: torch.utils.data.DataLoader, threshold: float = 0.5
    ) -> Dict[str, List]:
        """Analyze misclassified examples."""
        self.model.eval()
        false_positives = []
        false_negatives = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                predictions = self.predict(inputs)
                pred_labels = (predictions[:, 1] >= threshold).cpu().numpy()

                # Store misclassified examples
                for i, (pred, true) in enumerate(zip(pred_labels, labels)):
                    if pred == 1 and true == 0:
                        false_positives.append(
                            {
                                "input": inputs[i].cpu(),
                                "confidence": predictions[i, 1].item(),
                            }
                        )
                    elif pred == 0 and true == 1:
                        false_negatives.append(
                            {
                                "input": inputs[i].cpu(),
                                "confidence": predictions[i, 1].item(),
                            }
                        )

        return {"false_positives": false_positives, "false_negatives": false_negatives}

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        threshold: float = 0.5,
        save_plots: bool = False,
        output_dir: str = None,
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of the model.

        Args:
            dataloader: DataLoader containing test data
            threshold: Classification threshold
            save_plots: Whether to save visualization plots
            output_dir: Directory to save plots if save_plots is True
        """
        predictions, labels = self.predict_dataset(dataloader)
        metrics = self.calculate_metrics(labels, predictions, threshold)

        self.logger.info("Evaluation Metrics:")
        for metric_name, value in metrics.items():
            self.logger.info(f"{metric_name}: {value:.4f}")

        if save_plots:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            self.plot_roc_curve(labels, predictions, output_path)
            self.plot_precision_recall_curve(labels, predictions, output_path)
            self.plot_confusion_matrix(
                labels, (predictions[:, 1] >= threshold).astype(int), output_path
            )

        return metrics
