"""Evaluation metrics for sign language recognition."""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score,
)
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns


def compute_top_k_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, k: int = 5
) -> float:
    """
    Compute top-k accuracy.

    Args:
        logits: Model predictions (batch, num_classes).
        targets: Ground truth labels (batch,).
        k: Number of top predictions to consider.

    Returns:
        Top-k accuracy as a float.
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        _, top_k_preds = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = top_k_preds.eq(targets.view(-1, 1).expand_as(top_k_preds))
        correct_k = correct.sum().item()
        accuracy = correct_k / batch_size
    return accuracy


class SignRecognitionMetrics:
    """Comprehensive metrics for sign language recognition."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize metrics calculator.

        Args:
            model: Trained model.
            device: Device to run evaluation on.
            class_names: List of class names for reporting.
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model.to(device)
        self.model.eval()

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        metrics: List[str] = ["top1", "top5", "per_signer", "latency"],
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            dataloader: DataLoader for test set.
            metrics: List of metrics to compute.

        Returns:
            Dictionary with metric values.
        """
        all_predictions = []
        all_targets = []
        all_logits = []
        all_signer_ids = []
        inference_times = []

        with torch.no_grad():
            for batch in dataloader:
                keypoints = batch["keypoints"].to(self.device)
                targets = batch["label"].to(self.device)
                lengths = batch.get("sequence_length", None)
                signer_ids = batch.get("signer_id", None)

                # Measure inference time
                start_time = time.time()
                logits = self.model(keypoints, lengths)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                # Collect results
                predictions = logits.argmax(dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_logits.append(logits.cpu())

                if signer_ids is not None:
                    all_signer_ids.extend(signer_ids.cpu().numpy())

        # Convert to arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_logits = torch.cat(all_logits, dim=0)

        # Compute metrics
        results = {}

        if "top1" in metrics:
            results["top1_accuracy"] = accuracy_score(all_targets, all_predictions)

        if "top5" in metrics:
            results["top5_accuracy"] = compute_top_k_accuracy(
                all_logits, torch.tensor(all_targets), k=5
            )

        if "latency" in metrics:
            results["mean_latency_ms"] = np.mean(inference_times) * 1000
            results["std_latency_ms"] = np.std(inference_times) * 1000

        if "per_signer" in metrics and len(all_signer_ids) > 0:
            signer_metrics = self._compute_per_signer_metrics(
                all_predictions, all_targets, all_signer_ids
            )
            results.update(signer_metrics)

        if "confusion_matrix" in metrics:
            results["confusion_matrix"] = confusion_matrix(all_targets, all_predictions)

        if "per_class" in metrics:
            class_report = classification_report(
                all_targets,
                all_predictions,
                target_names=self.class_names,
                output_dict=True,
            )
            results["per_class_metrics"] = class_report

        logger.info(f"Evaluation results: {results}")
        return results

    def _compute_per_signer_metrics(
        self, predictions: np.ndarray, targets: np.ndarray, signer_ids: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute per-signer accuracy metrics.

        Args:
            predictions: Predicted labels.
            targets: Ground truth labels.
            signer_ids: Signer identifiers.

        Returns:
            Dictionary with per-signer metrics.
        """
        unique_signers = np.unique(signer_ids)
        signer_accuracies = []

        for signer_id in unique_signers:
            mask = signer_ids == signer_id
            if mask.sum() > 0:
                signer_acc = accuracy_score(targets[mask], predictions[mask])
                signer_accuracies.append(signer_acc)

        return {
            "per_signer_mean_accuracy": np.mean(signer_accuracies),
            "per_signer_std_accuracy": np.std(signer_accuracies),
            "per_signer_min_accuracy": np.min(signer_accuracies),
            "per_signer_max_accuracy": np.max(signer_accuracies),
        }

    def plot_confusion_matrix(
        self,
        confusion_mat: np.ndarray,
        output_path: str,
        top_k: Optional[int] = 50,
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            confusion_mat: Confusion matrix.
            output_path: Path to save plot.
            top_k: Show only top-k most confused classes.
        """
        if top_k:
            # Select top-k most frequent classes
            class_counts = confusion_mat.sum(axis=1)
            top_indices = np.argsort(class_counts)[-top_k:]
            confusion_mat = confusion_mat[np.ix_(top_indices, top_indices)]

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            confusion_mat,
            annot=False,
            fmt="d",
            cmap="Blues",
            xticklabels=False,
            yticklabels=False,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Confusion matrix saved to {output_path}")

    def analyze_failure_modes(
        self,
        dataloader: torch.utils.data.DataLoader,
        top_k_confusions: int = 10,
    ) -> List[Tuple[str, str, int]]:
        """
        Analyze most common failure modes (confusion pairs).

        Args:
            dataloader: DataLoader for test set.
            top_k_confusions: Number of top confusions to return.

        Returns:
            List of (true_class, predicted_class, count) tuples.
        """
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                keypoints = batch["keypoints"].to(self.device)
                targets = batch["label"].to(self.device)
                lengths = batch.get("sequence_length", None)

                logits = self.model(keypoints, lengths)
                predictions = logits.argmax(dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Find confusions
        confusion_pairs = {}
        for true_label, pred_label in zip(all_targets, all_predictions):
            if true_label != pred_label:
                pair = (true_label, pred_label)
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

        # Sort by frequency
        top_confusions = sorted(
            confusion_pairs.items(), key=lambda x: x[1], reverse=True
        )[:top_k_confusions]

        # Convert to class names if available
        confusion_list = []
        for (true_idx, pred_idx), count in top_confusions:
            true_name = (
                self.class_names[true_idx] if self.class_names else str(true_idx)
            )
            pred_name = (
                self.class_names[pred_idx] if self.class_names else str(pred_idx)
            )
            confusion_list.append((true_name, pred_name, count))

        return confusion_list


def compute_metrics_for_mlflow(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute metrics suitable for MLflow logging.

    Args:
        model: Trained model.
        dataloader: DataLoader for evaluation.
        device: Device to run on.

    Returns:
        Dictionary of metrics.
    """
    metrics_calculator = SignRecognitionMetrics(model, device)
    return metrics_calculator.evaluate(
        dataloader, metrics=["top1", "top5", "latency", "per_signer"]
    )
