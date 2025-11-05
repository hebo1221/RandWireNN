"""
Uncertainty estimation utilities for Bayesian RandWireNN.

Provides tools for:
- Monte Carlo sampling for uncertainty quantification
- Epistemic and aleatoric uncertainty decomposition
- Uncertainty visualization and analysis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class UncertaintyEstimator:
    """
    Estimate uncertainty using Monte Carlo sampling.

    Supports both MC Dropout and variational inference approaches.
    """

    def __init__(self, model: nn.Module, num_samples: int = 30, device: str = 'cuda'):
        """
        Args:
            model: Bayesian neural network
            num_samples: Number of MC samples for uncertainty estimation
            device: Device to run inference on
        """
        self.model = model
        self.num_samples = num_samples
        self.device = device

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.

        Args:
            x: Input tensor [batch_size, ...]

        Returns:
            mean_pred: Mean predictions [batch_size, num_classes]
            epistemic_uncertainty: Model uncertainty [batch_size, num_classes]
            total_uncertainty: Total uncertainty [batch_size, num_classes]
        """
        self.model.train()  # Enable dropout/sampling
        predictions = []

        for _ in range(self.num_samples):
            pred = self.model(x)
            predictions.append(pred)

        predictions = torch.stack(predictions)  # [num_samples, batch_size, num_classes]

        # Compute statistics
        mean_pred = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.var(dim=0)  # Model uncertainty

        # For classification, compute predictive entropy
        probs = torch.softmax(predictions, dim=-1)
        mean_probs = probs.mean(dim=0)
        total_uncertainty = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)

        return mean_pred, epistemic_uncertainty, total_uncertainty

    @torch.no_grad()
    def estimate_uncertainty_batch(self, dataloader) -> dict:
        """
        Estimate uncertainty for entire dataset.

        Args:
            dataloader: DataLoader for the dataset

        Returns:
            Dictionary containing uncertainty statistics
        """
        all_mean_preds = []
        all_epistemic = []
        all_total = []
        all_targets = []

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)

            mean_pred, epistemic, total = self.predict_with_uncertainty(inputs)

            all_mean_preds.append(mean_pred.cpu())
            all_epistemic.append(epistemic.cpu())
            all_total.append(total.cpu())
            all_targets.append(targets)

        return {
            'predictions': torch.cat(all_mean_preds),
            'epistemic_uncertainty': torch.cat(all_epistemic),
            'total_uncertainty': torch.cat(all_total),
            'targets': torch.cat(all_targets)
        }

    def get_confident_predictions(self, x: torch.Tensor, threshold: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions that meet confidence threshold.

        Args:
            x: Input tensor
            threshold: Confidence threshold (0-1)

        Returns:
            predictions: Class predictions for confident samples
            mask: Boolean mask of confident samples
        """
        mean_pred, _, total_uncertainty = self.predict_with_uncertainty(x)

        # Convert uncertainty to confidence (lower uncertainty = higher confidence)
        # Normalize by maximum possible entropy
        num_classes = mean_pred.size(-1)
        max_entropy = -np.log(1.0 / num_classes)
        confidence = 1 - (total_uncertainty / max_entropy)

        mask = confidence > threshold
        predictions = torch.argmax(mean_pred, dim=-1)

        return predictions, mask


class CalibrationMetrics:
    """
    Compute calibration metrics for uncertainty estimates.

    Evaluates how well predicted uncertainties match actual errors.
    """

    @staticmethod
    def expected_calibration_error(confidences: np.ndarray, accuracies: np.ndarray,
                                  num_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Args:
            confidences: Predicted confidences [num_samples]
            accuracies: Actual accuracies (0 or 1) [num_samples]
            num_bins: Number of bins for calibration plot

        Returns:
            ECE score (lower is better)
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    @staticmethod
    def reliability_diagram_data(confidences: np.ndarray, accuracies: np.ndarray,
                                num_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for reliability diagram (calibration plot).

        Args:
            confidences: Predicted confidences
            accuracies: Actual accuracies
            num_bins: Number of bins

        Returns:
            bin_centers: Center of each bin
            bin_accuracies: Actual accuracy in each bin
            bin_counts: Number of samples in each bin
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            count = np.sum(in_bin)

            if count > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(np.mean(accuracies[in_bin]))
                bin_counts.append(count)

        return np.array(bin_centers), np.array(bin_accuracies), np.array(bin_counts)


def analyze_uncertainty(results: dict, save_path: Optional[str] = None) -> dict:
    """
    Analyze uncertainty estimation results.

    Args:
        results: Dictionary from UncertaintyEstimator.estimate_uncertainty_batch
        save_path: Optional path to save analysis plots

    Returns:
        Dictionary of analysis metrics
    """
    predictions = results['predictions']
    epistemic = results['epistemic_uncertainty']
    total = results['total_uncertainty']
    targets = results['targets']

    # Get predicted classes
    pred_classes = torch.argmax(predictions, dim=-1)
    correct = (pred_classes == targets).numpy()

    # Compute confidence from total uncertainty
    num_classes = predictions.size(-1)
    max_entropy = -np.log(1.0 / num_classes)
    confidence = 1 - (total.numpy() / max_entropy)

    # Calibration metrics
    ece = CalibrationMetrics.expected_calibration_error(confidence, correct)

    # Uncertainty statistics
    analysis = {
        'accuracy': correct.mean(),
        'expected_calibration_error': ece,
        'mean_epistemic_uncertainty': epistemic.mean().item(),
        'mean_total_uncertainty': total.mean().item(),
        'mean_confidence': confidence.mean(),
        'correct_samples_mean_confidence': confidence[correct].mean() if correct.sum() > 0 else 0,
        'incorrect_samples_mean_confidence': confidence[~correct].mean() if (~correct).sum() > 0 else 0,
    }

    logger.info(f"Uncertainty Analysis:")
    logger.info(f"  Accuracy: {analysis['accuracy']:.4f}")
    logger.info(f"  ECE: {analysis['expected_calibration_error']:.4f}")
    logger.info(f"  Mean Epistemic Uncertainty: {analysis['mean_epistemic_uncertainty']:.4f}")
    logger.info(f"  Correct samples confidence: {analysis['correct_samples_mean_confidence']:.4f}")
    logger.info(f"  Incorrect samples confidence: {analysis['incorrect_samples_mean_confidence']:.4f}")

    # Visualization
    if save_path:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Reliability diagram
            bin_centers, bin_accs, bin_counts = CalibrationMetrics.reliability_diagram_data(confidence, correct)
            axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            axes[0].scatter(bin_centers, bin_accs, s=bin_counts*5, alpha=0.6, label='Model')
            axes[0].set_xlabel('Confidence')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title(f'Reliability Diagram (ECE={ece:.4f})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Uncertainty distribution
            axes[1].hist(total.numpy(), bins=50, alpha=0.7, label='All samples')
            axes[1].hist(total.numpy()[correct], bins=50, alpha=0.7, label='Correct')
            axes[1].hist(total.numpy()[~correct], bins=50, alpha=0.7, label='Incorrect')
            axes[1].set_xlabel('Total Uncertainty')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Uncertainty Distribution')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Uncertainty analysis plot saved to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to create plots: {e}")

    return analysis
