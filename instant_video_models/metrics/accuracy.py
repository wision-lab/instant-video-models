"""Defines the accuracy metric."""

import torch

from instant_video_models.metrics.base import MeanFrameMetric


class Accuracy(MeanFrameMetric):
    """The accuracy metric."""

    def __init__(
        self,
        assume_batch_dim: bool = True,
        class_dim: int = -3,
        mask_nan: bool = False,
    ) -> None:
        """Initializes the metric.

        Args:
            assume_batch_dim: Whether to assume the first axis is a batch dimension. If
                True, inputs are separated along the batch dimension and passed
                independently to the core update logic. Defaults to True.
            class_dim: The dimension along which argmax should be applied to determine
                category predictions. Defaults to 0.
            mask_nan: Whether to ignore nan elements when computing the metric. Defaults
                to False.
        """
        super().__init__(assume_batch_dim, mask_nan)
        self.class_dim = class_dim

    def compute_metric(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Computes the accuracy metric for a single item.

        Args:
            pred: The predicted class probabilities or logits.
            true: The ground-truth class indices.

        Returns:
            The fraction of elements whose predicted category matches the true category.
        """
        return pred.argmax(dim=self.class_dim).eq(true).float().mean()
