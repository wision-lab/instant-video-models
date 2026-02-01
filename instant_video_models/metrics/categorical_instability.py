"""Defines the categorical instability metric."""

import torch

from instant_video_models.metrics.base import MeanInstability
from instant_video_models.utils import ensure_floating_point


class CategoricalInstability(MeanInstability):
    """The categorical instability metric."""

    def __init__(self, assume_batch_dim: bool = True, class_dim: int = 0) -> None:
        """Initializes the metric.

        The mask_nan option is not supported here because masking flattens the shape,
        and the class_dim argument assumes a particular shape.

        Args:
            assume_batch_dim: Whether to assume the first axis is a batch dimension. If
                True, inputs are separated along the batch dimension and passed
                independently to the core update logic. Defaults to True.
            class_dim: The dimension along which argmax should be applied to determine
                category predictions. Defaults to 0.
        """
        super().__init__(assume_batch_dim)
        self.class_dim = class_dim

    def compute_stability(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Computes the instability for a single item with no batch dimension.

        Args:
            a: Logits or class probabilities for the last frame.
            b: Logits or class probabilities for the current frame.

        Returns:
            The fraction of elements whose predicted category does not match.
        """
        a, b = ensure_floating_point(a, b)
        a = a.argmax(dim=self.class_dim)
        b = b.argmax(dim=self.class_dim)
        return a.eq(b).logical_not().float().mean()
