"""Defines the norm instability metric."""

from typing import Callable

import torch

from instant_video_models.metrics.base import MeanInstability
from instant_video_models.utils import ensure_floating_point


class NormInstability(MeanInstability):
    """The norm instability metric."""

    def __init__(
        self,
        assume_batch_dim: bool = True,
        mask_nan: bool = False,
        norm_dims: tuple = None,
        order: int | float = 2,
    ) -> None:
        """Initializes the metric.

        Args:
            assume_batch_dim: Whether to assume the first axis is a batch dimension. If
                True, inputs are separated along the batch dimension and passed
                independently to the core update logic. Defaults to True.
            mask_nan: Whether to ignore nan elements when computing the metric. Defaults
                to False.
            norm_dims: The dimensions along which the norm should be computed. If None,
                flatten all dimensions before computing the norm. Defaults to None.
            order: The order of the norm. Defaults to 2.
        """
        super().__init__(assume_batch_dim, mask_nan)
        self.norm_dims = norm_dims
        self.order = order

    def compute_stability(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Computes the instability for a single item with no batch dimension.

        Args:
            a: The last frame.
            b: The current frame.

        Returns:
            The norm difference between the frames.
        """
        a, b = ensure_floating_point(a, b)
        return torch.linalg.vector_norm(a - b, ord=self.order, dim=self.norm_dims)
