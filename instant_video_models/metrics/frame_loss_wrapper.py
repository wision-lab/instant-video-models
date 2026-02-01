"""Defines the frame loss wrapper metric.

This metric wraps some other callable in the StreamingMetric interface.
"""

from typing import Callable

import torch

from instant_video_models.metrics.base import MeanFrameMetric


class FrameLossWrapper(MeanFrameMetric):
    """The frame loss wrapper metric.

    This metric wraps some other callable in the StreamingMetric interface.
    """

    def __init__(
        self,
        loss_metric: Callable,
        assume_batch_dim: bool = True,
        mask_nan: bool = True,
        unsqueeze: bool = False,
    ) -> None:
        """Initializes the metric.

        Use caution when setting mask_nan=True. Masking flattens the shape, which may
        violate the expectations of the wrapped metric.

        Args:
            loss_metric: The wrapped callable. This should take one or more tensors as
                input and return a single metric tensor.
            assume_batch_dim: Whether to assume the first axis is a batch dimension. If
                True, inputs are separated along the batch dimension and passed
                independently to the core update logic. Defaults to True.
            mask_nan: Whether to ignore nan elements when computing the metric. Defaults
                to False.
            unsqueeze: Whether to add a dummy batch dimension before invoking the
                wrapped callable. Defaults to False.
        """
        super().__init__(assume_batch_dim, mask_nan)
        self.loss_metric = loss_metric
        self.unsqueeze = unsqueeze

    def compute_metric(self, *args) -> torch.Tensor:
        """Computes the wrapped callable on the input arguments.

        Returns:
            The output of the callable.
        """
        if self.unsqueeze:
            args = [x.unsqueeze(dim=0) for x in args]
        result = self.loss_metric(*args)
        if self.unsqueeze:
            result.squeeze(dim=0)
        return result
