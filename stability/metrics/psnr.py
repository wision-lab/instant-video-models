"""Defines the PSNR metric."""

import math

import torch
from torchvision.transforms.v2.functional import convert_image_dtype as convert_dtype

from stability.metrics.base import MeanFrameMetric


class PSNR(MeanFrameMetric):
    """The PSNR metric."""

    def __init__(
        self,
        max_value: float,
        assume_batch_dim: bool = True,
        convert_image_dtype: str = None,
        mask_nan: bool = False,
    ) -> None:
        """Initializes the metric.

        Args:
            max_value: The upper limit for values. This will typically be 1.0 for dtype
                float32.
            assume_batch_dim: Whether to assume the first axis is a batch dimension. If
                True, inputs are separated along the batch dimension and passed
                independently to the core update logic. Defaults to True.
            convert_image_dtype: If this is not None, convert the input to the specified
                data type before computing PSNR. This may perform some scaling (1/255 if
                converting uint8 -> float32). Defaults to None.
            mask_nan: Whether to ignore nan elements when computing the metric. Defaults
                to False.
        """
        super().__init__(assume_batch_dim, mask_nan)
        self.max_value = max_value
        self.convert_image_dtype = convert_image_dtype

    def compute_metric(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Computes the PSNR metric for a single item.

        Args:
            pred: The prediction tensor.
            true: The ground-truth tensor.

        Returns:
            The PSNR of the prediction relative to the ground truth.
        """
        if self.convert_image_dtype is not None:
            dtype = getattr(torch, self.convert_image_dtype)
            pred = convert_dtype(pred, dtype)
            true = convert_dtype(true, dtype)
        mse = ((pred - true) ** 2).mean()
        return 20.0 * math.log10(self.max_value) - 10.0 * torch.log10(mse)
