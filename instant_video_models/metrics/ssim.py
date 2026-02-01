"""Defines the SSIM metric."""

import torch
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchvision.transforms.v2.functional import convert_image_dtype as convert_dtype

from instant_video_models.metrics.base import MeanFrameMetric


class SSIM(MeanFrameMetric):
    """The SSIM metric."""

    def __init__(
        self,
        max_value: float,
        assume_batch_dim: bool = True,
        convert_image_dtype: str = None,
    ) -> None:
        """Initializes the metric.

        The mask_nan option is not supported here because masking flattens the shape,
        and SSIM assumes a particular tensor structure.

        Args:
            max_value: The upper limit for values. This will typically be 1.0 for dtype
                float32.
            assume_batch_dim: Whether to assume the first axis is a batch dimension. If
                True, inputs are separated along the batch dimension and passed
                independently to the core update logic. Defaults to True.
            convert_image_dtype: If this is not None, convert the input to the specified
                data type before computing SSIM. This may perform some scaling (1/255 if
                converting uint8 -> float32). Defaults to None.
        """
        super().__init__(assume_batch_dim)
        self.max_value = max_value
        self.convert_image_dtype = convert_image_dtype

    def compute_metric(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Computes the SSIM metric for a single item.

        Args:
            pred: The prediction image tensor.
            true: The ground-truth image tensor.

        Returns:
            The SSIM of the prediction relative to the ground truth.
        """
        if self.convert_image_dtype is not None:
            dtype = getattr(torch, self.convert_image_dtype)
            pred = convert_dtype(pred, dtype)
            true = convert_dtype(true, dtype)
        while pred.ndim < 4:
            pred = pred.unsqueeze(dim=0)
        while true.ndim < 4:
            true = true.unsqueeze(dim=0)
        return ssim(pred, true, data_range=self.max_value)
