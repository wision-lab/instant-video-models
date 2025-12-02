"""Defines the mean IOU segmentation metric."""

import torch
from torchmetrics import segmentation

from stability.metrics.base import StreamingMetric


class MeanIOU(StreamingMetric):
    """The mean IOU segmentation metric."""

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
        super().__init__()
        self.assume_batch_dim = assume_batch_dim
        self.class_dim = class_dim

        # The internal torchmetrics MeanIoU object
        self.mean_iou = None

        self.update_called = False

    def available(self) -> bool:
        """Determines whether a value for this metric is available.

        Returns:
            True if update() has been called at least once since the last flush().
        """
        return self.update_called

    def compute(self) -> torch.Tensor:
        return self.mean_iou.compute()

    def flush(self) -> None:
        """Resets accumulated metric statistics."""
        if self.mean_iou is not None:
            self.mean_iou.reset()
        self.update_called = False

    def update(self, pred: torch.Tensor, true: torch.Tensor) -> None:
        # The torchmetrics interface expects a batch dim
        if not self.assume_batch_dim:
            pred = pred.unsqueeze(dim=0)
            true = true.unsqueeze(dim=0)

        # Create the internal metric object if needed
        if self.mean_iou is None:
            # self.class_dim + 1 to skip the batch dimension
            mean_iou = segmentation.MeanIoU(
                num_classes=pred.shape[self.class_dim + 1],
                include_background=True,
                per_class=False,
                input_format="index",
            )
            self.mean_iou = mean_iou.to(pred.device)

        self.mean_iou.update(pred.argmax(dim=self.class_dim + 1), true)
        self.update_called = True
