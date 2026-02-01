"""Defines the unified accuracy-stability-robustness loss."""

from typing import Callable

import torch

from instant_video_models.metrics.base import StreamingMetric


class UnifiedLoss(StreamingMetric):
    """The unified accuracy-stability-robustness loss."""

    def __init__(
        self,
        loss_metric: Callable,
        stability_weight: float,
        mask_nan: bool = True,
        assume_batch_dim: bool = True,
    ) -> None:
        """Initializes the metric.

        Use caution when setting mask_nan=True. Masking flattens the shape, which may
        violate the expectations of the wrapped metric.

        Args:
            loss_metric: The core distance measure (delta).
            stability_weight: The weight (lambda) to assign to the stability term.
            assume_batch_dim: Whether to assume the first axis is a batch dimension. If
                True, inputs are separated along the batch dimension and passed
                independently to the core update logic. Defaults to True.
            mask_nan: Whether to ignore nan elements when computing the metric. Defaults
                to False.
        """
        super().__init__()
        self.loss_metric = loss_metric
        self.mask_nan = mask_nan
        self.stability_weight = stability_weight
        self.assume_batch_dim = assume_batch_dim

        # The number of items (time steps) seen so far
        self.count = 0

        # The total metric value
        self.sum = None

        # The previous inputs to update, required to compute frame differences
        self.memory = None

    def available(self) -> bool:
        """
        Returns:
            True if update() has been called at least once since the last flush().
        """
        return self.sum is not None

    def compute(self) -> float:
        return self.sum / self.count

    def flush(self) -> None:
        """Resets accumulated metric statistics."""
        self.count = 0
        self.sum = None

    def reset(self) -> None:
        """Resets the memory of the previous input."""
        self.memory = None

    def update(self, pred: torch.Tensor, true: torch.Tensor) -> None:
        if not self.assume_batch_dim:
            pred = pred.unsqueeze(dim=0)
            true = true.unsqueeze(dim=0)
        first = self.memory is None
        if first:
            self.memory = torch.empty_like(pred)
        for i, (pred_i, true_i) in enumerate(zip(pred, true)):
            if self.mask_nan:
                mask = ~(pred_i.isnan() | true_i.isnan() | self.memory[i].isnan())
                pred_i_masked = pred_i[mask]
                true_i_masked = true_i[mask]
                memory_i_masked = self.memory[i][mask]
            else:
                pred_i_masked = pred_i
                true_i_masked = true_i
                memory_i_masked = self.memory[i]
            loss = self.loss_metric(pred_i_masked, true_i_masked)
            if not first:
                change = self.loss_metric(pred_i_masked, memory_i_masked)
                loss = loss + self.stability_weight * change
            self.count += 1
            if self.sum is None:
                self.sum = torch.zeros_like(loss)
            self.sum = self.sum + loss
            self.memory[i] = pred_i
