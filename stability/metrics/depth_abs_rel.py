"""Defines the absrel depth metric."""

import torch

from stability.metrics.base import MeanFrameMetric


class DepthAbsRel(MeanFrameMetric):
    """The absrel depth metric."""

    def __init__(
        self,
        assume_batch_dim: bool = True,
        mask_nan: bool = True,
        max_pred_depth: float = None,
    ) -> None:
        """Initializes the metric.

        Args:
            assume_batch_dim: Whether to assume the first axis is a batch dimension. If
                True, inputs are separated along the batch dimension and passed
                independently to the core update logic. Defaults to True.
            mask_nan: Whether to ignore nan elements when computing the metric. Defaults
                to False.
            max_pred_depth: The upper limit for predicted depth values. Depths will be
                clipped to this maximum value. Defaults to None.
        """
        # Use stability.utils.AffineAlignPredictions if the disparity is relative
        # Justification for using a least-squares fit:
        # https://gist.github.com/ranftlr/45f4c7ddeb1bbb88d606bc600cab6c8d#file-validate_kitti-py-L67
        # https://github.com/isl-org/MiDaS/issues/129
        # Paper stating that scale and shift invariance is in disparity space:
        # https://arxiv.org/pdf/1907.01341 (Equation 1)
        super().__init__(assume_batch_dim, mask_nan)
        self.max_pred_depth = max_pred_depth

    def compute_metric(
        self, pred_disparity: torch.Tensor, true_disparity: torch.Tensor
    ) -> torch.Tensor:
        """Computes the absrel metric for a single pair of depth maps.

        Args:
            pred_disparity: The predicted disparity (inverse depth).
            true_disparity: The ground-truth disparity (inverse depth).

        Returns:
            The average fractional difference between the true and predicted depths.
        """
        # Example showing that this metric should be computed in depth space:
        # https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py#L27
        # https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py#L214
        # Paper stating that this metric should be computed in depth space:
        # https://proceedings.neurips.cc/paper_files/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf
        # (paragraph above Equation 1 and Section 4.3)
        true_depth = 1.0 / true_disparity
        pred_depth = 1.0 / pred_disparity

        # Example using a depth cap (applied after the relative disparity adjustment):
        # https://gist.github.com/ranftlr/45f4c7ddeb1bbb88d606bc600cab6c8d#file-validate_kitti-py-L63
        if self.max_pred_depth is not None:
            pred_depth = pred_depth.clip(max=self.max_pred_depth)

        return ((pred_depth - true_depth) / true_depth).abs().mean()
