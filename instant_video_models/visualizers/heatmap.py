"""Defines the heatmap visualizer."""

import torch

from instant_video_models.utils import visualize_with_cmap
from instant_video_models.visualizers.base import Visualizer


class HeatmapVisualizer(Visualizer):
    """The heatmap visualizer."""

    def __init__(self, cmap_name: str = "viridis", normalize: bool = True) -> None:
        """Creates the visualizer.

        Args:
            cmap_name: The name of the matplotlib color map to use. Defaults to
                "viridis".
            normalize: If True, apply a global scale and shift to values so they fill
                the range 0-1. Defaults to True.
        """
        self.cmap_name = cmap_name
        self.normalize = normalize

    def __call__(self, frame: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return visualize_with_cmap(
            pred, cmap_name=self.cmap_name, normalize=self.normalize
        )
