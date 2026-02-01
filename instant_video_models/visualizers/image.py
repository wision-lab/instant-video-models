"""Defines the image visualizer.

This visualizer simply returns the prediction as the visualized frame.
"""

import torch

from instant_video_models.visualizers.base import Visualizer


class ImageVisualizer(Visualizer):
    """The image visualizer.

    This visualizer simply returns the prediction as the visualized frame.
    """

    def __call__(self, frame: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return pred
