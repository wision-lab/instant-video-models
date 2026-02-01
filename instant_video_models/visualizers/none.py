"""Defines the none visualizer.

This visualizer simply returns the input image as the visualized frame.
"""

import torch

from instant_video_models.visualizers.base import Visualizer


class NoneVisualizer(Visualizer):
    """The none visualizer.

    This visualizer simply returns the input image as the visualized frame.
    """

    def __call__(self, frame: torch.Tensor, pred: tuple | torch.Tensor) -> torch.Tensor:
        return frame
