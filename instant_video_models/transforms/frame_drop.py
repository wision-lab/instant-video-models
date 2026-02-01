"""Defines the frame drop transform.

This transform randomly zeros out frames.
"""

import torch

from instant_video_models.transforms.base import FrameTransform


class FrameDrop(FrameTransform):
    """The frame drop transform.

    This transform randomly zeros out frames.
    """

    def __init__(self, drop_ratio: float = 0.1) -> None:
        """Creates the transform.

        Args:
            drop_ratio: The fraction of frames that should be randomly set to zero.
                Defaults to 0.1.
        """
        super().__init__()
        self.drop_ratio = drop_ratio

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < self.drop_ratio:
            return torch.zeros_like(frame)
        else:
            return frame
