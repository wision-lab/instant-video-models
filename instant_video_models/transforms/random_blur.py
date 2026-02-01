"""Defines the random blur transform.

This transform applies a random amount of blur to each frame.
"""

import torch
import torchvision.transforms.v2 as tf

from instant_video_models.transforms.base import FrameTransform


class RandomBlur(FrameTransform):
    """The random blur transform.

    This transform applies a random amount of blur to each frame.
    """

    def __init__(
        self, kernel_size: int = 11, min_sigma: float = 0.1, max_sigma: float = 2.0
    ) -> None:
        """Creates the transform.

        Args:
            kernel_size: The size in pixels of the kernel. Defaults to 11.
            min_sigma: The minimum standard deviation of the blur. Defaults to 0.1.
            max_sigma: The maximum standard deviation of the blur. Defaults to 2.0.
        """
        super().__init__()
        self.frame_transform = tf.GaussianBlur(kernel_size, (min_sigma, max_sigma))

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        return self.frame_transform(frame)
