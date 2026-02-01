"""Defines the color jitter transform."""

import torch
from torchvision.transforms import v2 as tf

from instant_video_models.transforms.base import FrameTransform


class ColorJitter(FrameTransform):
    """The color jitter transform."""

    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.2,
        hue: float = 0.1,
    ) -> None:
        """Creates the transform.

        See the documentation of torchvision.transforms.v2.ColorJitter for more detailed
        descriptions of parameters.

        Args:
            brightness: The amount of brightness jitter. Defaults to 0.1.
            contrast: The amount of contrast jitter. Defaults to 0.1.
            saturation: The amount of saturation jitter. Defaults to 0.2.
            hue: The amount of hue jitter. Defaults to 0.1.
        """
        super().__init__()
        self.frame_transform = tf.ColorJitter(brightness, contrast, saturation, hue)

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        return self.frame_transform(frame)
