"""Defines the image detouch transform.

This transform reduces brightness, contrast, and saturation.
"""

import torch
from torchvision.transforms import v2 as tf

from instant_video_models.transforms.base import FrameTransform


class ImageDetouch(FrameTransform):
    """The image detouch transform.

    This transform reduces brightness, contrast, and saturation.
    """

    def __init__(
        self, brightness: float = 0.7, contrast: float = 0.7, saturation: float = 0.7
    ) -> None:
        """Creates the transform.

        Args:
            brightness: The factor for brightness adjustment. Defaults to 0.7.
            contrast: The factor for contrast adjustment. Defaults to 0.7.
            saturation: The factor for saturation adjustment. Defaults to 0.7.
        """
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        frame = tf.functional.adjust_brightness(frame, self.brightness)
        frame = tf.functional.adjust_contrast(frame, self.contrast)
        frame = tf.functional.adjust_saturation(frame, self.saturation)
        return frame
