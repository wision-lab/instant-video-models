import torch
from torchvision.transforms import v2 as tf

from stability.transforms.base import FrameTransform


class ImageDetouch(FrameTransform):
    def __init__(
        self, brightness: float = 0.7, contrast: float = 0.7, saturation: float = 0.7
    ) -> None:
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        frame = tf.functional.adjust_brightness(frame, self.brightness)
        frame = tf.functional.adjust_contrast(frame, self.contrast)
        frame = tf.functional.adjust_saturation(frame, self.saturation)
        return frame
