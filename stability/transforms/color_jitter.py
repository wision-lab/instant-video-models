import torch
from torchvision.transforms import v2 as tf

from stability.transforms.base import FrameTransform


class ColorJitter(FrameTransform):
    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.2,
        hue: float = 0.1,
    ) -> None:
        super().__init__()
        self.frame_transform = tf.ColorJitter(brightness, contrast, saturation, hue)

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        return self.frame_transform(frame)
