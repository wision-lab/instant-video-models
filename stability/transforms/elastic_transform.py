import torch
import torchvision.transforms.v2 as tf

from stability.transforms.base import FrameTransform


class ElasticTransform(FrameTransform):
    def __init__(self, magnitude: float = 50.0, smoothness: float = 5.0) -> None:
        super().__init__()
        self.frame_transform = tf.ElasticTransform(alpha=magnitude, sigma=smoothness)

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        return self.frame_transform(frame)
