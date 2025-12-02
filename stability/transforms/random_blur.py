import torch
import torchvision.transforms.v2 as tf

from stability.transforms.base import FrameTransform


class RandomBlur(FrameTransform):
    def __init__(
        self, kernel_size: int = 11, min_sigma: float = 0.1, max_sigma: float = 2.0
    ) -> None:
        super().__init__()
        self.frame_transform = tf.GaussianBlur(kernel_size, (min_sigma, max_sigma))

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        return self.frame_transform(frame)
