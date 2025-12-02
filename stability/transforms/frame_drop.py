import torch

from stability.transforms.base import FrameTransform


class FrameDrop(FrameTransform):
    def __init__(self, drop_ratio: float = 0.1) -> None:
        super().__init__()
        self.drop_ratio = drop_ratio

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < self.drop_ratio:
            return torch.zeros_like(frame)
        else:
            return frame
