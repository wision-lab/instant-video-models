import torch
from torch import nn


# Cannot inherit from FrameTransform because it may transform the frame shapes
class RotateLandscape(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] > x.shape[-1]:
            return x.transpose(-1, -2)
        return x
