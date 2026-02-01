"""Defines the landscape rotation transform.

Landscape here means shape[-2] < shape[-1].
"""

import torch
from torch import nn


# Cannot inherit from FrameTransform because shapes may change
class RotateLandscape(nn.Module):
    """The landscape rotation transform.

    Landscape here means shape[-2] < shape[-1].
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Applies the transform to one or more frames.

        Args:
            tensor: A tensor with >= 2 axes to which the transform should be applied.

        Returns:
            The input tensor with the last two axes possibly transposed.
        """
        if tensor.shape[-2] > tensor.shape[-1]:
            return tensor.transpose(-1, -2)
        return tensor
