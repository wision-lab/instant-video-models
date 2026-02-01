"""Defines the base class for frame-wise transforms."""

from abc import ABC, abstractmethod

import torch
from torchvision.transforms import v2 as tf


class FrameTransform(tf.Transform, ABC):
    """The base class for frame-wise transforms."""

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Applies the transform to one or more frames.

        Args:
            tensor: A tensor of arbitrary shape to which the transform should be
                applied. Leading axes up to -4 (inclusive) are treated as batch
                dimensions. The remaining axes are treated as internal frame dimensions.

        Returns:
            The result of applying the transform to each frame of the input tensor.
        """
        tensor = tensor.clone()
        original_shape = tensor.shape

        # Ensure least 4 dimensions (batch or time, c, h, w)
        while tensor.ndim < 4:
            tensor = tensor.unsqueeze(dim=0)

        # Flatten extra leading dimensions
        if tensor.ndim > 4:
            tensor = tensor.flatten(start_dim=0, end_dim=-4)

        for i in range(tensor.shape[0]):
            tensor[i] = self.forward_frame(tensor[i])
        return tensor.reshape(original_shape)

    @abstractmethod
    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Applies the transform to an individual frame.

        Args:
            frame: A frame tensor of shape (c, h, w).

        Returns:
            The result of applying the transform to the frame.
        """
