from abc import ABC, abstractmethod

import torch
from torch import nn
from torchvision.transforms import v2 as tf

from stability.config import instantiate


class FrameTransform(tf.Transform, ABC):
    """
    Parent class for frame-wise transforms that implements common logic.
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.clone()
        original_shape = tensor.shape

        # Ensure the tensor has at least 4 dimensions (batch/time, c, h, w).
        while tensor.ndim < 4:
            tensor = tensor.unsqueeze(dim=0)

        # Flatten all leading dimensions into one.
        if tensor.ndim > 4:
            tensor = tensor.flatten(start_dim=0, end_dim=-4)

        for i in range(tensor.shape[0]):
            tensor[i] = self.forward_frame(tensor[i])
        return tensor.reshape(original_shape)

    @abstractmethod
    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        pass


def initialize_transforms(transform_configs: list) -> nn.Module:
    if len(transform_configs) != 0:
        return tf.Compose(instantiate(transform_configs))
    else:
        return nn.Identity()
