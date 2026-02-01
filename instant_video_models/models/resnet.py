"""Defines the ResNet classification model."""

import logging
from pathlib import Path

import torch
from torch import nn
from torchvision import models
from torchvision.transforms.v2 import Normalize, Resize
from torchvision.transforms.v2.functional import convert_image_dtype

logger = logging.getLogger(__name__)


class Resnet(nn.Module):
    """The ResNet classification model."""

    def __init__(
        self,
        input_size: tuple = (224, 224),
        n_classes: int = 2,
        normalize_mean: tuple = (0.485, 0.456, 0.406),
        normalize_std: tuple = (0.229, 0.224, 0.225),
        weights_filepath: str | Path = None,
    ) -> None:
        """Initializes the model.

        Args:
            input_size: Resize inputs to this size before passing them to the model.
                Defaults to (224, 224).
            n_classes: The number of possible output categories. Defaults to 2.
            normalize_mean: The mean for the pre-model normalization. Defaults to
                (0.485, 0.456, 0.406).
            normalize_std: The standard deviation for the pre-model normalization.
                Defaults to (0.229, 0.224, 0.225).
            weights_filepath: If this is not None, load weights from this location.
                Defaults to None.
        """
        super().__init__()
        self.input_resize = Resize(input_size, antialias=True)
        self.input_normalize = Normalize(mean=normalize_mean, std=normalize_std)
        self.model = models.resnet50(pretrained=True)

        # Modify the last layer to get the right number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

        if weights_filepath is not None:
            self.load_state_dict(torch.load(weights_filepath))
            logger.info(f"Loaded weights from {weights_filepath}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classifies the input image.

        Args:
            x: The image to be classified, assumed to have shape (b, c, h, w).

        Returns:
            A logit tensor of shape (b, n_classes).
        """
        x = convert_image_dtype(x, torch.float32)
        x = self.input_resize(x)
        x = self.input_normalize(x)
        result = self.model(x)
        return result
