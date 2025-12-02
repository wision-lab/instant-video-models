"""Defines the DeepLabv3+ segmentation model.

The main code lives in the extern/deeplab submodule.
"""

import logging
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision.transforms.v2 import Normalize
from torchvision.transforms.v2.functional import convert_image_dtype

# Force the deeplab directory onto the Python path
sys.path.append(str(Path(__file__).parents[2] / "extern" / "deeplab"))

import network

logger = logging.getLogger(__name__)


class Deeplab(nn.Module):
    """The DeepLabv3+ segmentation model.

    The main code lives in the extern/deeplab submodule.
    """

    def __init__(
        self,
        adapted_classes: int = None,
        model_name: str = "deeplabv3plus_mobilenet",
        native_classes: int = 19,
        normalize_mean: tuple = (0.485, 0.456, 0.406),
        normalize_std: tuple = (0.229, 0.224, 0.225),
        output_stride: int = 16,
        weights_filepath: str | Path = None,
    ) -> None:
        """Initializes the model.

        Args:
            adapted_classes: If this is not None, add a final class adapter (1x1
                convolution) layer that outputs the specified number of classes.
                Defaults to None.
            model_name: The model variant (see extern/deeplab for details). Defaults to
                "deeplabv3plus_mobilenet".
            native_classes: The number of classes in the original model. Defaults to 19.
            normalize_mean: The mean for the pre-model normalization. Defaults to
                (0.485, 0.456, 0.406).
            normalize_std: The standard deviation for the pre-model normalization.
                Defaults to (0.229, 0.224, 0.225).
            output_stride: The output stride (see extern/deeplab for details). Defaults
                to 16.
            weights_filepath: If this is not None, load weights from this location.
                Defaults to None.
        """
        super().__init__()
        self.model = network.modeling.__dict__[model_name](
            num_classes=native_classes, output_stride=output_stride
        )
        self.normalize = Normalize(mean=normalize_mean, std=normalize_std)

        if adapted_classes is None:
            self.class_adapter = None
        else:
            self.class_adapter = nn.LazyConv2d(
                out_channels=adapted_classes, kernel_size=1
            )
        if weights_filepath is not None:
            self.load_state_dict(
                torch.load(weights_filepath), strict=(adapted_classes is None)
            )
            logger.info(f"Loaded weights from {weights_filepath}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes segmentation logits from the input image.

        Args:
            x: The image to be segmented, assumed to have shape (b, c, h, w).

        Returns:
            A logit tensor of shape (b, n_classes, h, w).
        """
        x = convert_image_dtype(x, torch.float32)

        # Preprocessing
        x = self.normalize(x)

        # Inference
        logits = self.model(x)

        # Optional class adaption as the last layer
        if self.class_adapter is not None:
            logits = self.class_adapter(logits)

        return logits
