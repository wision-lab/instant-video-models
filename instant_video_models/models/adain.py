"""Defines the AdaIN stylization model.

The main code lives in the extern/adain submodule.
"""

import logging
import sys
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torchvision.io import read_image
from torchvision.transforms.v2.functional import convert_image_dtype, resize

# Force the adain directory onto the Python path
sys.path.append(str(Path(__file__).parents[2] / "extern" / "adain"))

from function import adaptive_instance_normalization
from net import decoder, vgg

VGG_LAYERS = 31

logger = logging.getLogger(__name__)


class AdaIN(nn.Module):
    """The AdaIN stylization model.

    The main code lives in the extern/adain submodule.
    """

    def __init__(
        self,
        style_filepath: str | Path,
        style_size: int = None,
        weights_filepath: str | Path = None,
    ) -> None:
        """Initializes the model.

        Args:
            style_filepath: The location of an image to use as a style reference.
            style_size: If this is not None, scale the style image such that this is the
                length of the short size. Resizing the style image will change the
                density of textures in the stylized outputs. Defaults to None.
            weights_filepath: If this is not None, load weights from this location.
                Defaults to None.
        """
        super().__init__()
        self.decoder = deepcopy(decoder)

        # Use separate submodules for vgg_style and vgg_content, even though they have
        # the same architecture and weights. This allows us to apply stabilization and
        # monitoring to each branch independently.
        vgg_layers = list(vgg.children())[:VGG_LAYERS]
        self.vgg_style = nn.Sequential(*deepcopy(vgg_layers))
        self.vgg_content = nn.Sequential(*deepcopy(vgg_layers))

        self.style_image = convert_image_dtype(
            read_image(str(style_filepath)), torch.float32
        )
        logger.info(f"Loaded style from {style_filepath}.")

        # Short-edge resize
        if style_size is not None:
            self.style_image = resize(self.style_image, style_size)

        if weights_filepath is not None:
            self.load_state_dict(torch.load(weights_filepath))
            logger.info(f"Loaded weights from {weights_filepath}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stylizes the input image.

        Args:
            x: The content image, assumed to have shape (b, c, h, w).

        Returns:
            A stylized image of shape (b, c, h, w).
        """
        original_dtype = x.dtype
        x = convert_image_dtype(x, torch.float32)
        style_image = self.style_image.unsqueeze(dim=0).to(x.device)
        feat = adaptive_instance_normalization(
            self.vgg_content(x), self.vgg_style(style_image)
        )
        result = self.decoder(feat).clip(min=0.0, max=1.0)
        return convert_image_dtype(result, original_dtype)
