"""Defines the NAFNet denoising model.

The main code lives in the extern/nafnet submodule.
"""

import logging
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision.transforms.v2.functional import convert_image_dtype

NAFNET_DIR = Path(__file__).parents[2] / "extern" / "nafnet"

# Force the nafnet directory onto the Python path
sys.path.append(str(NAFNET_DIR))

from basicsr.models import create_model
from basicsr.utils import options

logger = logging.getLogger(__name__)

# See extern/nafnet/basicsr/demo.py for example usage


class NAFNet(nn.Module):
    """The NAFNet denoising model.

    The main code lives in the extern/nafnet submodule.
    """

    def __init__(self, weights_filepath: str | Path = None) -> None:
        """Initializes the model.

        Args:
            weights_filepath: If this is not None, load weights from this location.
                Defaults to None.
        """
        super().__init__()
        denoise_opt = options.parse(
            str(NAFNET_DIR / "options" / "test" / "SIDD" / "NAFNet-width32.yml"),
            is_train=False,
        )

        # The original implementation sets dist to False
        denoise_opt["dist"] = False

        # Don't load weights when creating the model
        denoise_opt["path"]["pretrain_network_g"] = None

        self.model = create_model(denoise_opt)

        # Track the PyTorch module to allow submodule access
        self.module = self.model.net_g

        if weights_filepath is not None:
            self.load_state_dict(torch.load(weights_filepath))
            logger.info(f"Loaded weights from {weights_filepath}.")

        self.clip_warning_done = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoises the input image.

        Args:
            x: The input image, assumed to have shape (b, c, h, w).

        Returns:
            A denoised image with the same shape as the input.
        """
        original_dtype = x.dtype
        x = convert_image_dtype(x, torch.float32)

        # Take care of padding here (instead of letting self.module do it) so the aspect
        # ratio stays exactly uniform throughout self.module. This prevents controller
        # backbone misalignment due to padding-induced offsets.
        h, w = x.shape[-2:]
        x = self.module.check_image_size(x)
        result = self.module(x)[..., :h, :w]

        # If the learning rate is set too high, clipping the output to 0-1 can lead to
        # an irrecoverable state where all outputs are <0 or >1 and thus all gradients
        # are zero. To prevent this, do not clip outputs during training.
        if not torch.is_grad_enabled():
            result = result.clip(min=0.0, max=1.0)
        elif (not self.training) and (not self.clip_warning_done):
            logger.warning(
                "This module is in evaluation mode. However, because gradients are "
                "enabled, the output will *not* be clipped to the range 0-1. This may "
                "lead to out-of-bounds results."
            )
            self.clip_warning_done = True

        return convert_image_dtype(result, original_dtype)
