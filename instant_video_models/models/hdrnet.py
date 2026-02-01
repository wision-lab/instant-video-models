"""Defines the HDRNet image enhancement model.

The implementation is based on this TensorFlow code:
https://github.com/WISION-Lab/event-nn-tf/blob/public/models/hdrnet.py
"""

import logging
from pathlib import Path

import torch
from einops import rearrange
from torch import nn
from torchvision.transforms.v2.functional import (
    InterpolationMode,
    convert_image_dtype,
    resize,
)

# Resources:
# https://dl.acm.org/doi/abs/10.1145/3072959.3073592
# https://github.com/mgharbi/hdrnet
# https://github.com/mgharbi/hdrnet/blob/master/hdrnet/layers.py
# https://github.com/mgharbi/hdrnet/blob/master/hdrnet/models.py
# https://groups.csail.mit.edu/graphics/hdrnet/

logger = logging.getLogger(__name__)


class BilateralSlice(nn.Module):
    """The HDRNet bilateral slicing operation."""

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Computes the bilateral slice operation.

        Args:
            x: The output of the coefficient prediction sub-network.
            g: The guidance map.

        Returns:
            A tensor of sliced coefficients.
        """
        # Spatially upsample x to have the same size as the guidance map
        # Use antialias=False to match the TensorFlow default
        x = resize(
            x,
            list(g.shape[-2:]),
            interpolation=InterpolationMode.BILINEAR,
            antialias=False,
        )

        # Isolate the slicing dimension (s)
        x = rearrange(x, "b (c4 c3 s) ... -> b s c3 c4 ...", c3=3, c4=4)

        # Use the guidance map to slice into the channel dimension of x. See Equation 5.
        # The paper describes this as trilinear interpolation. Trillinear interpolation
        # is separable, so we can do bilinear (spatial, done above) -> linear.
        d = x.shape[1]
        k = torch.arange(d, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        w = torch.clip(1.0 - torch.abs(d * g - k), min=0.0)
        w = w.view(w.shape[:2] + (1, 1) + w.shape[2:])
        return torch.sum(x * w, dim=1)


class HDRNet(nn.Module):
    """The HDRNet image enhancement model.

    The implementation is based on this TensorFlow code:
    https://github.com/WISION-Lab/event-nn-tf/blob/public/models/hdrnet.py
    """

    def __init__(
        self,
        internal_size: tuple = (256, 256),
        weights_filepath: str | Path = None,
        width_factor: int = 1,
    ) -> None:
        """Initializes the model.

        Args:
            internal_size: The input size for the coefficient prediction sub-network.
                Defaults to (256, 256).
            weights_filepath: If this is not None, load weights from this location.
                Defaults to None.
            width_factor: A factor by which the number of internal channels should be
                multiplied. Defaults to 1.
        """
        super().__init__()
        self.internal_size = internal_size

        # Low-level feature network
        self.low_level_net = nn.Sequential()
        in_channels = 3
        for channels in 8, 16, 32, 64:
            out_channels = channels * width_factor
            self.low_level_net.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )
            self.low_level_net.append(nn.ReLU())
            in_channels = out_channels

        # Local feature network
        self.local_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
        )

        # Global feature network
        self.global_net = nn.Sequential()
        for i in range(2):
            self.global_net.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
            )
            self.global_net.append(nn.ReLU())
        self.global_net.append(nn.Flatten())
        in_features = in_channels * (internal_size[0] // 64) * (internal_size[1] // 64)
        for features in 256, 128, 64:
            out_features = features * width_factor
            self.global_net.append(nn.Linear(in_features, out_features))
            if features > 64:
                self.global_net.append(nn.ReLU())
            in_features = out_features

        # Local-global fusion network (bilateral coefficient prediction)
        self.fusion_net = nn.Conv2d(in_channels, out_channels=96, kernel_size=1)

        # Pixel-wise guidance network
        self.guidance_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
            nn.ReLU(),
            PiecewiseLinearTransfer(channels=3, n_functions=16),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1),
        )

        # Bilateral coefficient slicing
        self.bilateral_slice = BilateralSlice()

        if weights_filepath is not None:
            self.load_state_dict(torch.load(weights_filepath))
            logger.info(f"Loaded weights from {weights_filepath}.")

        self.clip_warning_done = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs enhancement on the input image.

        Args:
            x: The input image, assumed to have shape (b, c, h, w).

        Returns:
            An enhanced image with the same shape as the input.
        """
        original_dtype = x.dtype
        x = convert_image_dtype(x, torch.float32)
        x_full = x

        # Low-resolution coefficient prediction
        # Use antialias=False to match the TensorFlow default
        # This is just a straight resize without any padding, so we can use x_full (the
        # global input) as the input to the controller backbone without worrying about
        # padding-induced offsets.
        x = resize(
            x,
            list(self.internal_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=False,
        )
        x = self.low_level_net(x)
        x = self.local_net(x) + self.global_net(x).view(x.shape[0], -1, 1, 1)
        x = self.fusion_net(x.clip(min=0.0))

        # Guidance map prediction
        g = self.guidance_net(x_full).clip(min=0.0, max=1.0)

        # Bilateral slicing
        a = self.bilateral_slice(x, g)

        # Affine image transform
        x = (a[:, :, :3] * x_full.unsqueeze(dim=1)).sum(dim=2) + a[:, :, 3]

        # If the learning rate is set too high, clipping the output to 0-1 can lead to
        # an irrecoverable state where all outputs are <0 or >1 and thus all gradients
        # are zero. To prevent this, do not clip outputs during training.
        if not torch.is_grad_enabled():
            x = x.clip(min=0.0, max=1.0)
        elif (not self.training) and (not self.clip_warning_done):
            logger.warning(
                "This module is in evaluation mode. However, because gradients are "
                "enabled, the output will *not* be clipped to the range 0-1. This may "
                "lead to out-of-bounds results."
            )
            self.clip_warning_done = True

        return convert_image_dtype(x, original_dtype)


class PiecewiseLinearTransfer(nn.Module):
    """The HDRNet piecewise linear transfer operation."""

    def __init__(self, channels, n_functions) -> None:
        """Initializes the operation.

        Args:
            channels: The number of image channels.
            n_functions: The number of ReLU functions comprising the transfer.
        """
        super().__init__()
        self.shifts = nn.Parameter(torch.zeros((1, channels, 1, 1, n_functions)))
        self.slopes = nn.Parameter(torch.zeros((1, channels, 1, 1, n_functions)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the piecewise linear transfer operation.

        Args:
            x: An input tensor of shape (b, c, h, w).

        Returns:
            The result of the piecewise linear transfer.
        """
        x = x.unsqueeze(dim=-1)
        x = self.slopes * (x - self.shifts).clip(min=0.0)
        return x.sum(dim=-1)
