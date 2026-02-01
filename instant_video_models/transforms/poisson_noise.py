"""Defines the Poisson noise transform."""

import torch
from torchvision.transforms.v2.functional import convert_image_dtype

from instant_video_models.transforms.base import FrameTransform


class PoissonNoise(FrameTransform):
    """The Poisson noise transform."""

    def __init__(self, scale: float = 64.0) -> None:
        """Creates the transform.

        Args:
            scale: The event count corresponding to the maximum possible image value
                (e.g., 255 for a uint8 image or 1.0 for a float32 image). Defaults to
                64.0.
        """
        super().__init__()
        self.scale = scale

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        original_dtype = frame.dtype
        frame = convert_image_dtype(frame, torch.float32)
        frame = torch.poisson(self.scale * frame) / self.scale
        frame.clip_(min=0.0, max=1.0)
        return convert_image_dtype(frame, original_dtype)
