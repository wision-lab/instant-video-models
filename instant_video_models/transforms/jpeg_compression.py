"""Defines the JPEG compression transform.

This transform compresses and un-compresses an image to generate JPEG artifacts.
"""

from io import BytesIO

import torch
from PIL import Image
from torchvision.transforms.v2.functional import (
    convert_image_dtype,
    to_pil_image,
    to_tensor,
)

from instant_video_models.transforms.base import FrameTransform


class JPEGCompression(FrameTransform):
    """The JPEG compression transform.

    This transform compresses and un-compresses an image to generate JPEG artifacts.
    """

    def __init__(self, quality: int = 10) -> None:
        """Creates the transform.

        Args:
            quality: The JPEG quality value, in the range 0-100. Defaults to 10.
        """
        super().__init__()
        self.quality = quality

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        original_dtype = frame.dtype
        frame = convert_image_dtype(frame, torch.uint8)
        frame = to_pil_image(frame)
        output = BytesIO()
        frame.save(output, "JPEG", quality=self.quality)
        output.seek(0)
        frame = Image.open(output)
        frame = to_tensor(frame)
        return convert_image_dtype(frame, original_dtype)
