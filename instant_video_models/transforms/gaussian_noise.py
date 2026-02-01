"""Defines the Gaussian noise transform.

The input is converted to the float32 image type with values in [0, 1] before noise is
added. The final output is converted back to the original image data type.
"""

import torch
from torchvision.transforms.v2.functional import convert_image_dtype

from instant_video_models.transforms.base import FrameTransform


class GaussianNoise(FrameTransform):
    """The Gaussian noise transform.

    The input is converted to the float32 image type with values in [0, 1] before noise
    is added. The final output is converted back to the original image data type.
    """

    def __init__(self, sigma: float = 0.2) -> None:
        """Creates the transform.

        Args:
            sigma: The amount of Gaussian noise to add, assuming image values in [0, 1].
                Defaults to 0.2.
        """
        super().__init__()
        self.sigma = sigma

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        original_dtype = frame.dtype
        frame = convert_image_dtype(frame, torch.float32)
        frame = frame + self.sigma * torch.randn_like(frame)
        frame.clip_(min=0.0, max=1.0)
        return convert_image_dtype(frame, original_dtype)
