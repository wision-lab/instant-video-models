import torch
from torchvision.transforms.v2.functional import convert_image_dtype

from stability.transforms.base import FrameTransform


class PoissonNoise(FrameTransform):
    """
    The input is converted to the float32 image type before the noise is applied. The
    final output is converted back to the original image data type.
    """

    def __init__(self, scale: float = 64.0) -> None:
        super().__init__()
        self.scale = scale

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        original_dtype = frame.dtype
        frame = convert_image_dtype(frame, torch.float32)
        frame = torch.poisson(self.scale * frame) / self.scale
        frame.clip_(min=0.0, max=1.0)
        return convert_image_dtype(frame, original_dtype)
