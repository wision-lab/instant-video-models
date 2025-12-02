import torch
from PIL import Image
from torchvision.transforms.v2.functional import (
    convert_image_dtype,
    to_pil_image,
    to_tensor,
)

from stability.transforms.base import FrameTransform


class Pixelate(FrameTransform):
    def __init__(self, quality: float = 0.25) -> None:
        super().__init__()
        self.quality = quality

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        height, width = frame.shape[-2:]
        original_dtype = frame.dtype
        frame = convert_image_dtype(frame, torch.uint8)
        frame = to_pil_image(frame)
        frame = frame.resize(
            (int(width * self.quality), int(height * self.quality)), Image.BOX
        )
        frame = frame.resize((width, height), Image.BOX)
        frame = to_tensor(frame)
        return convert_image_dtype(frame, original_dtype)
