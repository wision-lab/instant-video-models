from io import BytesIO

import torch
from PIL import Image
from torchvision.transforms.v2.functional import (
    convert_image_dtype,
    to_pil_image,
    to_tensor,
)

from stability.transforms.base import FrameTransform


class JPEGCompression(FrameTransform):
    def __init__(self, quality: int = 10) -> None:
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
