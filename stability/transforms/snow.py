import random

import math
import torch
from PIL import Image, ImageDraw, ImageFilter
from torchvision.transforms.v2.functional import (
    convert_image_dtype,
    to_pil_image,
    to_tensor,
)

from stability.transforms.base import FrameTransform


class Snow(FrameTransform):
    def __init__(
        self,
        min_density: float = 5e-5,
        max_density: float = 1e-4,
        min_size: float = 5.0,
        max_size: float = 10.0,
        min_rotation: float = 0.0,
        max_rotation: float = 180.0,
        min_transparency: float = 0.5,
        max_transparency: float = 1.0,
        min_blur: float = 1.0,
        max_blur: float = 4.0,
    ) -> None:
        super().__init__()
        self.min_density = min_density
        self.max_density = max_density
        self.min_size = min_size
        self.max_size = max_size
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        self.min_transparency = min_transparency
        self.max_transparency = max_transparency
        self.min_blur = min_blur
        self.max_blur = max_blur

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        height, width = frame.shape[-2:]
        original_dtype = frame.dtype
        frame = convert_image_dtype(frame, torch.uint8)
        frame = to_pil_image(frame)

        size = height * width
        min_count = round(self.min_density * size)
        max_count = round(self.max_density * size)
        n_snowflakes = random.randint(min_count, max_count)
        for i in range(n_snowflakes):
            axis_1 = random.uniform(self.min_size, self.max_size)
            axis_2 = random.uniform(self.min_size, self.max_size)
            rotation = random.uniform(self.min_rotation, self.max_rotation)
            transparency = random.uniform(self.min_transparency, self.max_transparency)
            blur = random.uniform(self.min_blur, self.max_blur)
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            frame = draw_ellipse(
                frame, axis_1, axis_2, rotation, transparency, blur, x, y
            )

        frame = to_tensor(frame)
        return convert_image_dtype(frame, original_dtype)


def draw_ellipse(
    frame,
    axis_1: float,
    axis_2: float,
    rotation: float,
    transparency: float,
    blur: float,
    x: float,
    y: float,
) -> torch.Tensor:
    # Calculate minimal required size for the layer
    max_axis = max(axis_1, axis_2)
    layer_size = math.ceil(max_axis + 4 * blur)  # Add some padding for blur
    ellipse_layer = Image.new("RGBA", (layer_size, layer_size), (255, 255, 255, 0))

    # Adjust coordinates for the smaller layer
    left = layer_size / 2 - axis_1 / 2
    top = layer_size / 2 - axis_2 / 2
    right = left + axis_1
    bottom = top + axis_2

    # Draw the ellipse
    draw = ImageDraw.Draw(ellipse_layer)
    transparency = round(255 * transparency)
    draw.ellipse([left, top, right, bottom], fill=(255, 255, 255, transparency))

    # Apply rotation and blur
    if rotation != 0:
        ellipse_layer = ellipse_layer.rotate(
            rotation, expand=True, fillcolor=(255, 255, 255, 0)
        )
    if blur > 0:
        ellipse_layer = ellipse_layer.filter(ImageFilter.GaussianBlur(radius=blur))

    # Paste the small layer onto the frame at the correct position
    position = (round(x - layer_size // 2), round(y - layer_size // 2))
    frame.paste(ellipse_layer, position, ellipse_layer)

    return frame
