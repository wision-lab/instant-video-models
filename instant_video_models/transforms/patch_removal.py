"""Defines the patch removal transform.

This transform sets randomly-sized spatial patches to zero.
"""

import torch

from instant_video_models.transforms.base import FrameTransform


class PatchRemoval(FrameTransform):
    """The patch removal transform.

    This transform sets randomly-sized spatial patches to zero.
    """

    def __init__(
        self,
        min_number: int = 1,
        max_number: int = 8,
        min_size: int = 40,
        max_size: int = 120,
    ) -> None:
        """Creates the transform.

        Args:
            min_number: The minimum number of patches. Defaults to 1.
            max_number: The maximum number of patches. Defaults to 8.
            min_size: The minimum size (height or width) of each patch. Defaults to 40.
            max_size: The maximum size (height or width) of each patch. Defaults to 120.
        """
        super().__init__()
        self.min_number = min_number
        self.max_number = max_number
        self.min_size = min_size
        self.max_size = max_size

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        frame = frame.clone()
        for _ in range(
            torch.randint(self.min_number, self.max_number + 1, size=()).item()
        ):
            h = torch.randint(self.min_size, self.max_size + 1, size=())
            w = torch.randint(self.min_size, self.max_size + 1, size=())
            y = torch.randint(0, frame.shape[-2] - h + 1, size=())
            x = torch.randint(0, frame.shape[-1] - w + 1, size=())
            frame[..., y : y + h, x : x + w] = 0
        return frame
