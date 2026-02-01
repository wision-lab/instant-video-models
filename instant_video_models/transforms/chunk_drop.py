"""Defines the chunk drop transform.

This transform randomly zeros out spatial patches.
"""

import torch

from instant_video_models.transforms.base import FrameTransform


class ChunkDrop(FrameTransform):
    """The chunk drop transform.

    This transform randomly zeros out spatial patches.
    """

    def __init__(self, chunk_size: int = 8, drop_ratio: float = 0.1) -> None:
        """Creates the transform.

        Args:
            chunk_size: The size of square chunks. Defaults to 8.
            drop_ratio: The fraction of chunks that should be randomly set to zero.
                Defaults to 0.1.
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.drop_ratio = drop_ratio

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        h, w = frame.shape[-2:]
        mask_shape = (1, h // self.chunk_size + 1, w // self.chunk_size + 1)
        mask = torch.rand(mask_shape) > self.drop_ratio
        mask = mask.repeat_interleave(self.chunk_size, dim=-2)
        mask = mask.repeat_interleave(self.chunk_size, dim=-1)
        return frame * mask[:, :h, :w]
