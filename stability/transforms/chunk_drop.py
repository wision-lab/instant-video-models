import torch

from stability.transforms.base import FrameTransform


class ChunkDrop(FrameTransform):
    def __init__(self, chunk_size: int = 8, drop_ratio: float = 0.1) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.drop_ratio = drop_ratio

    # noinspection PyUnresolvedReferences
    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        h, w = frame.shape[-2:]
        mask_shape = (1, h // self.chunk_size + 1, w // self.chunk_size + 1)
        mask = torch.rand(mask_shape) > self.drop_ratio
        mask = mask.repeat_interleave(self.chunk_size, dim=-2)
        mask = mask.repeat_interleave(self.chunk_size, dim=-1)
        return frame * mask[:, :h, :w]
