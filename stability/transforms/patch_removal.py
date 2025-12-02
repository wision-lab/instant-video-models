import torch

from stability.transforms.base import FrameTransform


class PatchRemoval(FrameTransform):
    def __init__(
        self,
        min_number: int = 1,
        max_number: int = 8,
        min_size: int = 40,
        max_size: int = 120,
    ) -> None:
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
