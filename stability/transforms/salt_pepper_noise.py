import torch
from torchvision.transforms.v2.functional import convert_image_dtype

from stability.transforms.base import FrameTransform


class SaltPepperNoise(FrameTransform):
    def __init__(self, salt_amount: float = 0.05, pepper_amount: float = 0.05) -> None:
        super().__init__()
        if salt_amount + pepper_amount > 1.0:
            raise ValueError("salt_amount and pepper_amount cannot sum to more than 1.")
        self.salt_amount = salt_amount
        self.pepper_amount = pepper_amount

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        # Determine the correct salt value based on the frame dtype.
        salt = torch.tensor(1.0, dtype=torch.float32, device=frame.device)
        salt = convert_image_dtype(salt, frame.dtype)

        frame = frame.clone()
        rand = torch.rand(frame.shape)
        frame[rand < self.pepper_amount] = 0
        frame[rand > 1.0 - self.salt_amount] = salt
        return frame
