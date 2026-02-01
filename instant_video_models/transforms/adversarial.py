import torch
import torch.nn as nn
from torchvision.transforms.v2.functional import convert_image_dtype

from instant_video_models.transforms.base import FrameTransform


class IterativeFGSMAttack(FrameTransform):
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        epsilon: float = 0.5,
        iters: int = 5,
    ) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.iters = iters
        self.model.to(device)
        self.model.eval()

    @torch.enable_grad
    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        original_dtype = frame.dtype
        original_device = frame.device
        frame = convert_image_dtype(frame, torch.float32).to(self.device)
        target = torch.randint(2, ())  # Random 0/1 target
        alpha = self.epsilon / self.iters
        perturbed = frame.detach().clone()
        for _ in range(self.iters):
            perturbed.requires_grad = True
            output = self.model(perturbed.unsqueeze(dim=0)).squeeze(dim=0)
            loss = nn.CrossEntropyLoss()(output, target.to(self.device))
            self.model.zero_grad()
            loss.backward()
            perturbed = perturbed + alpha * perturbed.grad.sign()
            perturbed = perturbed.clamp(0.0, 1.0)
            perturbed = perturbed.detach().clone()
        return convert_image_dtype(perturbed, original_dtype).to(original_device)
