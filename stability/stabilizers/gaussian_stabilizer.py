import math
from collections import deque

import torch

from stability.stabilizers.base import Stabilizer


class GaussianStabilizer(Stabilizer):
    def __init__(
        self, hook_target: str = "output", sigma: float = 1.0, window_size: int = 4
    ) -> None:
        super().__init__(hook_target=hook_target)
        self.sigma = sigma
        self.window_size = window_size

        # Input history queue
        self.history = deque()

    def reset(self) -> None:
        super().reset()
        self.history.clear()

    def stabilize(self, z: torch.Tensor) -> torch.Tensor:
        # Detach before updating the history queue so we retain grads on this step.
        if self.detach_memory:
            for tensor in self.history:
                tensor.detach_()

        # Maintain an input history queue.
        self.history.append(z.clone())
        if len(self.history) > self.window_size:
            self.history.popleft()

        # Apply stabilization.
        gamma = torch.empty(
            (len(self.history),) + z.shape, dtype=z.dtype, device=z.device
        )
        for s, z_s in enumerate(self.history):
            dt = len(self.history) - s - 1
            gamma[s] = math.exp(-(dt**2) / (2.0 * self.sigma**2))
        gamma /= gamma.sum(dim=0)
        # noinspection PyTypeChecker
        return sum(z_s * gamma_s for z_s, gamma_s in zip(self.history, gamma))
