"""Defines a stabilizer that performs temporal Gaussian smoothing."""

import math
from collections import deque

import torch

from instant_video_models.stabilizers.base import Stabilizer


class GaussianStabilizer(Stabilizer):
    """A stabilizer that performs temporal Gaussian smoothing."""

    def __init__(
        self, hook_target: str = "output", sigma: float = 1.0, window_size: int = 4
    ) -> None:
        """Creates the stabilizer.

        Args:
            hook_target: Whether to apply this hook to the parent's input or output. Can
                be "inputs" or "output". Defaults to "inputs".
            sigma: The temporal size of the smoothing kernel. Defaults to 1.0.
            window_size: The size of the temporal window to consider when smoothing.
                This should be at least ~3 times larger than sigma. Defaults to 4.
        """
        super().__init__(hook_target=hook_target)
        self.sigma = sigma
        self.window_size = window_size

        # Input history queue
        self.history = deque()

    def reset(self) -> None:
        """Clears internal short-term state.

        Specifically, clears the input history queue.
        """
        self.history.clear()

    def stabilize(self, z: torch.Tensor) -> torch.Tensor:
        # Update the input history queue
        self.history.append(z.clone())
        if len(self.history) > self.window_size:
            self.history.popleft()

        # Gaussian stabilization
        gamma = torch.empty(
            (len(self.history),) + z.shape, dtype=z.dtype, device=z.device
        )
        for s, _ in enumerate(self.history):
            dt = len(self.history) - s - 1
            gamma[s] = math.exp(-(dt**2) / (2.0 * self.sigma**2))
        gamma /= gamma.sum(dim=0)
        return sum(z_s * gamma_s for z_s, gamma_s in zip(self.history, gamma))
