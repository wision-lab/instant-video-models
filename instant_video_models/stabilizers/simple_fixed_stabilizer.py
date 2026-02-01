"""Defines a stabilizer that performs simple weighted fusion."""

import torch

from instant_video_models.stabilizers.base import Stabilizer


class SimpleFixedStabilizer(Stabilizer):
    """A stabilizer that performs simple weighted fusion."""

    def __init__(self, alpha: float = 0.9, hook_target: str = "output") -> None:
        """Creates the stabilizer.

        Args:
            alpha: The weight to assign to the current time step. Defaults to 0.9.
            hook_target: Whether to apply this hook to the parent's input or output. Can
                be "inputs" or "output". Defaults to "inputs".
        """
        super().__init__(hook_target=hook_target)
        self.alpha = alpha

        # Tensor for the last stabilized features
        self.memory = None

    def reset(self) -> None:
        """Clears internal short-term state.

        Specifically, clears the stored tensor for the last stabilized features.
        """
        self.memory = None

    def stabilize(self, z: torch.Tensor) -> torch.Tensor:
        if self.memory is not None:
            z_stabilized = self.alpha * z + (1.0 - self.alpha) * self.memory
        else:
            z_stabilized = z

        # Used on the next time step
        self.memory = z_stabilized.clone()

        return z_stabilized
