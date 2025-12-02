import torch

from stability.stabilizers.base import Stabilizer


class EMAStabilizer(Stabilizer):
    def __init__(self, alpha: float = 0.9, hook_target: str = "output") -> None:
        super().__init__(hook_target=hook_target)
        self.alpha = alpha

        # Reference to the last stabilized feature tensor
        self.memory = None

    def reset(self) -> None:
        super().reset()
        self.memory = None

    def stabilize(self, z: torch.Tensor) -> torch.Tensor:
        # Apply stabilization.
        if self.memory is not None:
            z_stabilized = self.alpha * z + (1.0 - self.alpha) * self.memory
        else:
            z_stabilized = z

        # Save the stabilized features for the next time step.
        self.memory = z_stabilized.clone()

        # Allows preventing backpropagation through time
        if self.detach_memory:
            self.memory.detach_()

        return z_stabilized
