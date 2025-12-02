import torch
from torch import nn

from stability.stabilizers.base import Stabilizer


class LearnedEMAStabilizer(Stabilizer):
    def __init__(
        self,
        channel_axis: int = 1,
        hook_target: str = "output",
        logits_init: float = 4.0,
    ) -> None:
        super().__init__(hook_target=hook_target)

        # The logits must be initialized lazily (we don't know the number of channels
        # until we see the first input).
        self.logits = nn.UninitializedParameter(dtype=torch.float32)

        self.channel_axis = channel_axis
        self.logits_init = logits_init

        # Reference to the last stabilized feature tensor
        self.memory = None

    def reset(self) -> None:
        super().reset()
        self.memory = None

    def stabilize(self, z: torch.Tensor) -> torch.Tensor:
        # Lazy parameter initialization
        if torch.nn.parameter.is_lazy(self.logits):
            logits_shape = [1] * z.ndim
            logits_shape[self.channel_axis] = z.shape[self.channel_axis]
            self.logits.materialize(logits_shape)
            with torch.no_grad():
                self.logits.copy_(torch.full_like(self.logits, self.logits_init))

        # Apply stabilization.
        if self.memory is not None:
            alpha = self.logits.sigmoid()
            z_stabilized = alpha * z + (1.0 - alpha) * self.memory
        else:
            z_stabilized = z

        # Save the stabilized features for the next time step.
        self.memory = z_stabilized.clone()

        # Allows preventing backpropagation through time
        if self.detach_memory:
            self.memory.detach_()

        return z_stabilized
