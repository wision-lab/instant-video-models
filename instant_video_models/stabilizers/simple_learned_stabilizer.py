"""Defines a stabilizer with a learned per-channel decay."""

import torch
from torch import nn

from instant_video_models.stabilizers.base import Stabilizer


class SimpleLearnedStabilizer(Stabilizer):
    """A stabilizer with a learned per-channel decay."""

    def __init__(
        self,
        channel_axis: int = 1,
        hook_target: str = "output",
        logits_init: float = 4.0,
    ) -> None:
        """Creates the stabilizer.

        Args:
            channel_axis: The axis to assume as the channel axis. Defaults to 1.
            hook_target: Whether to apply this hook to the parent's input or output. Can
                be "inputs" or "output". Defaults to "inputs".
            logits_init: The value to which logits should be initialized. Logits are
                passed through a sigmoid to compute decay. Defaults to 4.0.
        """
        super().__init__(hook_target=hook_target)
        self.channel_axis = channel_axis
        self.logits_init = logits_init

        # The number of channels is not known before we see the first input
        self.logits = nn.UninitializedParameter(dtype=torch.float32)

        # Tensor for the last stabilized features
        self.memory = None

    def reset(self) -> None:
        """Clears internal short-term state.

        Specifically, clears the stored tensor for the last stabilized features.
        """
        self.memory = None

    def stabilize(self, z: torch.Tensor) -> torch.Tensor:
        # Parameter materialization and initialization
        if torch.nn.parameter.is_lazy(self.logits):
            logits_shape = [1] * z.ndim
            logits_shape[self.channel_axis] = z.shape[self.channel_axis]
            self.logits.materialize(logits_shape)
            with torch.no_grad():
                self.logits.copy_(torch.full_like(self.logits, self.logits_init))

        if self.memory is not None:
            alpha = self.logits.sigmoid()
            z_stabilized = alpha * z + (1.0 - alpha) * self.memory
        else:
            z_stabilized = z

        # Used on the next time step
        self.memory = z_stabilized.clone()

        return z_stabilized
