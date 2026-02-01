"""Defines the base class for stabilizers."""

from abc import ABC, abstractmethod

import torch

from instant_video_models.hooks import HookableModule


class Stabilizer(HookableModule, ABC):
    """The base class for stabilizers."""

    def __init__(self, hook_target: str = "output") -> None:
        """Creates the stabilizer.

        Args:
            hook_target: Whether to apply this hook to the parent's input or output. Can
                be "inputs" or "output". Defaults to "output".
        """
        super().__init__(hook_target=hook_target)
        self.enabled = True

    def disable(self) -> None:
        """Disables this module.

        If disabled, stabilize() will not be called during the forward pass (this module
        will do nothing).
        """
        self.enabled = False

    def enable(self) -> None:
        """Enables this module.

        If enabled, stabilize() will be called during the forward pass.
        """
        self.enabled = True

    def flush(self) -> None:
        """Clears internal long-term state.

        Intended to be called between dataset passes. The specific logic is defined by
        the child class.
        """

    def forward(self, z: torch.Tensor) -> torch.Tensor | None:
        """A wrapper around stabilize() that first checks whether this module is
        enabled.

        Args:
            x: The input to this module (either the input or output of the parent).
        """
        if self.enabled:
            return self.stabilize(z)

    def reset(self) -> None:
        """Clears internal short-term state.

        Intended to be called between sequences or videos. The specific logic is defined
        by the child class.
        """

    @abstractmethod
    def stabilize(self, z: torch.Tensor) -> torch.Tensor:
        """Stabilizes the input tensor.

        Args:
            z: The input to this module (either the input or output of the parent).

        Returns:
            A stabilized version of z (should have the same shape and dtype).
        """
