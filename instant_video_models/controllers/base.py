"""Defines the base class for stabilization controllers and backbones."""

from abc import ABC, abstractmethod

import torch

from instant_video_models.hooks import HookableModule


class Controller(HookableModule, ABC):
    """The base class for stabilization controllers and backbones."""

    def __init__(self, hook_target: str = "inputs") -> None:
        """Creates the controller or backbone.

        Args:
            hook_target: Whether to apply this hook to the parent's input or output. Can
                be "inputs" or "output". Defaults to "inputs".
        """
        super().__init__(hook_target=hook_target)
        self.enabled = True

    @abstractmethod
    def available(self) -> bool:
        """Determines whether an output for this module is available.

        This method should be called before retrieve().

        Returns:
            True if an output is available, False otherwise.
        """

    def disable(self) -> None:
        """Disables this module.

        If disabled, update() will not be called during the forward pass (this module
        will do nothing).
        """
        self.enabled = False

    def enable(self) -> None:
        """Enables this module.

        If enabled, update() will be called during the forward pass.
        """
        self.enabled = True

    def flush(self) -> None:
        """Clears internal long-term state.

        Intended to be called between dataset passes. The specific logic is defined by
        the child class.
        """

    def forward(self, x: torch.Tensor) -> None:
        """A wrapper around update() that first checks whether this module is enabled.

        Args:
            x: The input to this module (either the input or output of the parent).
        """
        if self.enabled:
            self.update(x)

    def reset(self) -> None:
        """Clears internal short-term state.

        Intended to be called between sequences or videos. The specific logic is defined
        by the child class.
        """

    @abstractmethod
    def retrieve(self) -> torch.Tensor:
        """Retrieves the stored output of this module.

        The user should first call available() to check if an output is available.

        Returns:
            The stored output of this module.
        """

    @abstractmethod
    def update(self, x: torch.Tensor) -> None:
        """Updates the state of this module.

        Args:
            x: The input to this module (either the input or output of the parent).
        """
