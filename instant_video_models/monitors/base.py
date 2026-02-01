"""Defines the base class for monitor hooks."""

from abc import ABC, abstractmethod

import torch

from instant_video_models.hooks import HookableModule


class Monitor(HookableModule, ABC):
    """The base class for monitor hooks."""

    @abstractmethod
    def available(self) -> bool:
        """Determines whether an output for this module is available.

        This method should be called before retrieve().

        Returns:
            True if an output is available, False otherwise.
        """

    def reset(self) -> None:
        """Clears internal short-term state.

        Intended to be called between sequences or videos. The specific logic is defined
        by the child class.
        """

    @abstractmethod
    def retrieve(self) -> torch.Tensor | list:
        """Retrieves the stored output of this module.

        The user should first call available() to check if an output is available.

        Returns:
            The stored output of this module.
        """
