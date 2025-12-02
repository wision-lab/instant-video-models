from abc import ABC, abstractmethod

import torch

from stability.utils import HookableModule


class Stabilizer(HookableModule, ABC):
    def __init__(self, hook_target: str = "output") -> None:
        super().__init__(hook_target=hook_target)
        self.enabled = True
        self.counter = 0

    def disable(self) -> None:
        self.enabled = False

    def enable(self) -> None:
        self.enabled = True

    def flush(self) -> None:
        """
        Intended to be called before a dataset iteration. Defined by the child class.
        """

    def forward(self, z: torch.Tensor) -> torch.Tensor | None:
        if self.enabled:
            self.counter += 1
            return self.stabilize(z)

    def reset(self) -> None:
        """
        Intended to be called between sequences. Can be overriden by the child class.
        """
        self.counter = 0

    @abstractmethod
    def stabilize(self, z: torch.Tensor) -> torch.Tensor:
        pass
