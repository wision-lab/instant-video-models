from abc import ABC, abstractmethod

import torch


class Visualizer(ABC):
    @abstractmethod
    def __call__(self, frame: torch.Tensor, pred: tuple | torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def markdown_legend(self) -> str | None:
        pass
