import torch

from stability.visualizers.base import Visualizer


class NoneVisualizer(Visualizer):
    def __call__(self, frame: torch.Tensor, pred: tuple | torch.Tensor) -> torch.Tensor:
        return frame

    def markdown_legend(self) -> str | None:
        return None
