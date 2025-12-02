import torch

from stability.visualizers.base import Visualizer


class ImageVisualizer(Visualizer):
    def __call__(self, frame: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return pred

    def markdown_legend(self) -> str | None:
        return None
