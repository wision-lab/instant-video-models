import torch

from stability.utils import visualize_with_cmap
from stability.visualizers.base import Visualizer


class HeatmapVisualizer(Visualizer):
    def __init__(self, cmap_name: str = "viridis", normalize: bool = True):
        self.cmap_name = cmap_name
        self.normalize = normalize

    def __call__(self, frame: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return visualize_with_cmap(
            pred, cmap_name=self.cmap_name, normalize=self.normalize
        )

    def markdown_legend(self) -> str | None:
        return None
