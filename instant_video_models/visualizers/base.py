"""Defines the base class for output visualizers."""

from abc import ABC, abstractmethod

import torch


class Visualizer(ABC):
    """The base class for output visualizers."""

    @abstractmethod
    def __call__(self, frame: torch.Tensor, pred: tuple | torch.Tensor) -> torch.Tensor:
        """Generates a visualized frame.

        Args:
            frame: The original (model input) image.
            pred: A prediction (e.g., segmentation mask, bounding boxes) for the frame.

        Returns:
            A visualization of the prediction.
        """

    def markdown_legend(self) -> str | None:
        """Generates a markdown color legend (as a string).

        Returns:
            None if a Markdown legend is not supported.
        """
