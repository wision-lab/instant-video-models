"""Defines the segmentation visualizer."""

import matplotlib as mpl
import torch
from torchvision.transforms.v2.functional import convert_image_dtype

from instant_video_models.utils import ColorGenerator
from instant_video_models.visualizers.base import Visualizer


class SegmentationVisualizer(Visualizer):
    """The segmentation visualizer."""

    def __init__(
        self, alpha: float = 1.0, cmap_name: str = "rainbow", id_to_name: dict = None
    ) -> None:
        """Creates the visualizer.

        Args:
            alpha: The opacity of the segmentation mask. Defaults to 1.0.
            cmap_name: The name of the matplotlib color map to sample colors from.
                Defaults to "rainbow".
            id_to_name: A mapping from class IDs to names, used when creating the
                legend. Defaults to None.
        """
        self.alpha = alpha
        self.cmap_name = cmap_name
        self.id_to_name = id_to_name
        self.color_generator = ColorGenerator(convert_to_hex=False)

    def __call__(self, frame: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        original_dtype = frame.dtype
        output = convert_image_dtype(frame, torch.float32).to(pred.device)
        class_ids = pred.argmax(dim=-3)
        for id_value in class_ids.unique():
            class_label = id_value.item()
            if self.id_to_name is not None:
                class_label = self.id_to_name[class_label]
            color = self.color_generator(class_label)
            color = torch.tensor(color[:3]).view(-1, 1, 1).to(output.device)
            mask = class_ids.eq(id_value).expand(output.shape)
            output = output * (1.0 - self.alpha * mask) + color * self.alpha * mask
        return convert_image_dtype(output, original_dtype)

    def markdown_legend(self) -> str | None:
        lines = []
        for label, color in self.color_generator.colors.items():
            lines.append(
                f'<span style="color:{mpl.colors.to_hex(color)}">{label}</span>'
            )
        return "<br/>\n".join(lines)
