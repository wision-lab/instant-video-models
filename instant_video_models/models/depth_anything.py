"""Defines the Depth Anything v2 model.

Uses a Huggingface implementation.
"""

import torch
from torch import nn
from torchvision.transforms.v2.functional import convert_image_dtype
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthAnything(nn.Module):
    """The Depth Anything v2 model.

    Uses a Huggingface implementation.
    """

    def __init__(self, revision: str = "5426e4f") -> None:
        """Initializes the model.

        Args:
            revision: The specific model revision to use. See the
                "depth-anything/Depth-Anything-V2-Small-hf" Huggingface repo for
                options. Defaults to "5426e4f".
        """
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf", revision=revision
        )

        # The default model config predicts relative disparity (inverse depth, up to a
        # scale and shift in the inverse-depth space)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf", revision=revision
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimates a depth map from the input image.

        Args:
            x: The input image, assumed to have shape (b, c, h, w).

        Returns:
            A floating-point depth map of shape (b, h, w).
        """
        x = convert_image_dtype(x, torch.float32)

        # Preprocessing
        # do_rescale=False because images have already been scaled to 0-1
        inputs = self.image_processor(images=x, return_tensors="pt", do_rescale=False)

        # Inference
        # The input to self.model has already gone through the initial resize and
        # padding, so we can pass it to the controller backbone without worrying about
        # padding-induced offsets.
        results = self.model(**{k: v.to(x.device) for k, v in inputs.items()})
        depth = results["predicted_depth"]

        # Postprocessing
        depth = depth.unsqueeze(dim=1)
        depth = nn.functional.interpolate(depth, size=x.shape[-2:], mode="bicubic")
        return depth.squeeze(dim=1)
