"""Defines the elastic distortion transform."""

import torch
import torchvision.transforms.v2 as tf

from instant_video_models.transforms.base import FrameTransform


class ElasticTransform(FrameTransform):
    """The elastic distortion transform."""

    def __init__(self, magnitude: float = 50.0, smoothness: float = 5.0) -> None:
        """Creates the transform.

        Args:
            magnitude: The amount of distortion, corresponding to alpha in
                torchvision.transforms.v2.ElasticTransform. Defaults to 50.0.
            smoothness: The smoothness of the distortion, corresponding to sigma in
                torchvision.transforms.v2.ElasticTransform. Defaults to 5.0.
        """
        super().__init__()
        self.frame_transform = tf.ElasticTransform(alpha=magnitude, sigma=smoothness)

    def forward_frame(self, frame: torch.Tensor) -> torch.Tensor:
        return self.frame_transform(frame)
