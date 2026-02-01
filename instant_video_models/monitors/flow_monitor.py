"""Defines an optical flow monitor hook.

Uses RAFT internally to compute flow.
"""

import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from instant_video_models.monitors.base import Monitor
from instant_video_models.utils import pad_to_multiple


class FlowMonitor(Monitor):
    """An optical flow monitor hook.

    Uses RAFT internally to compute flow.
    """

    def __init__(
        self,
        hook_input_index: int = 0,
        hook_selections: dict = None,
        hook_target: str = "output",
    ) -> None:
        super().__init__(hook_input_index, hook_selections, hook_target)
        weights = Raft_Large_Weights.DEFAULT
        self.preprocessing = weights.transforms()
        self.flow_model = raft_large(weights=weights, progress=False).eval()

        # The output to return when retrieve() is called
        self.flow = None

        # The previous input to forward, required to compute flow
        self.memory = None

    def available(self) -> bool:
        """
        Returns:
            True if forward() has been called at least twice since the last reset().
        """
        return self.flow is not None

    def forward(self, x: torch.Tensor) -> None:
        """Computes optical flow between the previous and current inputs.

        Args:
            x: The current input (flow destination).
        """
        x = x.clip(min=0.0, max=1.0)
        add_batch = x.ndim <= 3
        if add_batch:
            x = x.unsqueeze(dim=0)
        if self.memory is not None:
            a, b = self.preprocessing(self.memory, x)
            old_h, old_w = a.shape[-2:]
            a = pad_to_multiple(a, 8)
            b = pad_to_multiple(b, 8)
            flow = self.flow_model(a, b)[-1][..., :old_h, :old_w]
            if add_batch:
                flow = flow.squeeze(dim=0)
            self.flow = flow
        self.memory = x.clone()

    def reset(self) -> None:
        """Resets memory of the previous input and the stored flow."""
        self.flow = None
        self.memory = None

    def retrieve(self) -> torch.Tensor:
        return self.flow.clone()
