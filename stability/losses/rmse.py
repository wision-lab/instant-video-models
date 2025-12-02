"""Defines the RMSE loss.

This is a simple nn.Module that does not accumulate state.
"""

import torch
from torch import nn


class RMSE(nn.Module):
    """The RMSE loss.

    This is a simple nn.Module that does not accumulate state.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        """Initializes the metric.

        Args:
            epsilon: An epsilon to use within the square root to prevent nan gradients
                at zero. Defaults to 1e-6.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Computes the RMSE loss between predicted and ground-truth tensors.

        Args:
            pred: The predicted tensor.
            true: The ground-truth tensor.

        Returns:
            The value of the RMSE loss.
        """
        # The gradient of the square root is infinite at zero
        return torch.sqrt((pred - true).square().mean() + self.epsilon)
