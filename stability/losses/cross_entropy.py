"""Defines the cross-entropy loss.

This is a simple nn.Module that does not accumulate state.
"""

import torch
from torch import nn


class CrossEntropy(nn.Module):
    """The cross-entropy loss.

    This is a simple nn.Module that does not accumulate state.
    """

    def __init__(self, class_dim: int = 0, epsilon: float = 1e-6) -> None:
        """Initializes the loss.

        Args:
            class_dim: The dimension along which softmax should be applied to determine
                category scores. Defaults to 0.
            epsilon: An epsilon to use within the log function to prevent -inf/nan
                outputs. Defaults to 1e-6.
        """
        self.class_dim = class_dim
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute the cross-entropy loss between predicted and ground-truth tensors.

        If the shapes don't match, assume true contains class indices. Otherwise, assume
        true contains logits (predictions from the last time step, in the case of a
        stability metric).

        Args:
            pred: Predicted logits.
            true: Ground-truth logits or class indices.

        Returns:
            The value of the cross-entropy loss.
        """
        pred = pred.softmax(dim=self.class_dim)

        if true.shape != pred.shape:
            true = nn.functional.one_hot(true, pred.shape[self.class_dim])
            permute_dims = list(range(true.ndim - 1))
            permute_dims.insert(self.class_dim, true.ndim - 1)
            true = true.permute(*permute_dims)
        else:
            true = true.softmax(dim=self.class_dim)

        return torch.mean(
            -torch.sum(true * torch.log(pred + self.epsilon), dim=self.class_dim)
        )
