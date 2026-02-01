"""Defines a convolutional controller backbone."""

import torch
from torch import nn

from instant_video_models.controllers.base import Controller


class ConvBackbone(Controller):
    """A stabilization controller backbone with a convolutional architecture."""

    def __init__(
        self,
        conv_kernel_size: int = 3,
        hidden_layers: int = 6,
        hook_target: str = "inputs",
        internal_channels: int = 32,
    ) -> None:
        """Creates the backbone.

        Args:
            conv_kernel_size: The kernel size for convolutions. Defaults to 3.
            hidden_layers: The number of hidden layers in the backbone. This count
                includes the backbone output layer. Defaults to 6.
            hook_target: Whether to apply this hook to the parent's input or output. Can
                be "inputs" or "output". Defaults to "inputs".
            internal_channels: The number of channels for hidden layers. Defaults to 32.
        """
        super().__init__(hook_target=hook_target)
        self.hidden_layers = hidden_layers
        self.conv_layers = nn.ModuleList()

        # Use a lazy conv to avoid specifying input channels
        self.conv_layers.append(
            nn.LazyConv2d(
                internal_channels,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
            )
        )

        # Hidden and backbone output layers
        for _ in range(hidden_layers):
            self.conv_layers.append(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=conv_kernel_size,
                    padding="same",
                )
            )

        # Leaky ReLU on all layers
        self.activation_layers = nn.ModuleList()
        for _ in range(hidden_layers + 1):
            self.activation_layers.append(nn.LeakyReLU())

        # Tensors for the last input and output
        self.memory = None
        self.output = None

    def available(self) -> bool:
        """Determines whether an output for this module is available.

        Returns:
            True if update() has been called at least twice since the last reset().
        """
        return self.output is not None

    def reset(self) -> None:
        """Clears internal short-term state.

        Specifically, clears the stored tensors for the last input and output.
        """
        super().reset()
        self.memory = None
        self.output = None

    def retrieve(self) -> torch.Tensor:
        return self.output.clone()

    def update(self, x: torch.Tensor) -> None:
        # Parameter materialization and initialization
        if torch.nn.parameter.is_lazy(self.conv_layers[0].weight):
            dummy_input = torch.concatenate([x, x], dim=1)
            self.conv_layers[0](dummy_input)
            for layer in self.conv_layers:
                nn.init.xavier_uniform_(
                    layer.weight, nn.init.calculate_gain("leaky_relu")
                )
                nn.init.zeros_(layer.bias)

        # Controller backbone
        if self.memory is not None:
            q = self.conv_layers[0](torch.concatenate([x, self.memory], dim=1))
            q = self.activation_layers[0](q)
            for i in range(self.hidden_layers):
                q = self.conv_layers[i + 1](q)
                q = self.activation_layers[i + 1](q)
            self.output = q

        # Used by the backbone on the next time step
        self.memory = x.clone()
