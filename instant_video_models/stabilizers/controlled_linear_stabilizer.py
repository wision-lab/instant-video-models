"""Defines a stabilizer with a controller using linear layers."""

import torch
from torch import nn

from instant_video_models.stabilizers.base import Stabilizer


class ControlledLinearStabilizer(Stabilizer):
    """A stabilizer with a controller using linear layers."""

    def __init__(
        self,
        final_bias_init: float = 4.0,
        hidden_layers: int = 2,
        hook_target: str = "output",
        internal_channels: int = 32,
        pool_backbone_output: bool = False,
    ) -> None:
        """Creates the stabilizer.

        Args:
            final_bias_init: The initialization value for the bias of the controller's
                final layer. Defaults to -4.0.
            hidden_layers: The number of hidden layers in the controller. Defaults to 2.
            hook_target: Whether to apply this hook to the parent's input or output. Can
                be "inputs" or "output". Defaults to "inputs".
            internal_channels: The number of channels for hidden layers. Defaults to 32.
            pool_backbone_output: Whether to perform a global spatial pooling on the
                backbone output. Defaults to False.
        """
        super().__init__(hook_target=hook_target)
        self.final_bias_init = final_bias_init
        self.hidden_layers = hidden_layers
        self.internal_channels = internal_channels
        self.pool_backbone_output = pool_backbone_output

        # The number of channels is not known before we see the first input
        self.linear_weights = nn.ParameterList()
        self.linear_biases = nn.ParameterList()
        for _ in range(hidden_layers + 2):
            self.linear_weights.append(nn.UninitializedParameter(dtype=torch.float32))
            self.linear_biases.append(nn.UninitializedParameter(dtype=torch.float32))

        # Leaky ReLU on all layers except the output
        self.activation_layers = nn.ModuleList()
        for _ in range(hidden_layers + 1):
            self.activation_layers.append(nn.LeakyReLU())

        # Tensors for the last stabilized and unstabilized features
        self.memory_stabilized = None
        self.memory_unstabilized = None

    def reset(self) -> None:
        """Clears internal short-term state.

        Specifically, clears the stored tensors for the last stabilized and unstabilized
        features.
        """
        self.memory_stabilized = None
        self.memory_unstabilized = None

    def stabilize(self, z: torch.Tensor) -> torch.Tensor:
        if self.memory_unstabilized is not None:
            # Grab and optionally pool the backbone output
            backbone_output = self.module_links["controller_backbone"].retrieve()
            if self.pool_backbone_output:
                backbone_output = torch.flatten(
                    backbone_output.mean(dim=(-2, -1)), start_dim=1
                )

            # Parameter materialization and initialization
            if torch.nn.parameter.is_lazy(self.linear_weights[0]):
                input_channels = backbone_output.shape[-1] + z.shape[-1] * 3
                output_channels = z.shape[-1]
                self.linear_weights[0].materialize(
                    (self.internal_channels, input_channels)
                )
                self.linear_biases[0].materialize((self.internal_channels,))
                for i in range(self.hidden_layers):
                    self.linear_weights[i + 1].materialize(
                        (self.internal_channels, self.internal_channels)
                    )
                    self.linear_biases[i + 1].materialize((self.internal_channels,))
                self.linear_weights[-1].materialize(
                    (output_channels, self.internal_channels)
                )
                self.linear_biases[-1].materialize((output_channels,))
                for weight in self.linear_weights[:-1]:
                    nn.init.xavier_uniform_(
                        weight, nn.init.calculate_gain("leaky_relu")
                    )
                nn.init.xavier_uniform_(
                    self.linear_weights[-1], nn.init.calculate_gain("linear")
                )
                for bias in self.linear_biases[:-1]:
                    nn.init.zeros_(bias)
                with torch.no_grad():
                    self.linear_biases[-1].copy_(
                        torch.full_like(self.linear_biases[-1], self.final_bias_init)
                    )

            # Controller head
            q = torch.concatenate(
                [backbone_output, z, self.memory_stabilized, self.memory_unstabilized],
                dim=-1,
            )
            q = nn.functional.linear(q, self.linear_weights[0], self.linear_biases[0])
            q = self.activation_layers[0](q)
            for i in range(self.hidden_layers):
                q = nn.functional.linear(
                    q, self.linear_weights[i + 1], self.linear_biases[i + 1]
                )
                q = self.activation_layers[i + 1](q)
            head_output = nn.functional.linear(
                q, self.linear_weights[-1], self.linear_biases[-1]
            )

            # Temporal fusion
            alpha = head_output.sigmoid()
            z_stabilized = alpha * z + (1.0 - alpha) * self.memory_stabilized
        else:
            z_stabilized = z

        # Used on the next time step
        self.memory_stabilized = z_stabilized.clone()
        self.memory_unstabilized = z.clone()

        return z_stabilized
