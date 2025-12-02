import torch
from torch import nn

from stability.stabilizers.base import Stabilizer


class ControlledLinearEMAStabilizer(Stabilizer):
    def __init__(
        self,
        final_bias_init: float = 4.0,
        hidden_layers: int = 2,
        hook_target: str = "output",
        internal_channel_ratio: int = None,
        internal_channels: int = 32,
        pool_backbone_output: bool = False,
        skip_connection: bool = False,
    ) -> None:
        super().__init__(hook_target=hook_target)

        # Linear weights and biases must be initialized lazily (we may not know the
        # number of channels until we see the first input).
        self.linear_weights = nn.ParameterList()
        self.linear_biases = nn.ParameterList()
        for _ in range(hidden_layers + 2):
            self.linear_weights.append(nn.UninitializedParameter(dtype=torch.float32))
            self.linear_biases.append(nn.UninitializedParameter(dtype=torch.float32))

        # Leaky ReLU on all layers except the output
        self.activation_layers = nn.ModuleList()
        for _ in range(hidden_layers + 1):
            self.activation_layers.append(nn.LeakyReLU())

        self.final_bias_init = final_bias_init
        self.hidden_layers = hidden_layers
        self.internal_channel_ratio = internal_channel_ratio
        self.internal_channels = internal_channels
        self.pool_backbone_output = pool_backbone_output
        self.skip_connection = skip_connection

        # References to the last stabilized and unstabilized feature tensors
        self.memory_stabilized = None
        self.memory_unstabilized = None

    def reset(self) -> None:
        super().reset()
        self.memory_stabilized = None
        self.memory_unstabilized = None

    def stabilize(self, z: torch.Tensor) -> torch.Tensor:
        # Apply stabilization.
        if self.memory_unstabilized is not None:
            # Retrieve the backbone output.
            backbone_output = self.module_links["controller_backbone"].retrieve()
            if self.pool_backbone_output:
                backbone_output = torch.flatten(
                    backbone_output.mean(dim=(-2, -1)), start_dim=1
                )

            # Lazy parameter initialization
            if torch.nn.parameter.is_lazy(self.linear_weights[0]):
                input_channels = backbone_output.shape[-1] + z.shape[-1] * 3
                if self.internal_channel_ratio is None:
                    internal_channels = self.internal_channels
                else:
                    internal_channels = int(z.shape[-1] * self.internal_channel_ratio)
                output_channels = z.shape[-1]
                self.linear_weights[0].materialize((internal_channels, input_channels))
                self.linear_biases[0].materialize((internal_channels,))
                for i in range(self.hidden_layers):
                    self.linear_weights[i + 1].materialize(
                        (internal_channels, internal_channels)
                    )
                    self.linear_biases[i + 1].materialize((internal_channels,))
                self.linear_weights[-1].materialize(
                    (output_channels, internal_channels)
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

            # Apply the controller head with a skip connection over hidden layers.
            q = torch.concatenate(
                [backbone_output, z, self.memory_stabilized, self.memory_unstabilized],
                dim=-1,
            )
            q = nn.functional.linear(q, self.linear_weights[0], self.linear_biases[0])
            q = self.activation_layers[0](q)
            skip = q
            for i in range(self.hidden_layers):
                q = nn.functional.linear(
                    q, self.linear_weights[i + 1], self.linear_biases[i + 1]
                )
                q = self.activation_layers[i + 1](q)
            if self.skip_connection:
                q = q + skip
            head_output = nn.functional.linear(
                q, self.linear_weights[-1], self.linear_biases[-1]
            )

            # Perform the temporal fusion.
            alpha = head_output.sigmoid()
            z_stabilized = alpha * z + (1.0 - alpha) * self.memory_stabilized
        else:
            z_stabilized = z

        # Save the stabilized and unstabilized features so we can pass them to the
        # controller head on the next time step.
        self.memory_stabilized = z_stabilized.clone()
        self.memory_unstabilized = z.clone()

        # Allows preventing backpropagation through time
        if self.detach_memory:
            self.memory_stabilized.detach_()
            self.memory_unstabilized.detach_()

        return z_stabilized
