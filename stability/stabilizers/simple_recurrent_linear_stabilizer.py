import torch
from torch import nn

from stability.stabilizers.base import Stabilizer


class SimpleRecurrentLinearStabilizer(Stabilizer):
    def __init__(
        self,
        hidden_layers: int = 2,
        hook_target: str = "output",
        internal_channels: int = 64,
    ):
        super().__init__(hook_target=hook_target)
        initial_layers = [
            nn.LazyLinear(internal_channels),
            nn.LeakyReLU(),
        ]
        for _ in range(hidden_layers):
            initial_layers.append(nn.Linear(internal_channels, internal_channels))
            initial_layers.append(nn.LeakyReLU())
        self.initial_layers = nn.Sequential(*initial_layers)
        self.final_linear_weights = nn.UninitializedParameter(dtype=torch.float32)
        self.final_linear_bias = nn.UninitializedParameter(dtype=torch.float32)

        self.internal_channels = internal_channels

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
            # Lazy parameter initialization
            if torch.nn.parameter.is_lazy(self.final_linear_weights):
                self.final_linear_weights.materialize(
                    (z.shape[-1], self.internal_channels)
                )
                self.final_linear_bias.materialize((z.shape[-1],))
                nn.init.xavier_uniform_(
                    self.final_linear_weights, nn.init.calculate_gain("linear")
                )
                nn.init.zeros_(self.final_linear_bias)

            # Apply the backbone.
            q = torch.concatenate(
                [z, self.memory_stabilized, self.memory_unstabilized], dim=-1
            )
            q = self.initial_layers(q)
            q = nn.functional.linear(
                q, self.final_linear_weights, self.final_linear_bias
            )
            z_stabilized = q
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
