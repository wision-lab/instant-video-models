import torch
from torch import nn

from stability.stabilizers.base import Stabilizer


class SimpleRecurrentConvStabilizer(Stabilizer):
    def __init__(
        self,
        conv_kernel_size: int = 3,
        hidden_layers: int = 2,
        hook_target: str = "output",
        internal_channels: int = 64,
    ):
        super().__init__(hook_target=hook_target)
        initial_layers = [
            nn.LazyConv2d(
                internal_channels,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
            ),
            nn.LeakyReLU(),
        ]
        for _ in range(hidden_layers):
            initial_layers.append(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=conv_kernel_size,
                    padding=conv_kernel_size // 2,
                )
            )
            initial_layers.append(nn.LeakyReLU())
        self.initial_layers = nn.Sequential(*initial_layers)
        self.final_conv_weights = nn.UninitializedParameter(dtype=torch.float32)
        self.final_conv_bias = nn.UninitializedParameter(dtype=torch.float32)

        self.conv_kernel_size = conv_kernel_size
        self.internal_channels = internal_channels

        # References to the last stabilized and unstabilized feature tensors
        self.memory_stabilized = None
        self.memory_unstabilized = None

    def reset(self) -> None:
        super().reset()
        self.memory_stabilized = None
        self.memory_unstabilized = None

    def stabilize(self, z: torch.Tensor) -> torch.Tensor:
        # Add a channel axis if needed.
        unsqueeze_channels = z.ndim == 3
        if unsqueeze_channels:
            z = z.unsqueeze(dim=1)

        # Apply stabilization.
        if self.memory_unstabilized is not None:
            # Lazy parameter initialization
            if torch.nn.parameter.is_lazy(self.final_conv_weights):
                self.final_conv_weights.materialize(
                    (
                        z.shape[1],
                        self.internal_channels,
                        self.conv_kernel_size,
                        self.conv_kernel_size,
                    )
                )
                self.final_conv_bias.materialize((z.shape[1],))
                nn.init.xavier_uniform_(
                    self.final_conv_weights, nn.init.calculate_gain("linear")
                )
                nn.init.zeros_(self.final_conv_bias)

            # Apply the backbone.
            q = torch.concatenate(
                [z, self.memory_stabilized, self.memory_unstabilized], dim=1
            )
            q = self.initial_layers(q)
            q = nn.functional.conv2d(
                q, self.final_conv_weights, self.final_conv_bias, padding="same"
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

        # Remove any channel axis that was added.
        if unsqueeze_channels:
            z_stabilized = z_stabilized.squeeze(dim=1)

        return z_stabilized
