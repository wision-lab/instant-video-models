import torch
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from stability.stabilizers.base import Stabilizer


# Decrease final_bias_init when expanding the fusion kernel (as this reduces the
# resulting weight on the current time step).
class ControlledConvEMAStabilizer(Stabilizer):
    def __init__(
        self,
        conv_kernel_size: int = 3,
        exclude_feature_inputs: bool = False,
        final_bias_init: float = -4.0,
        fusion_kernel_size: int = 1,
        hidden_layers: int = 2,
        hook_target: str = "output",
        internal_channel_ratio: int = None,
        internal_channels: int = 64,
        interpolate_mode: str = "bilinear",
        skip_connection: bool = False,
    ) -> None:
        super().__init__(hook_target=hook_target)

        # Convolutional weights and biases must be initialized lazily (we may not know
        # the number of channels until we see the first input).
        self.conv_weights = nn.ParameterList()
        self.conv_biases = nn.ParameterList()
        for _ in range(hidden_layers + 2):
            self.conv_weights.append(nn.UninitializedParameter(dtype=torch.float32))
            self.conv_biases.append(nn.UninitializedParameter(dtype=torch.float32))

        # Leaky ReLU on all layers except the output
        self.activation_layers = nn.ModuleList()
        for _ in range(hidden_layers + 1):
            self.activation_layers.append(nn.LeakyReLU())

        self.conv_kernel_size = conv_kernel_size
        self.exclude_feature_inputs = exclude_feature_inputs
        self.final_bias_init = final_bias_init
        self.fusion_kernel_size = fusion_kernel_size
        self.hidden_layers = hidden_layers
        self.internal_channel_ratio = internal_channel_ratio
        self.internal_channels = internal_channels
        self.interpolate_mode = interpolate_mode
        self.skip_connection = skip_connection

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
            # Retrieve and spatially interpolate the backbone output.
            backbone_output = self.module_links["controller_backbone"].retrieve()
            feature_size = z.shape[-2:]
            if backbone_output.shape[-2:] != feature_size:
                backbone_output = nn.functional.interpolate(
                    backbone_output, size=feature_size, mode=self.interpolate_mode
                )

            # Lazy parameter initialization
            if torch.nn.parameter.is_lazy(self.conv_weights[0]):
                input_channels = backbone_output.shape[1]
                if not self.exclude_feature_inputs:
                    input_channels += z.shape[1] * 3
                if self.internal_channel_ratio is None:
                    internal_channels = self.internal_channels
                else:
                    internal_channels = int(z.shape[1] * self.internal_channel_ratio)
                output_channels = z.shape[1] * (self.fusion_kernel_size**2)
                kernel_shape = (self.conv_kernel_size, self.conv_kernel_size)
                self.conv_weights[0].materialize(
                    (internal_channels, input_channels) + kernel_shape
                )
                self.conv_biases[0].materialize((internal_channels,))
                for i in range(self.hidden_layers):
                    self.conv_weights[i + 1].materialize(
                        (internal_channels, internal_channels) + kernel_shape
                    )
                    self.conv_biases[i + 1].materialize((internal_channels,))
                self.conv_weights[-1].materialize(
                    (output_channels, internal_channels) + kernel_shape
                )
                self.conv_biases[-1].materialize((output_channels,))
                for weight in self.conv_weights[:-1]:
                    nn.init.xavier_uniform_(
                        weight, nn.init.calculate_gain("leaky_relu")
                    )
                nn.init.xavier_uniform_(
                    self.conv_weights[-1], nn.init.calculate_gain("linear")
                )
                for bias in self.conv_biases[:-1]:
                    nn.init.zeros_(bias)
                with torch.no_grad():
                    self.conv_biases[-1].copy_(
                        torch.full_like(self.conv_biases[-1], self.final_bias_init)
                    )

            # Apply the controller head with a skip connection over hidden layers.
            q = backbone_output
            if not self.exclude_feature_inputs:
                q = torch.concatenate(
                    [q, z, self.memory_stabilized, self.memory_unstabilized], dim=1
                )
            q = nn.functional.conv2d(
                q, self.conv_weights[0], self.conv_biases[0], padding="same"
            )
            q = self.activation_layers[0](q)
            skip = q
            for i in range(self.hidden_layers):
                q = nn.functional.conv2d(
                    q, self.conv_weights[i + 1], self.conv_biases[i + 1], padding="same"
                )
                q = self.activation_layers[i + 1](q)
            if self.skip_connection:
                q = q + skip

            # Perform the last convolution and spatiotemporal fusion (using gradient
            # checkpointing to reduce training memory).
            z_stabilized = checkpoint(
                _spatiotemporal_fusion,
                q,
                z,
                self.conv_weights[-1],
                self.conv_biases[-1],
                self.memory_stabilized,
                self.fusion_kernel_size,
                use_reentrant=False,
            )
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


# Extracted to a function for gradient checkpointing (product with unfolded tensor is
# quite large)
def _spatiotemporal_fusion(
    q: torch.Tensor,
    z: torch.Tensor,
    last_conv_weights: nn.Parameter,
    last_conv_biases: nn.Parameter,
    stabilized_memory: torch.Tensor,
    fusion_kernel_size: int,
) -> torch.Tensor:
    head_output = nn.functional.conv2d(
        q, last_conv_weights, last_conv_biases, padding="same"
    )
    shape = z.shape
    head_output = rearrange(head_output, "b (c p) h w -> b c p (h w)", c=shape[1])
    eta = torch.concatenate(
        [head_output, torch.zeros_like(head_output[:, :, :1])], dim=2
    )
    eta = eta.softmax(dim=2)
    stabilized_memory = nn.functional.unfold(
        stabilized_memory, fusion_kernel_size, padding=fusion_kernel_size // 2
    )
    stabilized_memory = rearrange(
        stabilized_memory, "b (c p) hw -> b c p hw", c=shape[1]
    )
    z = rearrange(z, "b c h w -> b c (h w)")
    z_stabilized = (stabilized_memory * eta[:, :, :-1]).sum(dim=2) + eta[:, :, -1] * z
    return z_stabilized.view(shape)
