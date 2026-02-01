"""Defines a stabilizer with a convolutional controller."""

import torch
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from instant_video_models.stabilizers.base import Stabilizer


class ControlledConvStabilizer(Stabilizer):
    """A stabilizer with a convolutional controller."""

    def __init__(
        self,
        conv_kernel_size: int = 3,
        exclude_feature_inputs: bool = False,
        final_bias_init: float = -4.0,
        fusion_kernel_size: int = 1,
        hidden_layers: int = 2,
        hook_target: str = "output",
        internal_channels: int = 64,
        interpolate_mode: str = "bilinear",
    ) -> None:
        """Creates the stabilizer.

        Args:
            conv_kernel_size: The kernel size for convolutions. Defaults to 3.
            exclude_feature_inputs: If True, the controller consumes only the output of
                the backbone, not feature tensors. This is useful if stabilizers are
                being directly composed, as it prevents interactions between
                stabilizers. Defaults to False.
            final_bias_init: The initialization value for the bias of the controller's
                final layer. Defaults to -4.0.
            fusion_kernel_size: The kernel size to use for stabilizer spatial fusion.
                The controller predicts a kernel of this size for each stabilized value.
                Defaults to 1.
            hidden_layers: The number of hidden layers in the controller. Defaults to 2.
            hook_target: Whether to apply this hook to the parent's input or output. Can
                be "inputs" or "output". Defaults to "inputs".
            internal_channels: The number of channels for hidden layers. Defaults to 32.
            interpolate_mode: The method to use when interpolating backbone outputs.
                Defaults to "bilinear".
        """
        super().__init__(hook_target=hook_target)
        self.conv_kernel_size = conv_kernel_size
        self.exclude_feature_inputs = exclude_feature_inputs
        self.final_bias_init = final_bias_init
        self.fusion_kernel_size = fusion_kernel_size
        self.hidden_layers = hidden_layers
        self.internal_channels = internal_channels
        self.interpolate_mode = interpolate_mode

        # The number of channels is not known before we see the first input
        self.conv_weights = nn.ParameterList()
        self.conv_biases = nn.ParameterList()
        for _ in range(hidden_layers + 2):
            self.conv_weights.append(nn.UninitializedParameter(dtype=torch.float32))
            self.conv_biases.append(nn.UninitializedParameter(dtype=torch.float32))

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
        # Add a channel axis if needed
        unsqueeze_channels = z.ndim == 3
        if unsqueeze_channels:
            z = z.unsqueeze(dim=1)

        if self.memory_unstabilized is not None:
            # Grab and spatially interpolate the backbone output
            backbone_output = self.module_links["controller_backbone"].retrieve()
            feature_size = z.shape[-2:]
            if backbone_output.shape[-2:] != feature_size:
                backbone_output = nn.functional.interpolate(
                    backbone_output, size=feature_size, mode=self.interpolate_mode
                )

            # Parameter materialization and initialization
            if torch.nn.parameter.is_lazy(self.conv_weights[0]):
                input_channels = backbone_output.shape[1]
                if not self.exclude_feature_inputs:
                    input_channels += z.shape[1] * 3
                output_channels = z.shape[1] * (self.fusion_kernel_size**2)
                kernel_shape = (self.conv_kernel_size, self.conv_kernel_size)
                self.conv_weights[0].materialize(
                    (self.internal_channels, input_channels) + kernel_shape
                )
                self.conv_biases[0].materialize((self.internal_channels,))
                for i in range(self.hidden_layers):
                    self.conv_weights[i + 1].materialize(
                        (self.internal_channels, self.internal_channels) + kernel_shape
                    )
                    self.conv_biases[i + 1].materialize((self.internal_channels,))
                self.conv_weights[-1].materialize(
                    (output_channels, self.internal_channels) + kernel_shape
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

            # Controller head
            q = backbone_output
            if not self.exclude_feature_inputs:
                q = torch.concatenate(
                    [q, z, self.memory_stabilized, self.memory_unstabilized], dim=1
                )
            q = nn.functional.conv2d(
                q, self.conv_weights[0], self.conv_biases[0], padding="same"
            )
            q = self.activation_layers[0](q)
            for i in range(self.hidden_layers):
                q = nn.functional.conv2d(
                    q, self.conv_weights[i + 1], self.conv_biases[i + 1], padding="same"
                )
                q = self.activation_layers[i + 1](q)

            # Final convolution and spatiotemporal fusion
            # Use gradient checkpointing to reduce training memory
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

        # Used on the next time step
        self.memory_stabilized = z_stabilized.clone()
        self.memory_unstabilized = z.clone()

        # Remove temporary channel axis
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
