"""Defines the transport (Wasserstein) instability metric.

This is a prototype implementation - results may vary.
"""

import math

import numpy as np
import scipy.sparse
import torch
from einops import rearrange
from scipy.optimize import linprog

from instant_video_models.metrics.base import MeanInstability
from instant_video_models.utils import pad_to_multiple


class TransportInstability(MeanInstability):
    """The transport (Wasserstein) instability metric.

    This is a prototype implementation - results may vary.
    """

    def __init__(
        self, assume_batch_dim: bool = True, gamma: float = 1.0, tile_size: int = None
    ) -> None:
        """Initializes the metric.

        The mask_nan option is not supported here because masking flattens the shape,
        and the transport metric assumes a particular tensor structure.

        Args:
            assume_batch_dim: Whether to assume the first axis is a batch dimension. If
                True, inputs are separated along the batch dimension and passed
                independently to the core update logic. Defaults to True.
            gamma: The cost to create or destroy a unit of mass. Defaults to 1.0.
            tile_size: If this is not None, compute the metric tile-wise to reduce
                compute and memory costs. Defaults to None.
        """
        super().__init__(assume_batch_dim)
        self.gamma = gamma
        self.tile_size = tile_size

        # The cost and constraint matrics only depend on the input size and gamma. Cache
        # them to avoid costly re-generation.
        self.matrices_cache = {}

    def compute_stability(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Computes the instability for a single item with no batch dimension.

        Args:
            a: The last frame.
            b: The current frame.

        Returns:
            The transport instability between the frames.
        """
        h, w = a.shape[-2:]
        if (h, w) not in self.matrices_cache:
            self.matrices_cache[(h, w)] = transport_matrices(
                h, w, self.gamma, self.tile_size
            )
        return transport_distance(
            a, b, self.gamma, self.matrices_cache[(h, w)], self.tile_size
        )


def transport_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    gamma: float = 1.0,
    matrices: tuple = None,
    tile_size: int = None,
) -> torch.Tensor:
    """Computes the transport distance (instability) between two frames.

    Args:
        a: The last frame.
        b: The current frame.
        gamma: The cost to create or destroy a unit of mass. Defaults to 1.0.
        matrices: If this is not None, use the provided optimization matrices instead of
            generating them. This can be used to cache the matrices for repeated use.
            Defaults to None.
        tile_size: If this is not None, compute the metric tile-wise to reduce
            compute and memory costs. Defaults to None.

    Raises:
        RuntimeError: If the linear optimization fails.

    Returns:
        The transport distance between the frames.
    """
    output_shape = a.shape[:-2]
    h, w = a.shape[-2:]
    if matrices is None:
        matrices = transport_matrices(h, w, gamma, tile_size)
    cost_vector, constraint_matrix = matrices

    # Flatten all leading axes and set up the tiling axis
    a = a.unsqueeze(dim=0).flatten(end_dim=-3)
    b = b.unsqueeze(dim=0).flatten(end_dim=-3)
    if tile_size is None:
        # Dummy tile axis
        rearrange_kwargs = {"pattern": "b h w -> b 1 (h w)"}
    else:
        a = pad_to_multiple(a, tile_size)
        b = pad_to_multiple(b, tile_size)
        rearrange_kwargs = {
            "pattern": "b (q h) (r w) -> b (q r) (h w)",
            "h": tile_size,
            "w": tile_size,
        }
    a = rearrange(a, **rearrange_kwargs).cpu()
    b = rearrange(b, **rearrange_kwargs).cpu()

    # Iteration over flattened leading axes
    results = []
    for i in range(a.shape[0]):

        # Iteration over tile axis
        tile_sum = 0.0
        for j in range(a.shape[1]):
            constraint_vector = np.concatenate(
                [b[i, j] - a[i, j], torch.zeros(1, device=a.device, dtype=a.dtype)]
            )

            # highs-ds seems to be the fastest method
            result = linprog(
                cost_vector,
                A_eq=constraint_matrix,
                b_eq=constraint_vector,
                method="highs-ds",
                bounds=(0, None),
            )
            if not result.success:
                raise RuntimeError(f'Optimization failed, message "{result.message}".')
            tile_sum += result.fun
        results.append(tile_sum)
    return torch.tensor(results, device=a.device).view(output_shape)


def transport_matrices(
    h: int, w: int, gamma: float = 1.0, tile_size: int = None
) -> tuple:
    """Generates optimization matrices for the specified size and gamma.

    Args:
        h: The tensor height.
        w: The tensor width.
        gamma: The cost to create or destroy a unit of mass. Defaults to 1.0.
        tile_size: If this is not None, assume the metric will be computed tile-wise and
            override the height/width with this value. Defaults to None.

    Returns:
        A tuple (cost_vector, constraint_matrix), where constraint_matrix is CSR sparse.
    """
    # If we don't move mass, then we have to create and destroy it. The cost to create
    # and destroy a unit of mass is 2 * gamma. So it will never make sense to move
    # something further than a distance of 2 * gamma.
    max_distance = math.floor(2 * gamma)

    if tile_size is not None:
        h = tile_size
        w = tile_size
    total_pixels = h * w
    window_size = max_distance * 2 + 1
    pixels_per_window = window_size**2
    nodes_per_window = pixels_per_window + 2

    # Construct the cost vector
    cost_vector = np.zeros(total_pixels * nodes_per_window)
    window_dx, window_dy = np.meshgrid(
        *[np.arange(-max_distance, max_distance + 1) for _ in range(2)]
    )
    window_distances = np.sqrt(window_dx**2 + window_dy**2).flatten()
    for i in range(total_pixels):
        offset = nodes_per_window * i

        # Pixel nodes
        cost_vector[offset : offset + pixels_per_window] = window_distances

        # Source and sink nodes
        cost_vector[offset + pixels_per_window : offset + nodes_per_window] = gamma

    # Construct the equality constraint matrix
    # - Rows 0:total_pixels constrain the incoming flow to each pixel
    # - The last row forces some flows to zero (where the source is out of bounds)
    constraint_matrix = np.zeros(
        (total_pixels + 1, total_pixels * nodes_per_window), dtype=np.int8
    )
    for i in range(total_pixels):
        offset = i * nodes_per_window

        # Sum of incoming flow from pixel nodes
        constraint_matrix[i, offset : offset + pixels_per_window] = 1

        # Incoming flow from source node
        constraint_matrix[i, offset + pixels_per_window] = 1

        # Iteration over neighborhood
        k = -1
        for dy in range(-max_distance, max_distance + 1):
            for dx in range(-max_distance, max_distance + 1):
                k += 1

                # Check whether the neighborhood pixel is in-bounds
                y = i // w + dy
                x = i % w + dx
                if y < 0 or x < 0 or y >= h or x >= w:
                    # Force incoming flow to zero for invalid neighbors
                    constraint_matrix[-1, offset + k] = 1

                    # Skip outgoing flow constraints for invalid neighbors
                    continue

                # To the beginning of the window
                j = (y * w + x) * nodes_per_window

                # To the middle of the window
                j += max_distance * window_size + max_distance

                # To the pixel (subtract due to outgoing relationship)
                j -= dy * window_size + dx

                # Outgoing flow to pixel node (decrement instead of assign -1 to return
                # the constraint for the self link to zero)
                constraint_matrix[i, j] -= 1

        # Outgoing flow to sink node
        constraint_matrix[i, offset + pixels_per_window + 1] = -1

    return cost_vector, scipy.sparse.csr_matrix(constraint_matrix)
