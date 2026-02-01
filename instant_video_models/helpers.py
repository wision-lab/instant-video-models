"""Defines helper functions common to more than one script."""

from typing import Callable

import torch
from torch.utils.data import DataLoader


def prepare_item(item: tuple, num_workers: int) -> tuple:
    """Prepares a time series for temporal iteration.

    Args:
        item: A tuple returned by a top-level (dataset) iterator. Each item in the tuple
            represents a time series.
        num_workers: The number of worker threads to use if creating a data loader for
            a time series.

    Returns:
        A tuple where each item has been prepared for temporal iteration.
    """
    prepared = []
    for x in item:
        if isinstance(x, torch.Tensor):
            # Put the time axis first so iteration will split the tensor in time
            prepared.append(x.transpose(0, 1))
        else:
            prepared.append(DataLoader(x, batch_size=None, num_workers=num_workers))
    return tuple(prepared)


def prepare_time_step(
    time_step: tuple, input_transform: Callable | None, device: str | torch.device
) -> tuple:
    """Prepares a time step for consumption by the model and metrics.

    Args:
        time_step: A tuple containing one or two tensors.
        input_transform: An optional transform to apply to the ground-truth in order to
            generate the input. If this is None, time_step must have the form
            (frame, ground_truth). Otherwise, time_step must have the form
            (ground_truth,).
        device: The device where tensors should be sent (this should match the model).

    Returns:
        A tuple of tensors that have been prepared for consumption.
    """
    if input_transform is None:
        frame, ground_truth = time_step
    else:
        ground_truth = time_step[0]
        frame = input_transform(ground_truth)
    return frame.to(device), ground_truth.to(device)
