"""Contains helper functions common to more than one script."""

import torch
from torch.utils.data import DataLoader


def prepare_item(item: tuple, num_workers: int) -> list:
    prepared = []
    for x in item:
        if isinstance(x, torch.Tensor):
            # Put the time axis first so iteration will split the tensor in time.
            prepared.append(x.transpose(0, 1))
        else:
            prepared.append(DataLoader(x, batch_size=None, num_workers=num_workers))
    return prepared


def prepare_time_step(time_step, input_transform, device):
    if input_transform is None:
        frame, ground_truth = time_step
    else:
        ground_truth = time_step[0]
        frame = input_transform(ground_truth)
    return frame.to(device), ground_truth.to(device)
