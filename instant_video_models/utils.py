"""Defines miscellaneous utilities."""

import logging
import random
import subprocess
from pathlib import Path
from typing import Hashable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset
from torchvision.io import read_video, write_png, write_video
from torchvision.transforms.v2.functional import convert_image_dtype as convert_dtype
from torchvision.transforms.v2.functional import pad

logger = logging.getLogger(__name__)


class AffineAlignPredictions(nn.Module):
    """An operation for monocular depth affine alignment."""

    def __init__(self, mask_nan: bool = True) -> None:
        """Initializes the operation.

        Args:
            mask_nan: Whether to ignore nan values when performing the linear fit.
                Defaults to True.
        """
        super().__init__()
        self.mask_nan = mask_nan

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> tuple:
        """Aligns predicted depth with ground truth via linear fit.

        Args:
            pred: The predicted depth tensor.
            true: The ground-truth depth tensor.

        Returns:
            A tuple containing (pred_transformed, true), where pred_transformed is the
                result of applying an affine transform to pred.
        """
        if self.mask_nan:
            mask = ~(pred.isnan() | true.isnan())
            pred_masked = pred[mask]
            true_masked = true[mask]
        else:
            pred_masked = pred
            true_masked = true
        try:
            fit = np.polynomial.Polynomial.fit(
                pred_masked.cpu().detach().flatten(),
                true_masked.cpu().detach().flatten(),
                deg=1,
            )
            pred = fit(pred)
        except np.linalg.LinAlgError:
            logger.warning("Skipping affine affine alignment due to failure in fit")
        return pred, true


class ColorGenerator:
    """A generator for producing an unspecified number of category colors."""

    def __init__(self, cmap_name: str = "rainbow", convert_to_hex: bool = True) -> None:
        """Creates the generator.

        Args:
            cmap_name: The name of the matplotlib color map to use for sampling colors.
                Defaults to "rainbow".
            convert_to_hex: If True, return colors as hex codes. Otherwise, colors are
                returned as tuples of floats. Defaults to True.
        """
        self.cmap_name = cmap_name
        self.convert_to_hex = convert_to_hex
        self.cmap = mpl.colormaps[cmap_name]
        self.colors = {}

        # Parameters used to generate sequential colors
        self.exp = 0
        self.position = 0.0
        self.step = 1.0

    def __call__(self, category: Hashable) -> tuple | str:
        """Returns a color for the category.

        Generates a new color if the category has not been seen yet.

        Args:
            category: The category for which a color should be provided.

        Returns:
            A hex code for the color if convert_to_hex is True, otherwise a float tuple.
        """
        if category not in self.colors.keys():
            color = self.cmap(self.position)
            if self.convert_to_hex:
                color = mpl.colors.to_hex(color)
            self.colors[category] = color

            # Update the current position within the color map (0-1)
            self.position += self.step
            if self.position > 1.0:
                self.exp += 1
                self.position = 2 ** (-self.exp)
                self.step = 2 ** (-self.exp + 1)

        return self.colors[category]


class MeanValue:
    """A tracker for a cumulative mean."""

    def __init__(self) -> None:
        """Creates the tracker."""
        self.sum = 0.0
        self.count = 0

    def compute(self) -> float:
        """Computes the cumulative mean.

        Returns:
            The mean value.
        """
        return self.sum / self.count

    def flush(self) -> None:
        """Resets internal statistics."""
        self.sum = 0.0
        self.count = 0

    def update(self, value: float) -> None:
        """Adds a new value to the total considered for the mean.

        Args:
            value: The value to be incorporated.
        """
        self.sum += value
        self.count += 1


class ZippedIterators(IterableDataset):
    """An iterator for traversing a list of sub-iterators in parallel."""

    def __init__(self, iterators: list) -> None:
        """Creates the iterator.

        Args:
            iterators: The list of sub-iterators to traverse in parallel.
        """
        self.iterators = iterators

    def __iter__(self):
        return self

    def __len__(self):
        return min(len(x) for x in self.iterators)

    def __next__(self) -> torch.Tensor:
        return torch.stack([next(x) for x in self.iterators])


def best_pytorch_device() -> str:
    """Determines the PyTorch device to use.

    Returns:
        "cuda" if available, otherwise "cpu".
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def custom_collate(batch: list) -> tuple:
    """A custom batching function to use as collate_fn in a DataLoader.

    Args:
        batch: The batch to collate.

    Returns:
        If the batch contains iterators, a ZippedIterator. Otherwise, returns a stacked
            tensor.
    """
    output = []
    for i, item in enumerate(batch[0]):
        if isinstance(item, IterableDataset):
            output.append(ZippedIterators([x[i] for x in batch]))
        else:
            output.append(torch.stack([x[i] for x in batch]))
    return tuple(output)


def ensure_floating_point(*args) -> torch.Tensor | tuple:
    """Converts non-float arguments to float32.

    Returns:
        A single floating-point tensor if there was only one input, otherwise a tuple
            containing multiple floating-point tensors.
    """
    result = [(x if x.is_floating_point() else x.float()) for x in args]
    return result[0] if len(args) == 1 else tuple(result)


def invoke_on_values(container: dict | list, method_name: str, **kwargs) -> None:
    """Invokes a method on all values in a collection.

    Extra kwargs are passed to the method.

    Args:
        container: A dict or list.
        method_name: The name of the method to invoke on values.
    """
    values = container.values() if isinstance(container, dict) else container
    for item in values:
        getattr(item, method_name)(**kwargs)


def list_cuda_devices() -> list:
    """Lists the names of all CUDA devices.

    Returns:
        A list of CUDA device names (strings).
    """
    devices = []
    for i in range(torch.cuda.device_count()):
        devices.append(torch.cuda.get_device_properties(i).name)
    return devices


def load_clip(clip_config: dict) -> tuple:
    """Loads a video clip specified by a config dict.

    Args:
        clip_config: A configuration dict for the clip to load. The "video" key is
            required and specifies the filename. The "start_time" and "end_time" keys
            can be used to select a specific duration. The "crop" key (with sub-keys
            "x_min", "x_max", "y_min", and "y_max") can be used for spatial cropping.
            The "stride" key specifies a temporal stride.

    Returns:
        A tuple (video, fps), where video is a tensor and fps is the float frame rate.
    """
    video_filename = clip_config["video"]
    video, audio, metadata = read_video(
        video_filename,
        start_pts=clip_config.get("start_time", 0.0),
        end_pts=clip_config.get("end_time", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    if "crop" in clip_config:
        crop = clip_config["crop"]
        video = video[..., crop["y_min"] : crop["y_max"], crop["x_min"] : crop["x_max"]]
    if "stride" in clip_config:
        video = video[:: clip_config["stride"]]
    logger.info(f"Loaded clip from {video_filename}.")
    return video, metadata["video_fps"]


def make_log_header(content: str) -> str:
    """Formats a centered header with a row of "=" on each side.

    Args:
        content: The contents of the header.

    Returns:
        A string containing the formatted header.
    """
    left_pad = (78 - len(content)) // 2
    right_pad = 78 - len(content) - left_pad
    return "=" * left_pad + " " + content + " " + "=" * right_pad


def pad_to_multiple(tensor: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pads the height and width of an image-like tensor to some multiple.

    Args:
        tensor: The tensor to pad.
        multiple: The multiple to which the tensor should be padded. For example,
            setting multiple=8 causes the height and width to be multiples of 8.

    Returns:
        The padded tensor.
    """
    pad_h = -tensor.shape[-2] % multiple
    pad_w = -tensor.shape[-1] % multiple
    if pad_h != 0 or pad_w != 0:
        padding = [0, pad_w, 0, pad_h]
        tensor = nn.functional.pad(tensor, padding)
    return tensor


def save_video(
    filepath: str | Path,
    tensor: torch.Tensor,
    fps: float,
    crf: int = 23,
    preset: str = "medium",
) -> None:
    """Saves a video tensor as a file.

    Args:
        filepath: The filepath where the video should be saved. The file type is
            inferred from the extension.
        tensor: The tensor representing the video (should have layout TCHW).
        fps: The fps to assume when encoding the video.
        crf: The crf (quality) score for the x264 encoder. Lower indicates better
            quality, with 18 being effectively lossless. Defaults to 23.
        preset: The x264 encoding preset to use. Slower presets give better compression.
            Defaults to "medium".
    """
    filepath = Path(filepath)
    tensor = convert_dtype(tensor, torch.uint8)

    # The x264 encoder fails if the height or width is not even
    pad_h = tensor.shape[-2] % 2
    pad_w = tensor.shape[-1] % 2
    if pad_w > 0 or pad_h > 0:
        tensor = pad(tensor, (0, 0, pad_w, pad_h))

    # Convert to THWC
    tensor = tensor.permute(0, 2, 3, 1)

    # Write the video.
    write_video(str(filepath), tensor, fps, options={"crf": str(crf), "preset": preset})
    logger.info(f"Saved video to {filepath.resolve()}.")


def set_random_seeds(seed: int) -> None:
    """Sets the PyTorch, Python, and NumPy random seeds.

    https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        seed: The random seed value.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def stack_videos(
    input_filenames: list,
    output_filename: str,
    size: int,
    crf: int = 23,
    mode: str = "hstack",
    preset: str = "medium",
) -> None:
    """Spatially stacks multiple input videos into a single output video.

    Args:
        input_filenames: A list of input video files to stack.
        output_filename: The filename for the output video.
        size: The size of the output video along the non-stacked axis.
        crf: The crf (quality) score for the x264 encoder. Lower indicates better
            quality, with 18 being effectively lossless. Defaults to 23.
        mode: "hstack" to stack horizontally, or "vstack" to stack vertically. Defaults
            to "hstack".
        preset: The x264 encoding preset to use. Slower presets give better compression.
            Defaults to "medium".

    Raises:
        ValueError: If mode is not "hstack" or "vstack".
    """
    valid_modes = ["hstack", "vstack"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}', options are {valid_modes}.")

    # Build an ffmpeg filter graph that scales videos along one axis to a uniform size,
    # then stacks them along the other axis
    filter_spec = []
    n_videos = len(input_filenames)
    scale_spec = f"-1:{size}" if mode == "hstack" else f"{size}:-1"
    for i in range(n_videos):
        filter_spec.append(f"[{i}:v]scale={scale_spec}[v_{i}];")
    for i in range(n_videos):
        filter_spec.append(f"[v_{i}]")
    filter_spec.append(f"{mode}=inputs={n_videos}[v_stack];")
    filter_spec.append(f"[v_stack]scale=ceil(iw/2)*2:ceil(ih/2)*2")  # Pad to even size
    filter_spec = "".join(filter_spec)

    # ffmpeg argument list
    args = ["ffmpeg"]
    for filename in input_filenames:
        args.extend(["-i", str(filename)])
    args.extend(
        [
            "-codec:v",
            "libx264",
            "-filter_complex",
            filter_spec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-loglevel",
            "warning",
            "-y",
            output_filename,
        ]
    )
    subprocess.run(args, check=True)
    logger.info(f"Saved stacked video to {output_filename}.")


def visualize_with_cmap(
    tensor: torch.Tensor, cmap_name: str = "viridis", normalize: bool = True
) -> torch.Tensor:
    """Visualizes the values in a tensor using a maplotlib color map.

    Args:
        tensor: The tensor to visualize.
        cmap_name: The name of the matplotlib color map to use. Defaults to "viridis".
            normalize: If True, apply a global scale and shift to values so they fill
                the range 0-1. Defaults to True.

    Returns:
        A tensor containing mapped color values.
    """
    cmap = plt.get_cmap(cmap_name)
    if normalize:
        tensor = tensor + tensor.min()
        tensor /= tensor.max()
    original_device = tensor.device
    mapped = torch.tensor(cmap(tensor.cpu())[..., :3])  # Discard alpha
    mapped = mapped.to(original_device)
    return mapped.permute(tuple(range(mapped.ndim - 3)) + (-1, -3, -2))
