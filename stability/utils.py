import logging
import random
import re
import subprocess
from abc import ABC, abstractmethod
from collections import OrderedDict
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

from stability.config import instantiate

logger = logging.getLogger(__name__)


class AffineAlignPredictions(nn.Module):
    def __init__(self, mask_nan: bool = True) -> None:
        super().__init__()
        self.mask_nan = mask_nan

    # noinspection PyMethodMayBeStatic
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> tuple:
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
    def __init__(self, cmap_name: str = "rainbow", convert_to_hex: bool = True) -> None:
        self.cmap_name = cmap_name
        self.convert_to_hex = convert_to_hex
        self.cmap = mpl.colormaps[cmap_name]
        self.colors = {}

        # Parameters used to generate sequential colors
        self.exp = 0
        self.position = 0.0
        self.step = 1.0

    def __call__(self, category: Hashable) -> tuple | str:
        if category not in self.colors.keys():
            color = self.cmap(self.position)
            if self.convert_to_hex:
                color = mpl.colors.to_hex(color)
            self.colors[category] = color

            # Update the current position within the color map (0-1).
            self.position += self.step
            if self.position > 1.0:
                self.exp += 1
                self.position = 2 ** (-self.exp)
                self.step = 2 ** (-self.exp + 1)

        return self.colors[category]


class HookableModule(nn.Module, ABC):
    def __init__(
        self,
        hook_input_index: int = 0,
        hook_selections: dict = None,
        hook_target: str = "output",
    ) -> None:
        """Initializes the module.

        Args:
            hook_input_index: The index of the input to pass to the hook. This may be
                needed if the module has multiple inputs. Defaults to 0.
            hook_selections: A dict of the form {dim_1: i_1, dim_2: i_2}, defining a set
                of Tensor.select operations. The key gives the dimension to select along
                and the value gives the index to select. Selections are not applied in
                the order specified; they are applied in reverse dim order (because
                selecting along a dim changes the index of subsequent dims). Defaults to
                None.
            hook_target: Whether to apply this hook to the parent's input or output. Can
                be "inputs" or "output". Defaults to "inputs".
        """
        super().__init__()
        valid_targets = ["inputs", "output"]
        if hook_target not in valid_targets:
            raise ValueError(
                f"Invalid hook_target '{hook_target}', options are {valid_targets}."
            )
        self.hook_input_index = hook_input_index
        if hook_selections is None:
            hook_selections = {}
        self.hook_selections = OrderedDict()
        for k in reversed(sorted(hook_selections.keys())):
            self.hook_selections[k] = hook_selections[k]
        self.hook_target = hook_target
        self.module_links = {}

    def add_link(self, assigned_name: str, destination: nn.Module):
        self.module_links[assigned_name] = destination

    def clear_links(self):
        self.module_links = {}

    @abstractmethod
    def forward(self, *args) -> torch.Tensor | None:
        """Logic that runs during the forward hook.

        Returns:
            None if the output of the module should not be modified. Otherwise, returns
                an updated tensor for the module output.
        """

    def forward_hook(self, module: nn.Module, *args) -> torch.Tensor | None:
        if self.hook_target == "inputs":
            x = args[0][self.hook_input_index]
        elif len(args) < 2:
            raise RuntimeError(
                "No output available. This may occur if this module is being used as a "
                "pre hook instead of a standard (post) hook."
            )
        else:
            x = args[1]
        for dim, i in self.hook_selections.items():
            x = x.select(dim, i)
        packed = isinstance(x, tuple) and len(x) == 1
        if packed:
            x = x[0]
        output = self(x)
        if packed and (output is not None):
            output = (output,)
        return output

    def get_hook_name(self):
        name = self.get_name()
        if self.hook_target == "inputs":
            name += "-"
            name += f"inputs_{self.hook_input_index}"
        if len(self.hook_selections) > 0:
            name += "-"
            name += ",".join(f"{dim}_{i}" for dim, i in self.hook_selections.items())
        return name

    def get_name(self):
        return self.__class__.__name__


class MeanValue:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def compute(self) -> float:
        return self.sum / self.count

    def flush(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float) -> None:
        self.sum += value
        self.count += 1


class ZippedIterators(IterableDataset):
    def __init__(self, iterators: list) -> None:
        self.iterators = iterators

    def __iter__(self):
        return self

    def __len__(self):
        return min(len(x) for x in self.iterators)

    def __next__(self) -> torch.Tensor:
        return torch.stack([next(x) for x in self.iterators])


def add_hook_modules(
    parent_module: nn.Module,
    hooked_configs: list,
    device: str | torch.device = None,
    log_keys: bool = True,
) -> dict:
    hook_module_dict = {}
    for config_item in hooked_configs:
        config_filter = config_item.get("filter", {})
        filter_name = config_filter.get("name", None)
        if "regex" in config_filter:
            filter_regex = re.compile(config_filter["regex"])
        else:
            filter_regex = None
        if "neg_regex" in config_filter:
            filter_neg_regex = re.compile(config_filter["neg_regex"])
        else:
            filter_neg_regex = None
        filter_typename = config_filter.get("typename", None)
        for name, module in parent_module.named_modules():
            # Check whether this module matches the filter.
            if isinstance(filter_name, str) and (name != filter_name):
                continue
            if isinstance(filter_name, list) and (name not in filter_name):
                continue
            if (filter_regex is not None) and (not filter_regex.search(name)):
                continue
            if (filter_neg_regex is not None) and filter_neg_regex.search(name):
                continue
            typename = type(module).__name__
            if isinstance(filter_typename, str) and (typename != filter_typename):
                continue
            if isinstance(filter_typename, list) and (typename not in filter_typename):
                continue

            # Create the module and set up hooks and links.
            hook_module = instantiate(config_item["module"])
            if device is not None:
                hook_module.to(device)
            module_links = generate_module_links(
                parent_module, name, config_item.get("links", [])
            )
            for assigned_name, destination in module_links.items():
                hook_module.add_link(assigned_name, destination)
            key = hook_module.get_hook_name()
            if "registration_name" in config_item:
                key = f"{config_item['registration_name']}-{key}"
            if len(name) > 0:
                key = f"{name}-{key}"
            hook_module_dict[key] = hook_module
            if config_item.get("pre_hook", False):
                module.register_forward_pre_hook(hook_module.forward_hook)
            else:
                module.register_forward_hook(hook_module.forward_hook)
            if "registration_name" in config_item:
                module.register_module(config_item["registration_name"], hook_module)
            if log_keys:
                logger.info(f"Added {key}.")
    return hook_module_dict


def best_pytorch_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def custom_collate(batch: list) -> tuple:
    output = []
    for i, item in enumerate(batch[0]):
        if isinstance(item, IterableDataset):
            output.append(ZippedIterators([x[i] for x in batch]))
        else:
            output.append(torch.stack([x[i] for x in batch]))
    return tuple(output)


def ensure_floating_point(*args) -> torch.Tensor | list:
    result = [(x if x.is_floating_point() else x.float()) for x in args]
    return result[0] if len(args) == 1 else result


def expand_value(shape: tuple, value: float) -> torch.Tensor:
    return torch.tensor(value).view((1,) * len(shape)).expand(shape)


def generate_module_links(parent_module: nn.Module, name: str, links: list) -> dict:
    module_links = {}
    for link in links:
        location = link["location"]
        if location[0] == ".":
            stripped = location.lstrip(".")
            up_steps = len(location) - len(stripped) - 1
            base_components = name.split(".")
            if up_steps > 0:
                base_components = base_components[:-up_steps]
            base_location = ".".join(base_components)
            if len(base_location) > 0:
                absolute_location = f"{base_location}.{stripped}"
            else:
                absolute_location = stripped
        else:
            absolute_location = location
        module_links[link["name"]] = parent_module.get_submodule(absolute_location)
    return module_links


def invoke_on_values(container: dict | list, method_name: str, **kwargs):
    values = container.values() if isinstance(container, dict) else container
    for item in values:
        getattr(item, method_name)(**kwargs)


def label_index_lookup(categories: list) -> dict:
    return {c: i for i, c in enumerate(categories)}


def list_cuda_devices():
    devices = []
    for i in range(torch.cuda.device_count()):
        devices.append(torch.cuda.get_device_properties(i).name)
    return devices


def load_clip(clip_config: dict) -> tuple:
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
    left_pad = (78 - len(content)) // 2
    right_pad = 78 - len(content) - left_pad
    return "=" * left_pad + " " + content + " " + "=" * right_pad


def pad_to_multiple(tensor: torch.Tensor, multiple: int) -> torch.Tensor:
    pad_h = -tensor.shape[-2] % multiple
    pad_w = -tensor.shape[-1] % multiple
    if pad_h != 0 or pad_w != 0:
        padding = [0, pad_w, 0, pad_h]
        tensor = nn.functional.pad(tensor, padding)
    return tensor


def save_frames(dirpath: str | Path, frames: list) -> None:
    dirpath = Path(dirpath)
    dirpath.mkdir(exist_ok=True)
    padding_width = len(str(len(frames) - 1))
    format_str = "{:0" + str(padding_width) + "d}.png"
    for j, frame in enumerate(frames):
        write_png(
            convert_dtype(frame, torch.uint8), str(dirpath / format_str.format(j))
        )
    logger.info(f"Saved frames to {dirpath.resolve()}.")


def save_video(
    filepath: str | Path,
    tensor: torch.Tensor,
    fps: float,
    crf: int = 23,
    preset: str = "medium",
) -> None:
    filepath = Path(filepath)
    tensor = convert_dtype(tensor, torch.uint8)

    # The x264 encoder fails if the height or width is not even.
    pad_h = tensor.shape[-2] % 2
    pad_w = tensor.shape[-1] % 2
    if pad_w > 0 or pad_h > 0:
        # noinspection PyTypeChecker
        tensor = pad(tensor, (0, 0, pad_w, pad_h))

    # Convert to THWC.
    tensor = tensor.permute(0, 2, 3, 1)

    # Write the video.
    write_video(str(filepath), tensor, fps, options={"crf": str(crf), "preset": preset})
    logger.info(f"Saved video to {filepath.resolve()}.")


def set_random_seeds(seed: int) -> None:
    # https://pytorch.org/docs/stable/notes/randomness.html
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
    valid_modes = ["hstack", "vstack"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}', options are {valid_modes}.")

    # Construct a ffmpeg filter graph that scales videos along one axis to a uniform
    # size, then stacks them along the other axis.
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

    # Construct the ffmpeg argument list.
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


def verify_invocation_count(stabilizer_dict: dict, expected: int) -> None:
    for name, stabilizer in stabilizer_dict.items():
        actual = stabilizer.counter
        if not stabilizer.enabled:
            if actual != 0:
                raise RuntimeError(
                    f'"{name}" was invoked {actual} times but was disabled.'
                )
        elif stabilizer.counter != expected:
            raise RuntimeError(
                f'"{name}" was invoked {actual} times (expected {expected}).'
            )


def visualize_with_cmap(
    tensor: torch.Tensor, cmap_name: str = "viridis", normalize: bool = True
) -> torch.Tensor:
    cmap = plt.get_cmap(cmap_name)
    if normalize:
        # Adjust values so they fill the range [0, 1].
        tensor = tensor + tensor.min()
        tensor /= tensor.max()
    original_device = tensor.device
    mapped = torch.tensor(cmap(tensor.cpu())[..., :3])  # Discard alpha.
    mapped = mapped.to(original_device)
    return mapped.permute(tuple(range(mapped.ndim - 3)) + (-1, -3, -2))
