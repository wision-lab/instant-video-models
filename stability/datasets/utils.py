"""Defines dataset utilities."""

from typing import Callable

import OpenEXR
import torch
from torch.utils.data import IterableDataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.v2.functional import convert_image_dtype as convert_dtype


class DepthIterator(IterableDataset):
    """An iterator over EXR depth files."""

    def __init__(
        self,
        exr_paths: list,
        invert: bool = True,
        nan_threshold: float = None,
        random_crop_size: tuple = None,
        crop_rng: torch.Generator = None,
    ) -> None:
        """Initializes the iterator.

        Args:
            exr_paths: A list of depth filepaths to load (in the provided order).
            crop_rng: Overrides the rng used to generate random crops. This can be used
                to get consistent crops between multiple FrameIterator instances.
                Defaults to None.
            invert: Whether to invert depths after loading. Defaults to True.
            nan_threshold: The upper limit for depth values. Above this value, depths
                will be set to nan. Defaults to None.
            random_crop_size: If this is not None, randomly crop sequences to a
                rectangle of this size. The random crop is consistent over each returned
                sequence. Defaults to None.
        """
        self.exr_paths = exr_paths
        self.crop_rng = crop_rng
        self.invert = invert
        self.nan_threshold = nan_threshold
        self.random_crop_size = random_crop_size

        # Current position in the sequence
        self.index = 0

        # If this is not None, use it for the random crop instead of re-generating one
        self.random_crop_offset = None

    def __iter__(self):
        """Returns an iterator object.

        Returns:
            This DepthIterator as a Python iterator.
        """
        return self

    def __len__(self):
        """Gets the number of depth maps in this sequence.

        Returns:
            The length of the sequence.
        """
        return len(self.exr_paths)

    def __next__(self) -> torch.Tensor:
        """Gets the next item in the sequence.

        Raises:
            StopIteration: If no more items remain.

        Returns:
            A tensor for the next depth map in the sequence.
        """
        if self.index >= len(self.segmentation_paths):
            raise StopIteration
        with OpenEXR.File(str(self.exr_paths[self.index])) as exr_file:
            self.index += 1
            first_channel = list(exr_file.channels().values())[0]
            depth = torch.tensor(first_channel.pixels)
        if self.invert:
            depth = 1.0 / depth
        if self.nan_threshold is not None:
            depth[depth > self.nan_threshold] = torch.nan
        if self.random_crop_size is not None:
            h, w = self.random_crop_size
            if self.random_crop_offset is None:
                self.random_crop_offset = (
                    torch.randint(depth.shape[-2] - h + 1, (), generator=self.crop_rng),
                    torch.randint(depth.shape[-1] - w + 1, (), generator=self.crop_rng),
                )
            y, x = self.random_crop_offset
            depth = depth[..., y : y + h, x : x + w]
        return depth


class FrameIterator(IterableDataset):
    """An iterator over image files."""

    def __init__(
        self,
        frame_paths: list,
        convert_image_dtype: str = "float32",
        crop_rng: torch.Generator = None,
        frame_transform: Callable = None,
        random_crop_size: tuple = None,
    ) -> None:
        """Initializes the iterator.

        Args:
            frame_paths: A list of image filepaths to load (in the provided order).
            convert_image_dtype: If this is not None, convert image tensors to the
                specified data type. This may perform some scaling (1/255 if converting
                uint8 -> float32). Defaults to "float32".
            crop_rng: Overrides the rng used to generate random crops. This can be used
                to get consistent crops between multiple FrameIterator instances.
                Defaults to None.
            frame_transform: A transform operation to apply to each frame as it is
                loaded. Defaults to None.
            random_crop_size: If this is not None, randomly crop sequences to a
                rectangle of this size. The random crop is consistent over each returned
                sequence. Defaults to None.
        """
        self.frame_paths = frame_paths
        self.convert_image_dtype = convert_image_dtype
        self.crop_rng = crop_rng
        self.frame_transform = frame_transform
        self.random_crop_size = random_crop_size

        # Current position in the sequence
        self.index = 0

        # If this is not None, use it for the random crop instead of re-generating one
        self.random_crop_offset = None

    def __iter__(self):
        """Returns an iterator object.

        Returns:
            This FrameIterator as a Python iterator.
        """
        return self

    def __len__(self):
        """Gets the number of frames in this sequence.

        Returns:
            The length of the sequence.
        """
        return len(self.frame_paths)

    def __next__(self) -> torch.Tensor:
        """Gets the next item in the sequence.

        Raises:
            StopIteration: If no more items remain.

        Returns:
            A tensor for the next frame in the sequence.
        """
        if self.index >= len(self.frame_paths):
            raise StopIteration

        frame = read_image(str(self.frame_paths[self.index]))
        self.index += 1

        # Discard the alpha channel
        if (frame.ndim == 3) and (frame.shape[0] == 4):
            frame = frame[:3]

        if self.convert_image_dtype is not None:
            frame = convert_dtype(frame, getattr(torch, self.convert_image_dtype))
        if self.frame_transform is not None:
            frame = self.frame_transform(frame)
        if self.random_crop_size is not None:
            h, w = self.random_crop_size
            if self.random_crop_offset is None:
                self.random_crop_offset = (
                    torch.randint(frame.shape[-2] - h + 1, (), generator=self.crop_rng),
                    torch.randint(frame.shape[-1] - w + 1, (), generator=self.crop_rng),
                )
            y, x = self.random_crop_offset
            frame = frame[..., y : y + h, x : x + w]
        return frame


class SegmentationIterator(IterableDataset):
    """An iterator over segmentation masks."""

    def __init__(
        self,
        segmentation_paths: list,
        color_mapping: dict,
        crop_rng: torch.Generator = None,
        random_crop_size: tuple = None,
    ) -> None:
        """Constructs the iterator.

        Args:
            segmentation_paths: A list of segmentation mask filepaths to load (in the
                provided order).
            color_mapping: A dictionary mapping from colors to class indices.
            crop_rng: Overrides the rng used to generate random crops. This can be used
                to get consistent crops between multiple FrameIterator instances.
                Defaults to None.
            random_crop_size: If this is not None, randomly crop sequences to a
                rectangle of this size. The random crop is consistent over each returned
                sequence. Defaults to None.
        """
        self.segmentation_paths = segmentation_paths
        self.color_mapping = color_mapping
        self.crop_rng = crop_rng
        self.random_crop_size = random_crop_size

        # Current position in the sequence
        self.index = 0

        # If this is not None, use it for the random crop instead of re-generating one
        self.random_crop_offset = None

    def __iter__(self):
        """Returns an iterator object.

        Returns:
            This SegmentationIterator as a Python iterator.
        """
        return self

    def __len__(self):
        """Gets the number of segmentation masks in this sequence.

        Returns:
            The length of the sequence.
        """
        return len(self.segmentation_paths)

    def __next__(self) -> torch.Tensor:
        """Gets the next item in the sequence.

        Raises:
            StopIteration: If no more items remain.

        Returns:
            A tensor for the next segmentation mask in the sequence.
        """
        if self.index >= len(self.segmentation_paths):
            raise StopIteration
        segmentation = read_image(
            str(self.segmentation_paths[self.index]), mode=ImageReadMode.RGB
        )
        self.index += 1
        if self.random_crop_size is not None:
            h, w = self.random_crop_size
            if self.random_crop_offset is None:
                self.random_crop_offset = (
                    torch.randint(
                        segmentation.shape[-2] - h + 1, (), generator=self.crop_rng
                    ),
                    torch.randint(
                        segmentation.shape[-1] - w + 1, (), generator=self.crop_rng
                    ),
                )
            y, x = self.random_crop_offset
            segmentation = segmentation[..., y : y + h, x : x + w]

        # This is a little convoluted because the simple way is *slow*. Flatten
        # (uint8, uint8, uint8) -> int32, then use bucketize to recover indices.
        segmentation = segmentation.int()
        segmentation = (
            segmentation[0] * (256**2) + segmentation[1] * 256 + segmentation[2]
        )
        color_mapping = {
            c[0] * (256**2) + c[1] * 256 + c[2]: i
            for c, i in self.color_mapping.items()
        }
        buckets = sorted(color_mapping.keys())
        class_lookup = [color_mapping[c] for c in buckets]
        indices = torch.bucketize(segmentation, torch.tensor(buckets))
        segmentation = torch.tensor(class_lookup)[indices]

        return segmentation


def aligned_torch_rngs(count: int) -> tuple:
    """Returns multiple torch random generators with the same state/seed.

    Args:
        count: The number of random generators to create.

    Returns:
        A tuple consisting of `count` generators, all with the same seed.
    """
    seed = int(torch.randint(low=-(2**31), high=2**31, size=(), dtype=torch.int32))
    output = []
    for _ in range(count):
        rng = torch.Generator()
        rng.manual_seed(seed)
        output.append(rng)
    return tuple(output)
