"""Defines the VisionSim depth dataset."""

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from stability.datasets.utils import DepthIterator, FrameIterator, aligned_torch_rngs


class VisionSim(Dataset):
    """The VisionSim depth dataset."""

    def __init__(
        self,
        location: str | Path,
        depth_nan_threshold: float = 1e9,
        ending_frame: int = None,
        frame_transform: Callable = None,
        invert_depth: bool = True,
        random_chunk_size: int = None,
        random_crop_size: tuple = None,
        split_subset: list = None,
    ) -> None:
        """Creates the dataset.

        Args:
            location: The path where the data is stored (like "data/vision_sim").
            depth_nan_threshold: The upper limit for depth values. Above this value,
                depths will be set to nan. Defaults to 1e9.
            ending_frame: If this is not None, truncate all videos to this number of
                frames. Defaults to None.
            frame_transform: A transform operation to apply to each frame as it is
                loaded. Defaults to None.
            invert_depth: Whether depths should be inverted. Defaults to True.
            random_chunk_size: If this is not None, randomly select temporal chunks from
                each video instead of reading the whole thing. The integer value gives
                the number of consecutive frames in the chunk. Defaults to None.
            random_crop_size: If this is not None, randomly crop sequences to a
                rectangle of this size. The random crop is consistent over each returned
                sequence. Defaults to None.
            split_subset: The subset of sequences to use. This is used for producing
                consistent training and validation splits. Defaults to None.
        """
        self.seq_paths = []
        for scene_dir in Path(location, "renders").iterdir():
            if (split_subset is not None) and (scene_dir.stem not in split_subset):
                continue
            for seq_dir in scene_dir.iterdir():
                self.seq_paths.append(seq_dir)
        self.seq_paths.sort()
        self.depth_nan_threshold = depth_nan_threshold
        self.ending_frame = ending_frame
        self.frame_transform = frame_transform
        self.invert_depth = invert_depth
        self.random_chunk_size = random_chunk_size
        self.random_crop_size = random_crop_size

    def __getitem__(self, index: int) -> tuple:
        """Gets an item from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A tuple (frame_iterator, depth_iterator) containing iterators over the input
                frames and depth maps, respectively.
        """
        frame_paths = sorted((self.seq_paths[index] / "frames").glob("*.png"))
        depth_paths = sorted((self.seq_paths[index] / "depths").glob("*.exr"))
        if self.ending_frame is not None:
            frame_paths = frame_paths[: self.ending_frame]
            depth_paths = depth_paths[: self.ending_frame]
        if self.random_chunk_size is not None:
            start = torch.randint(len(frame_paths) - self.random_chunk_size + 1, ())
            frame_paths = frame_paths[start : start + self.random_chunk_size]
            depth_paths = depth_paths[start : start + self.random_chunk_size]
        rng_1, rng_2 = aligned_torch_rngs(2)
        frame_iterator = FrameIterator(
            frame_paths,
            frame_transform=self.frame_transform,
            random_crop_size=self.random_crop_size,
            crop_rng=rng_1,
        )
        depth_iterator = DepthIterator(
            depth_paths,
            invert=self.invert_depth,
            nan_threshold=self.depth_nan_threshold,
            random_crop_size=self.random_crop_size,
            crop_rng=rng_2,
        )
        return frame_iterator, depth_iterator

    def __len__(self) -> int:
        """Gets the number of items in the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.seq_paths)
