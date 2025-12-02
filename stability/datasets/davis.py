"""Defines the DAVIS video dataset (video only, no labels).

This dataset is used for denoising experiments.
"""

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from stability.datasets.utils import FrameIterator

SPLIT_OPTIONS = ["train", "val"]


class DAVIS(Dataset):
    """The DAVIS video dataset (video only, no labels).

    This dataset is used for denoising experiments.
    """

    def __init__(
        self,
        location: str | Path,
        ending_frame: int = None,
        frame_transform: Callable = None,
        random_chunk_size: int = None,
        random_crop_size: tuple = None,
        split: str = "train",
    ):
        """Creates the dataset.

        Args:
            location: The path where the data is stored (like "data/davis_2017").
            ending_frame: If this is not None, truncate all videos to this number of
                frames. Defaults to None.
            frame_transform: A transform operation to apply to each frame as it is
                loaded. Defaults to None.
            random_chunk_size: If this is not None, randomly select temporal chunks from
                each video instead of reading the whole thing. The integer value gives
                the number of consecutive frames in the chunk. Defaults to None.
            random_crop_size: If this is not None, randomly crop sequences to a
                rectangle of this size. The random crop is consistent over each returned
                sequence. Defaults to None.
            split: The dataset split ("train" or "val"). Defaults to "train".
        """
        if split not in SPLIT_OPTIONS:
            raise ValueError(
                f"Invalid split value '{split}'; options are {SPLIT_OPTIONS}"
            )
        self.ending_frame = ending_frame
        self.frame_transform = frame_transform
        self.random_chunk_size = random_chunk_size
        self.random_crop_size = random_crop_size

        # Determine which sequences are in the split
        base_path = Path(location, "DAVIS")
        with open(base_path / "ImageSets" / "2017" / f"{split}.txt") as split_file:
            split_subset = set()
            for line in split_file:
                split_subset.add(line.strip())

        # Enumerate sequences
        self.videos_path = base_path / "JPEGImages" / "Full-Resolution"
        self.video_ids = []
        for video_path in sorted(self.videos_path.iterdir()):
            video_id = video_path.stem
            if video_id in split_subset:
                self.video_ids.append(video_id)

    def __getitem__(self, index: int) -> tuple:
        """Gets an item from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A tuple (frame_iterator,) containing an iterator over a sequence.
        """
        video_id = self.video_ids[index]
        frame_paths = sorted((self.videos_path / video_id).glob("*.jpg"))
        if self.ending_frame is not None:
            frame_paths = frame_paths[: self.ending_frame]
        if self.random_chunk_size is not None:
            start = torch.randint(len(frame_paths) - self.random_chunk_size + 1, ())
            frame_paths = frame_paths[start : start + self.random_chunk_size]
        frame_iterator = FrameIterator(
            frame_paths,
            frame_transform=self.frame_transform,
            random_crop_size=self.random_crop_size,
        )
        return (frame_iterator,)

    def __len__(self) -> int:
        """Gets the number of items in the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.video_ids)
