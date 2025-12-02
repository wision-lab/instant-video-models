"""Defines the NFS video dataset (video only, no labels).

This dataset is used for denoising experiments.
"""

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from stability.datasets.utils import FrameIterator

FPS_OPTIONS = [30, 240]


class NFSVideo(Dataset):
    """The NFS video dataset (video only, no labels).

    This dataset is used for denoising experiments.
    """

    def __init__(
        self,
        location: str | Path,
        ending_frame: int = None,
        fps: int = 240,
        frame_transform: Callable = None,
        random_chunk_size: int = None,
        random_crop_size: tuple = None,
        split_subset: list = None,
    ) -> None:
        """Creates the dataset.

        Args:
            location: The path where the data is stored (like "data/nfs").
            ending_frame: If this is not None, truncate all videos to this number of
                frames. Defaults to None.
            fps: The FPS variant to use (30 or 240). Defaults to 240.
            frame_transform: A transform operation to apply to each frame as it is
                loaded. Defaults to None.
            random_chunk_size: If this is not None, randomly select temporal chunks from
                each video instead of reading the whole thing. The integer value gives
                the number of consecutive frames in the chunk. Defaults to None.
            random_crop_size: If this is not None, randomly crop sequences to a
                rectangle of this size. The random crop is consistent over each returned
                sequence. Defaults to None.
            split_subset: The subset of sequences to use. This is used for producing
                consistent training and validation splits. Defaults to None.
        """
        if fps not in FPS_OPTIONS:
            raise ValueError(f"Invalid fps value '{fps}'; options are {FPS_OPTIONS}.")
        self.ending_frame = ending_frame
        self.fps = fps
        self.frame_transform = frame_transform
        self.random_chunk_size = random_chunk_size
        self.random_crop_size = random_crop_size

        # Enumerate sequences
        self.videos_path = Path(location)
        self.video_ids = []
        for video_path in sorted(self.videos_path.iterdir()):
            if (not video_path.is_dir()) or (video_path.name == "__MACOSX"):
                continue
            video_id = video_path.stem
            if (split_subset is not None) and (video_id not in split_subset):
                continue
            self.video_ids.append(video_id)

    def __getitem__(self, index: int) -> tuple:
        """Gets an item from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A tuple (frame_iterator,) containing an iterator over a sequence.
        """
        frame_iterator = FrameIterator(
            self._list_frame_paths(index),
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

    def _list_frame_paths(self, index: int) -> list:
        video_id = self.video_ids[index]
        frame_paths = sorted(
            (self.videos_path / video_id / str(self.fps) / video_id).glob("*.jpg")
        )
        if self.ending_frame is not None:
            frame_paths = frame_paths[: self.ending_frame]
        if self.random_chunk_size is not None:
            start = torch.randint(len(frame_paths) - self.random_chunk_size + 1, ())
            frame_paths = frame_paths[start : start + self.random_chunk_size]
        return frame_paths
