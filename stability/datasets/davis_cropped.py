"""Defines the cropped DAVIS classification dataset.

This dataset is used for adversarial robustness experiments.
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from stability.datasets.utils import FrameIterator

SPLIT_OPTIONS = ["train", "val"]


class DAVISCropped(Dataset):
    """The cropped DAVIS classification dataset.

    This dataset is used for adversarial robustness experiments.
    """

    def __init__(
        self,
        location: str | Path,
        frame_transform: Callable = None,
        random_chunk_size: int = None,
        split: str = "val",
        split_seed: int = 42,
    ) -> None:
        """Creates the dataset.

        Args:
            location: The path where data is stored (like "data/vision_sim").
            frame_transform: A transform operation to apply to each frame as it is
                loaded. Defaults to None.
            random_chunk_size: If this is not None, randomly select temporal chunks from
                each video instead of reading the whole thing. The integer value gives
                the number of consecutive frames in the chunk. Defaults to None.
            split: The dataset split ("train" or "val"). Defaults to "val".
            split_seed: The random seed to use when generating the random dataset split.
                Defaults to 42.
        """

        if split not in SPLIT_OPTIONS:
            raise ValueError(f"Invalid split '{split}'; options are {SPLIT_OPTIONS}.")
        self.frame_transform = frame_transform
        self.random_chunk_size = random_chunk_size
        video_paths = [subdir for subdir in Path(location).iterdir() if subdir.is_dir()]

        # Randomly shuffle and split the dataset (using the seed)
        rng = random.Random(split_seed)
        rng.shuffle(video_paths)
        split_index = int(0.8 * len(video_paths))
        if split == "train":
            selected_video_paths = video_paths[:split_index]
        else:
            selected_video_paths = video_paths[split_index:]

        # Enumerate object sequences
        self.video_groups = []
        for video_path in sorted(selected_video_paths):
            object_groups = defaultdict(list)
            for image_filepath in sorted(video_path.glob("*.png")):
                parts = image_filepath.stem.split("_")
                frame_number = int(parts[0])
                object_id = "_".join(parts[2:])[:-4]
                object_groups[object_id].append(
                    {"frame_number": frame_number, "filepath": image_filepath}
                )
            for object_id, frames in object_groups.items():
                sorted_frames = sorted(frames, key=lambda x: x["frame_number"])
                frame_paths = [frame["filepath"] for frame in sorted_frames]
                parts = frame_paths[0].stem.split("_")
                object_id = 1 if "person" in parts[-2] else 0
                self.video_groups.append((frame_paths, object_id))

    def __getitem__(self, index: int) -> tuple:
        """Gets an item from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A tuple (frame_iterator, label_tensor), where label_tensor is a tensor of
                class index labels.
        """
        frame_paths, object_id = self.video_groups[index]
        if (
            self.random_chunk_size is not None
            and len(frame_paths) >= self.random_chunk_size
        ):
            chunk_start = torch.randint(
                len(frame_paths) - self.random_chunk_size + 1, ()
            )
            frame_paths = frame_paths[
                chunk_start : chunk_start + self.random_chunk_size
            ]
        label_tensor = torch.tensor([object_id] * len(frame_paths), dtype=torch.int64)
        frame_iterator = FrameIterator(
            frame_paths, frame_transform=self.frame_transform
        )
        return frame_iterator, label_tensor

    def __len__(self) -> int:
        """Gets the number of items in the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.video_groups)
