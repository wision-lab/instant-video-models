"""Defines the RobustSpring dataset."""

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from instant_video_models.datasets.utils import FrameIterator, aligned_torch_rngs


class RobustSpring(Dataset):
    """The RobustSpring dataset."""

    def __init__(
        self,
        location: str | Path,
        clean_transform: Callable = None,
        corrupted_transform: Callable = None,
        corruption_type: str = None,
        random_chunk_size: int = None,
        random_crop_size: int = None,
        split_subset: list = None,
    ) -> None:
        """Creates the dataset.

        Args:
            location: The path where the data is stored (like "data/robust_robust_spring").
            clean_transform: A transform operation to apply to each clean (uncorrupted)
                frame as it is loaded. Defaults to None.
            corrupted_transform: A transform operation to apply to each corrupted frame
                as it is loaded. Defaults to None.
            corruption_type: The type of corruption ("fog", "frost", "rain", "snow", or
                "spatter"). If None, only load clean sequences. Defaults to None.
            random_chunk_size: If this is not None, randomly select temporal chunks from
                each video instead of reading the whole thing. The integer value gives
                the number of consecutive frames in the chunk. Defaults to None.
            random_crop_size: If this is not None, randomly crop sequences to a
                rectangle of this size. The random crop is consistent over each returned
                sequence. Defaults to None.
            split_subset: The subset of sequences to use. This is used for producing
                consistent training and validation splits. Defaults to None.
        """
        self.clean_transform = clean_transform
        self.corrupted_transform = corrupted_transform
        self.random_chunk_size = random_chunk_size
        self.random_crop_size = random_crop_size

        # Enumerate sequences
        self.clean_path = Path(location, "robust_spring", "test")
        if corruption_type is None:
            self.corrupted_path = None
        else:
            self.corrupted_path = Path(location, corruption_type, "test")
        self.video_ids = []
        for video_path in sorted(self.clean_path.iterdir()):
            video_id = video_path.stem
            if (split_subset is not None) and (video_id not in split_subset):
                continue
            self.video_ids.append(Path(video_id, "frame_right"))
            self.video_ids.append(Path(video_id, "frame_left"))

    def __getitem__(self, index: int) -> tuple:
        """Gets an item from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A tuple (clean_iterator, corrupted_iterator) containing iterators over the
                clean (uncorrupted) and corrupted images, respectively. If the
                corruption_type is None, then the returned tuple contains only the first
                iterator is returned.
        """
        # Always return an iterator over clean frames
        video_id = self.video_ids[index]
        clean_paths = sorted((self.clean_path / video_id).iterdir())
        if self.random_chunk_size is not None:
            start = torch.randint(len(clean_paths) - self.random_chunk_size + 1, ())
            clean_paths = clean_paths[start : start + self.random_chunk_size]
        else:
            start = 0
        rng_1, rng_2 = aligned_torch_rngs(2)
        clean_iterator = FrameIterator(
            clean_paths,
            frame_transform=self.clean_transform,
            random_crop_size=self.random_crop_size,
            crop_rng=rng_2,
        )

        # Only return a corruption iterator if a corruption type was given
        if self.corrupted_path is None:
            return (clean_iterator,)
        else:
            corrupted_paths = sorted((self.corrupted_path / video_id).iterdir())
            if self.random_chunk_size is not None:
                corrupted_paths = corrupted_paths[
                    start : start + self.random_chunk_size
                ]
            corrupted_iterator = FrameIterator(
                corrupted_paths,
                frame_transform=self.corrupted_transform,
                random_crop_size=self.random_crop_size,
                crop_rng=rng_1,
            )
            return corrupted_iterator, clean_iterator

    def __len__(self) -> int:
        """Gets the number of items in the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.video_ids)
