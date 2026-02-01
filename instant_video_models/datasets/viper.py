"""Defines the VIPER (GTA) segmentation dataset."""

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from instant_video_models.datasets.utils import (
    FrameIterator,
    SegmentationIterator,
    aligned_torch_rngs,
)

SPLIT_OPTIONS = ["train", "val"]

# https://github.com/srrichter/viper/blob/master/classes.csv
CLASS_INDICES = {
    "unlabeled": 0,
    "ambiguous": 1,
    "sky": 2,
    "road": 3,
    "sidewalk": 4,
    "railtrack": 5,
    "terrain": 6,
    "tree": 7,
    "vegetation": 8,
    "building": 9,
    "infrastructure": 10,
    "fence": 11,
    "billboard": 12,
    "trafficlight": 13,
    "trafficsign": 14,
    "mobilebarrier": 15,
    "firehydrant": 16,
    "chair": 17,
    "trash": 18,
    "trashcan": 19,
    "person": 20,
    "animal": 21,
    "bicycle": 22,
    "motorcycle": 23,
    "car": 24,
    "van": 25,
    "bus": 26,
    "truck": 27,
    "trailer": 28,
    "train": 29,
    "plane": 30,
    "boat": 31,
}
COLOR_MAPPING = {
    "unlabeled": (0, 0, 0),
    "ambiguous": (111, 74, 0),
    "sky": (70, 130, 180),
    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "railtrack": (230, 150, 140),
    "terrain": (152, 251, 152),
    "tree": (87, 182, 35),
    "vegetation": (35, 142, 35),
    "building": (70, 70, 70),
    "infrastructure": (153, 153, 153),
    "fence": (190, 153, 153),
    "billboard": (150, 20, 20),
    "trafficlight": (250, 170, 30),
    "trafficsign": (220, 220, 0),
    "mobilebarrier": (180, 180, 100),
    "firehydrant": (173, 153, 153),
    "chair": (168, 153, 153),
    "trash": (81, 0, 21),
    "trashcan": (81, 0, 81),
    "person": (220, 20, 60),
    "animal": (255, 0, 0),
    "bicycle": (119, 11, 32),
    "motorcycle": (0, 0, 230),
    "car": (0, 0, 142),
    "van": (0, 80, 100),
    "bus": (0, 60, 100),
    "truck": (0, 0, 70),
    "trailer": (0, 0, 90),
    "train": (0, 80, 100),
    "plane": (0, 100, 100),
    "boat": (50, 0, 90),
}
CLASS_EVAL = {
    "unlabeled": 0,
    "ambiguous": 0,
    "sky": 1,
    "road": 1,
    "sidewalk": 1,
    "railtrack": 0,
    "terrain": 1,
    "tree": 1,
    "vegetation": 1,
    "building": 1,
    "infrastructure": 1,
    "fence": 1,
    "billboard": 1,
    "trafficlight": 1,
    "trafficsign": 1,
    "mobilebarrier": 1,
    "firehydrant": 1,
    "chair": 1,
    "trash": 1,
    "trashcan": 1,
    "person": 1,
    "animal": 0,
    "bicycle": 0,
    "motorcycle": 1,
    "car": 1,
    "van": 1,
    "bus": 1,
    "truck": 1,
    "trailer": 0,
    "train": 0,
    "plane": 0,
    "boat": 0,
}
TRAIN_INDICES = {
    "unlabeled": 255,
    "ambiguous": 255,
    "sky": 0,
    "road": 1,
    "sidewalk": 2,
    "railtrack": 255,
    "terrain": 3,
    "tree": 4,
    "vegetation": 5,
    "building": 6,
    "infrastructure": 7,
    "fence": 8,
    "billboard": 9,
    "trafficlight": 10,
    "trafficsign": 11,
    "mobilebarrier": 12,
    "firehydrant": 13,
    "chair": 14,
    "trash": 15,
    "trashcan": 16,
    "person": 17,
    "animal": 255,
    "bicycle": 255,
    "motorcycle": 18,
    "car": 19,
    "van": 20,
    "bus": 21,
    "truck": 22,
    "trailer": 255,
    "train": 255,
    "plane": 255,
    "boat": 255,
}

TRAIN_INDICES_SQUASHED = {k: (v if v < 255 else 23) for k, v in TRAIN_INDICES.items()}


class VIPERSegmentation(Dataset):
    """The VIPER (GTA) segmentation dataset."""

    def __init__(
        self,
        location: str | Path,
        ending_frame: int = None,
        frame_transform: Callable = None,
        random_chunk_size: int = None,
        random_crop_size: int = None,
        split: str = "train",
    ) -> None:
        """Creates the dataset.

        Args:
            location: The path where the data is stored (like "data/viper").
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

        # Enumerate sequences
        split_dirpath = Path(location, split)
        img_dirpath = split_dirpath / "img"
        cls_dirpath = split_dirpath / "cls"
        self.video_dirpaths = []
        self.segmentation_dirpaths = []
        for video_dirpath in sorted(img_dirpath.iterdir()):
            if not video_dirpath.is_dir():
                continue
            self.video_dirpaths.append(video_dirpath)
            self.segmentation_dirpaths.append(
                cls_dirpath / video_dirpath.relative_to(img_dirpath)
            )

    def __getitem__(self, index: int) -> tuple:
        """Gets an item from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A tuple (frame_iterator, segmentation_iterator) containing iterators over
                the input frames and segmentation masks, respectively.
        """
        frame_paths = []
        segmentation_paths = []
        for frame_path, segmentation_path in zip(
            sorted(self.video_dirpaths[index].iterdir()),
            sorted(self.segmentation_dirpaths[index].iterdir()),
        ):
            # Skip any items with zero size (there is one such item in VIPER)
            if frame_path.stat().st_size > 0 and segmentation_path.stat().st_size > 0:
                frame_paths.append(frame_path)
                segmentation_paths.append(segmentation_path)
        if self.ending_frame is not None:
            frame_paths = frame_paths[: self.ending_frame]
            segmentation_paths = segmentation_paths[: self.ending_frame]
        if self.random_chunk_size is not None:
            start = torch.randint(len(frame_paths) - self.random_chunk_size + 1, ())
            frame_paths = frame_paths[start : start + self.random_chunk_size]
            segmentation_paths = segmentation_paths[
                start : start + self.random_chunk_size
            ]
        rng_1, rng_2 = aligned_torch_rngs(2)
        frame_iterator = FrameIterator(
            frame_paths,
            frame_transform=self.frame_transform,
            random_crop_size=self.random_crop_size,
            crop_rng=rng_1,
        )
        segmentation_iterator = SegmentationIterator(
            segmentation_paths,
            {COLOR_MAPPING[k]: TRAIN_INDICES_SQUASHED[k] for k in COLOR_MAPPING},
            random_crop_size=self.random_crop_size,
            crop_rng=rng_2,
        )
        return frame_iterator, segmentation_iterator

    def __len__(self) -> int:
        """Gets the number of items in the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.video_dirpaths)
