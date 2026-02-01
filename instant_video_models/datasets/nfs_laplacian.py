"""Defines the NFS Laplacian dataset.

This data is used for image enhancement experiments.
"""

from pathlib import Path
from typing import Callable

from instant_video_models.datasets.nfs import NFSVideo
from instant_video_models.datasets.utils import FrameIterator, aligned_torch_rngs


class NFSLaplacian(NFSVideo):
    """The NFS Laplacian dataset.

    This data is used for image enhancement experiments.
    """

    def __init__(
        self,
        nfs_location: str | Path,
        laplacian_location: str | Path,
        ending_frame: int = None,
        fps: int = 240,
        frame_transform: Callable = None,
        label_transform: Callable = None,
        random_chunk_size: int = None,
        random_crop_size: tuple = None,
        split_subset: list = None,
    ) -> None:
        """Creates the dataset.

        Args:
            nfs_location: The path where the original NFS data is stored (like
                "data/nfs").
            laplacian_location: The path where the Laplacian-transformed images are
                stored (like "data/nfs_laplacian_strong").
            ending_frame: If this is not None, truncate all videos to this number of
                frames. Defaults to None.
            fps: The FPS variant to use (30 or 240). Defaults to 240.
            frame_transform: A transform operation to apply to each original (non-
                Laplacian) frame as it is loaded. Defaults to None.
            label_transform: A transform operation to apply to each Laplacian-
                transformed frame as it is loaded. Defaults to None.
            random_chunk_size: If this is not None, randomly select temporal chunks from
                each video instead of reading the whole thing. The integer value gives
                the number of consecutive frames in the chunk. Defaults to None.
            random_crop_size: If this is not None, randomly crop sequences to a
                rectangle of this size. The random crop is consistent over each returned
                sequence. Defaults to None.
            split_subset: The subset of sequences to use. This is used for producing
                consistent training and validation splits. Defaults to None.
        """
        super().__init__(
            nfs_location,
            ending_frame=ending_frame,
            fps=fps,
            frame_transform=frame_transform,
            random_chunk_size=random_chunk_size,
            random_crop_size=random_crop_size,
            split_subset=split_subset,
        )
        self.labels_path = Path(laplacian_location)
        self.label_transform = label_transform

    def __getitem__(self, index: int) -> tuple:
        """Gets an item from the dataset.

        Args:
            index: The index of the item to retrieve.

        Returns:
            A tuple (frame_iterator, label_iterator) containing iterators over the
                original (non-Laplacian) and Laplacian-transformed images, respectively.
        """
        frame_paths = self._list_frame_paths(index)
        label_paths = [
            self.labels_path / f.relative_to(self.videos_path) for f in frame_paths
        ]
        rng_1, rng_2 = aligned_torch_rngs(2)
        frame_iterator = FrameIterator(
            frame_paths,
            frame_transform=self.frame_transform,
            random_crop_size=self.random_crop_size,
            crop_rng=rng_1,
        )
        label_iterator = FrameIterator(
            label_paths,
            frame_transform=self.label_transform,
            random_crop_size=self.random_crop_size,
            crop_rng=rng_2,
        )
        return frame_iterator, label_iterator

    def __len__(self) -> int:
        """Gets the number of items in the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.video_ids)
