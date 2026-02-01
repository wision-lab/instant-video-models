"""Defines a simple tensor value monitor."""

from collections import deque

import torch

from instant_video_models.monitors.base import Monitor


class ValueMonitor(Monitor):
    """A simple tensor value monitor."""

    def __init__(
        self,
        duration: int = 1,
        hook_input_index: int = 0,
        hook_selections: dict = None,
        hook_target: str = "output",
    ) -> None:
        """Initializes the module.

        Args:
            duration: The number of time steps that should tracked (size of the memory
                queue). Defaults to 1.
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
        super().__init__(hook_input_index, hook_selections, hook_target)
        self.duration = duration

        # History queue
        self.memory = deque()

    def available(self) -> bool:
        """
        Returns:
            True if forward() has been called at least "duration" times since the last
                reset().
        """
        return len(self.memory) >= self.duration

    def forward(self, x: torch.Tensor) -> None:
        """Stores the current input in the history queue.

        Pops off the oldest element if the queue is full.

        Args:
            x: The input to be stored in the queue.
        """
        self.memory.append(x.clone())
        if len(self.memory) > self.duration:
            self.memory.popleft()

    def reset(self) -> None:
        """Resets the history queue."""
        self.memory.clear()

    def retrieve(self) -> torch.Tensor | list:
        result = [m.clone() for m in self.memory]
        return result[0] if (self.duration == 1) else result
