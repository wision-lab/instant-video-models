"""Defines metric base classes."""

from abc import ABC, abstractmethod

import torch


class StreamingMetric(ABC):
    """A metric designed to be invoked sequentially on individual time steps.

    Example usage:
    ```
    pred = torch.rand((b, t, h, w, c))
    true = torch.rand((b, t, h, w, c))
    metric.reset()
    for i in range(t):
        metric.update(pred[:, i], true[:, i])
    result = metric.compute()
    ```
    """

    @abstractmethod
    def available(self) -> bool:
        """Determines whether a value for this metric is available.

        This method should be called before compute().

        Returns:
            True if a value is available, False otherwise.
        """

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """Computes and returns the value of the metric.

        The internal state of the metric does not change. Subsequent calls to compute()
        will return the same value.

        Returns:
            A tensor containing the metric results.
        """

    def flush(self) -> None:
        """Clears internal long-term state.

        Intended to be called between dataset passes. The specific logic is defined by
        the child class.
        """

    def get_name(self) -> str:
        """Returns the class name as the metric name.

        This method can be overriden to enable custom metric naming.

        Returns:
            The class name.
        """
        return self.__class__.__name__

    def reset(self) -> None:
        """Clears internal short-term state.

        Intended to be called between sequences or videos. The specific logic is defined
        by the child class.
        """

    @abstractmethod
    def update(self, *args) -> None:
        """Updates the internal state of the metric with new inputs.

        Does not return anything. The value of the metric should be retrieved with
        compute().
        """


class MeanFrameMetric(StreamingMetric, ABC):
    """A StreamingMetric that can be represented as an average of per-frame metrics."""

    def __init__(self, assume_batch_dim: bool = True, mask_nan: bool = False) -> None:
        """Creates the metric.

        Args:
            assume_batch_dim: Whether to assume the first axis is a batch dimension. If
                True, inputs are separated along the batch dimension and passed
                independently to the core update logic. Defaults to True.
            mask_nan: Whether to ignore nan elements when computing the metric. Defaults
                to False.
        """
        self.assume_batch_dim = assume_batch_dim
        self.mask_nan = mask_nan

        # The number of items (time steps) seen so far
        self.count = 0

        # The total metric value
        self.sum = None

    def available(self) -> bool:
        """Determines whether a value for this metric is available.

        Returns:
            True if update() has been called at least once since the last flush().
        """
        return self.sum is not None

    def compute(self) -> torch.Tensor:
        return self.sum / self.count

    @abstractmethod
    def compute_metric(self, *args) -> torch.Tensor:
        """Computes the metric for a single item with no batch dimension.

        Returns:
            The value of the metric for the single item.
        """

    def flush(self) -> None:
        """Resets accumulated metric statistics."""
        self.count = 0
        self.sum = None

    def update(self, *args) -> None:
        batched = []
        for item in args:
            batched.append(item if self.assume_batch_dim else item.unsqueeze(dim=0))
        for args in zip(*batched):
            if self.mask_nan:
                mask = ~args[0].isnan()
                for item in args[1:]:
                    mask &= ~item.isnan()
                args_masked = [item[mask] for item in args]
            else:
                args_masked = args
            metric = self.compute_metric(*args_masked)
            self.count += 1
            if self.sum is None:
                self.sum = torch.zeros_like(metric)
            self.sum = self.sum + metric


class MeanInstability(StreamingMetric, ABC):
    """A StreamingMetric that can be represented as an average of frame differences."""

    def __init__(
        self,
        assume_batch_dim: bool = True,
        input_index: int = 0,
        mask_nan: bool = False,
    ) -> None:
        """Creates the metric.

        Args:
            assume_batch_dim: Whether to assume the first axis is a batch dimension. If
                True, inputs are separated along the batch dimension and passed
                independently to the core update logic. Defaults to True.
            input_index: If there are multiple input tensors, compute instability for
                this index. Defaults to 0.
            mask_nan: Whether to ignore nan elements when computing the metric. Defaults
                to False.
        """
        self.assume_batch_dim = assume_batch_dim
        self.input_index = input_index
        self.mask_nan = mask_nan

        # The number of items (time steps) seen so far
        self.count = 0

        # The total metric value
        self.sum = None

        # The previous input to update, required to compute frame differences
        self.memory = None

    def available(self) -> bool:
        """
        Returns:
            True if update() has been called at least twice since the last reset().
        """
        return self.sum is not None

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(float("nan"))
        return self.sum / self.count

    @abstractmethod
    def compute_stability(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Computes the instability for a single item with no batch dimension.

        Args:
            a: The last input to update().
            b: The current input to update().

        Returns:
            The value of the frame difference for the single item.
        """

    def flush(self) -> None:
        """Resets accumulated metric statistics."""
        self.count = 0
        self.sum = None

    def reset(self) -> None:
        """Resets the memory of the previous input."""
        self.memory = None

    def update(self, *args) -> None:
        x = args[self.input_index]
        if self.memory is None:
            self.memory = x.clone()
            return
        next_memory = x.clone()
        memory = self.memory
        if not self.assume_batch_dim:
            x = x.unsqueeze(dim=0)
            memory = memory.unsqueeze(dim=0)
        for x_i, memory_i in zip(x, memory):
            if self.mask_nan:
                mask = ~(x_i.isnan() | memory_i.isnan())
                x_i_masked = x_i[mask]
                memory_i_masked = memory_i[mask]
            else:
                x_i_masked = x_i
                memory_i_masked = memory_i
            metric = self.compute_stability(x_i_masked, memory_i_masked)
            self.count += 1
            if self.sum is None:
                self.sum = torch.zeros_like(metric)
            self.sum = self.sum + metric
        self.memory = next_memory
