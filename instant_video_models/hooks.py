"""Defines tools for working with PyTorch hooks."""

import logging
import re
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from torch import nn

from instant_video_models.config import instantiate

logger = logging.getLogger(__name__)


class HookableModule(nn.Module, ABC):
    """A PyTorch module designed to be invoked as a hook on another module.

    This is the base class for controllers, monitors, and stabilizers.
    """

    def __init__(
        self,
        hook_input_index: int = 0,
        hook_selections: dict = None,
        hook_target: str = "output",
    ) -> None:
        """Creates the module.

        Args:
            hook_input_index: The index of the input to pass to the hook. This may be
                needed if the module has multiple inputs. Defaults to 0.
            hook_selections: A dict of the form {dim_1: i_1, dim_2: i_2}, defining a set
                of Tensor.select operations. The key gives the dimension to select along
                and the value gives the index to select. Selections are not applied in
                the order specified; they are applied in reverse dim order (because
                selecting along a dim changes the index of subsequent dims). Defaults to
                None.
            hook_target: Whether to apply this hook to the parent's input or output. Can
                be "inputs" or "output". Defaults to "output".

        Raises:
            ValueError: If an invalid value is provided for hook_target.
        """
        super().__init__()
        valid_targets = ["inputs", "output"]
        if hook_target not in valid_targets:
            raise ValueError(
                f"Invalid hook_target '{hook_target}', options are {valid_targets}."
            )
        self.hook_input_index = hook_input_index
        if hook_selections is None:
            hook_selections = {}
        self.hook_selections = OrderedDict()
        for k in reversed(sorted(hook_selections.keys())):
            self.hook_selections[k] = hook_selections[k]
        self.hook_target = hook_target
        self.module_links = {}

    def add_link(self, assigned_name: str, destination: nn.Module) -> None:
        """Stores a reference ("link") to another module.

        Args:
            assigned_name: The name to use when referring to the link destination.
            destination: The module that the link should point to.
        """
        self.module_links[assigned_name] = destination

    @abstractmethod
    def forward(self, *args) -> torch.Tensor | None:
        """Logic that runs during the forward hook.

        Returns:
            None if the output of the module should not be modified. Otherwise, returns
                an updated tensor for the module output.
        """

    def forward_hook(self, module: nn.Module, *args) -> torch.Tensor | None:
        """The method called as a forward hook.

        This method should not be overridden by child classes.

        Args:
            module: The parent module to which this has been attached as a hook.

        Raises:
            RuntimeError: If hook_target is "output" but no output was provided in the
                arguments to this method.

        Returns:
            The output of this hook, which replaces the original output of the parent
                module, or None if the output of the parent module should be unchanged.
        """
        if self.hook_target == "inputs":
            x = args[0][self.hook_input_index]
        elif len(args) < 2:
            raise RuntimeError(
                "No output available. This may occur if this module is being used as a "
                "pre hook instead of a standard (post) hook."
            )
        else:
            x = args[1]
        for dim, i in self.hook_selections.items():
            x = x.select(dim, i)
        packed = isinstance(x, tuple) and len(x) == 1
        if packed:
            x = x[0]
        output = self(x)
        if packed and (output is not None):
            output = (output,)
        return output

    def get_hook_name(self) -> str:
        """Generates a name for the hook.

        This method should not be overridden by child classes.

        Returns:
            A name for the hook (a string).
        """
        name = self.get_name()
        if self.hook_target == "inputs":
            name += "-"
            name += f"inputs_{self.hook_input_index}"
        if len(self.hook_selections) > 0:
            name += "-"
            name += ",".join(f"{dim}_{i}" for dim, i in self.hook_selections.items())
        return name

    def get_name(self) -> str:
        """Returns a base name for the hook (by default the class name).

        This is used by get_hook_name().

        This method can be overridden by child classes.

        Returns:
            A base name for the hook.
        """
        return self.__class__.__name__


def add_hook_modules(
    parent_module: nn.Module,
    hook_configs: list,
    device: str | torch.device | None = None,
) -> dict:
    """Adds hook modules to an existing module hierarchy, as specified by a config.

    Args:
        parent_module: The top-level module in the existing module hierarchy.
        hooked_configs: A list of hook specifications. A hook specification may contain:
            (1) A "filter" section specifying which modules it should be added to.
            (2) A "module" section used to instantiate the hook module.
            (3) A "links" section specifying references to other modules.
            (4) A "registration_name" string used to register in the existing hierarchy.
        device: If not None, send all instantiated hook modules to this device. Defaults
            to None.

    Returns:
        A dictionary mapping hook names to added hook modules.
    """
    hook_module_dict = {}
    for config_item in hook_configs:
        # Prepare filters
        config_filter = config_item.get("filter", {})
        filter_name = config_filter.get("name", None)
        if "regex" in config_filter:
            filter_regex = re.compile(config_filter["regex"])
        else:
            filter_regex = None
        if "neg_regex" in config_filter:
            filter_neg_regex = re.compile(config_filter["neg_regex"])
        else:
            filter_neg_regex = None
        filter_typename = config_filter.get("typename", None)

        for name, module in parent_module.named_modules():
            # Check whether this module matches the filter
            if isinstance(filter_name, str) and (name != filter_name):
                continue
            if isinstance(filter_name, list) and (name not in filter_name):
                continue
            if (filter_regex is not None) and (not filter_regex.search(name)):
                continue
            if (filter_neg_regex is not None) and filter_neg_regex.search(name):
                continue
            typename = type(module).__name__
            if isinstance(filter_typename, str) and (typename != filter_typename):
                continue
            if isinstance(filter_typename, list) and (typename not in filter_typename):
                continue

            # Create the hook module
            hook_module = instantiate(config_item["module"])
            if device is not None:
                hook_module.to(device)

            # Set up links
            module_links = _generate_module_links(
                parent_module, name, config_item.get("links", [])
            )
            for assigned_name, destination in module_links.items():
                hook_module.add_link(assigned_name, destination)

            # Save a reference to the hook in the output dictionary
            key = hook_module.get_hook_name()
            if "registration_name" in config_item:
                key = f"{config_item['registration_name']}-{key}"
            if len(name) > 0:
                key = f"{name}-{key}"
            hook_module_dict[key] = hook_module

            # Register the hook
            if config_item.get("pre_hook", False):
                module.register_forward_pre_hook(hook_module.forward_hook)
            else:
                module.register_forward_hook(hook_module.forward_hook)
            if "registration_name" in config_item:
                module.register_module(config_item["registration_name"], hook_module)
            logger.info(f"Added {key}.")

    return hook_module_dict


def _generate_module_links(parent_module: nn.Module, name: str, links: list) -> dict:
    module_links = {}
    for link in links:
        location = link["location"]
        if location[0] == ".":
            stripped = location.lstrip(".")
            up_steps = len(location) - len(stripped) - 1
            base_components = name.split(".")
            if up_steps > 0:
                base_components = base_components[:-up_steps]
            base_location = ".".join(base_components)
            if len(base_location) > 0:
                absolute_location = f"{base_location}.{stripped}"
            else:
                absolute_location = stripped
        else:
            absolute_location = location
        module_links[link["name"]] = parent_module.get_submodule(absolute_location)
    return module_links
