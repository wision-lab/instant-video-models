"""Defines utilities for parsing config files and initializing experiment runs."""

import importlib
import logging
import os
import sys
from argparse import ArgumentParser
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf
from yaml import YAMLError, safe_load


def configure_loggers(log_filepath: str | Path = None, log_level: int = logging.INFO):
    """Sets up Python loggers.

    Args:
        log_filepath: If this is not None, write log messages to this filepath. Defaults
            to None.
        log_level: The minimum severity level to show in logs. Defaults to logging.INFO.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    # Remove existing handlers
    while root_logger.hasHandlers():
        root_logger.removeHandler(root_logger.handlers[0])

    # Set up stdout logging
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # Set up file logging if requested
    if log_filepath is not None:
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def initialize_run(
    config_location: str | Path = ".", to_container: bool = True
) -> DictConfig | ListConfig | dict | list:
    """A convenience function to initialize an experiment run.

    Parses command-line arguments, loads the configuration, and configures loggers.

    Args:
        config_location: The base directory for configuration files. Defaults to ".".
        to_container: If True, return the configuration as a primitive Python container
            (dict or list) instead of an OmegaConf config object. Defaults to True.

    Raises:
        ValueError: If invalid command-line overrides are provided.

    Returns:
        A container object (dict- or list-like) containing the parsed configuration.
    """
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "config_path",
        help=f'the config name (the file is "{config_location}/<config_path>")',
    )
    parser.add_argument(
        "overrides", nargs="*", help="config overrides (like a.b.c=value)"
    )
    args = parser.parse_args()

    # Merge the config file and command-line overrides
    config_path = Path(config_location, args.config_path)
    config = load_config(config_path)
    for item in args.overrides:
        if "=" not in item:
            raise ValueError(f'Override "{item}" must contain an equals sign.')
        key, value = item.split("=", maxsplit=1)
        try:
            OmegaConf.update(config, key, safe_load(value))
        except YAMLError:
            raise ValueError(f'Unable to parse override value "{value}".')

    if "_name" not in config:
        config["_name"] = config_path.stem

    # Remove "/" in case we are using the name to generate the output dirpath
    # Remove "$" to avoid interpolation when accessing override_str
    config["_override_str"] = (
        "-".join(args.overrides).replace("/", "_").replace("$", "")
    )

    # The config's parent directory
    config["_config_dir"] = config_path.parent.name

    # A string for the current datetime
    config["_datetime"] = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    # The current working directory (important if _chdir is used)
    config["_original_dir"] = os.getcwd()

    if "_output_dir" in config:
        # Create the output directory and save the final config
        output_dir = Path(config["_output_dir"])
        output_dir.mkdir(parents=True)
        OmegaConf.save(config, output_dir / "_config.yml", resolve=True)

        # Optionally move to the working directory
        if config.get("_chdir", False):
            os.chdir(output_dir)

    configure_loggers(
        config.get("_log_filepath", "main.log"), config.get("_log_level", logging.INFO)
    )
    if to_container:
        config = OmegaConf.to_container(config, resolve=True)
    return config


def instantiate(
    config: DictConfig | ListConfig | dict | list | None,
    recursive: bool = True,
    **kwargs,
) -> Any:
    """Instantiates one or more objects specified in a config.

    The class to instantiate is specified by the "_target" key. Other keys specify
    keyword arguments to the constructor.

    Extra kwargs are passed to the top-level constructor.

    Args:
        config: The configuration specifying objects to instantiate.
        recursive: Whether to recursively call instantiate on config sub-containers.
            Defaults to True.

    Raises:
        ValueError: If config is a dict, recursive=False, and config has no _target.
        ValueError: If an invalid _target key is encountered.
        ValueError: If config is a list and recursive=False.
        ValueError: If config is a list and extra kwargs are provided.

    Returns:
        The instantiated object(s).
    """
    container_types = (DictConfig, ListConfig, dict, list)

    # Allows deleting an instantiated object from a config hierarchy
    if config is None:
        return None

    elif isinstance(config, (DictConfig, dict)):
        if not recursive and "_target" not in config:
            raise ValueError(
                "Dict instantiation must be recursive or have a _target directive."
            )
        union = copy(config)
        for key in kwargs:
            union[key] = kwargs[key]
        if recursive:
            processed = copy(union)
            for key, value in union.items():
                if isinstance(value, container_types):
                    processed[key] = instantiate(value)
        else:
            processed = union
        if "_target" in processed:
            target = processed.pop("_target")
            if "." not in target:
                raise ValueError(f'"{target}" is not a valid _target directive.')
            args = processed.pop("_args", [])
            i = target.rfind(".")
            python_module = importlib.import_module(target[:i])
            result = getattr(python_module, target[i + 1 :])(*args, **processed)
        else:
            result = processed

    else:
        if not recursive:
            raise ValueError("List instantiation must be recursive.")
        if len(kwargs) > 0:
            raise ValueError("List instantiation does not support additional kwargs.")
        result = copy(config)
        for i, value in enumerate(config):
            if isinstance(value, container_types):
                result[i] = instantiate(value)

    return result


def list_config_names(location: str | Path, ext: str = ".yml") -> list:
    """Lists the names (without extension) of configs within a directory.

    Args:
        location: The dirpath containing configs.
        ext: The extension fo assume for configs. Defaults to ".yml".

    Returns:
        A list of config names in the specified directory.
    """
    return list(sorted(p.stem for p in Path(location).glob(f"*{ext}")))


def load_config(
    config_path: str | Path, to_container: bool = False
) -> DictConfig | ListConfig | dict | list:
    """Loads a config from a file.

    A config may contain references to other configs in the "_defaults" list. These are
    loaded recursively by this function.

    Args:
        config_path: The filepath of the config to load.
        to_container: If True, return the configuration as a primitive Python container
            (dict or list) instead of an OmegaConf config object. Defaults to False.

    Returns:
        A container object (dict- or list-like) containing the parsed configuration.
    """
    config = OmegaConf.load(config_path)

    # Recursively load configs in "_defaults"
    defaults = []
    for defaults_specifier in config.pop("_defaults", []):
        if isinstance(defaults_specifier, str):
            path = resolve_config_path(config_path, defaults_specifier)
            defaults_item = load_config(path)
        else:
            defaults_item = OmegaConf.create()
            for key, path in defaults_specifier.items():
                path = resolve_config_path(config_path, path)
                OmegaConf.update(defaults_item, key, load_config(path))
        defaults.append(defaults_item)
    defaults = OmegaConf.merge({}, *defaults)

    # Remove keys specified in "_remove"
    for remove_specifier in config.pop("_remove", []):
        del defaults[remove_specifier]

    config = OmegaConf.merge(defaults, config)
    if to_container:
        config = OmegaConf.to_container(config, resolve=True)
    return config


def resolve_config_path(parent_path: str | Path, path: str) -> str | Path:
    """Resolves the path of a recursively referenced config.

    Filepaths starting with "." are taken as relative to the current config. Other
    filepaths are taken as-is.

    Args:
        parent_path: The parent path of the current config file (required for resolving
            relative paths).
        path: The path to be resolved.

    Returns:
        The disambiguated path of the referenced config.
    """
    return (Path(parent_path).parent / path) if path.startswith(".") else path
