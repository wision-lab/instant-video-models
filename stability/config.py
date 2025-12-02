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
from yaml import safe_load, YAMLError


def configure_loggers(log_filepath: str | Path = None, log_level: int = logging.INFO):
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    )

    # Remove any existing handlers.
    while root_logger.hasHandlers():
        root_logger.removeHandler(root_logger.handlers[0])

    # Set up stdout logging.
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # Set up file logging if requested.
    if log_filepath is not None:
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def initialize_run(
    config_location: str | Path = ".", to_container: bool = True
) -> DictConfig | ListConfig | dict | list:
    # Parse command-line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "config_path",
        help=f'the config name (the file is "{config_location}/<config_path>")',
    )
    parser.add_argument(
        "overrides", nargs="*", help="config overrides (like a.b.c=value)"
    )
    args = parser.parse_args()

    # Merge the configuration file and command-line overrides.
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

    # Generate a unique name for this configuration.
    if "_name" not in config:
        if len(args.overrides) == 0:
            name = config_path.stem
        else:
            # Remove "/" in case we are using the name to generate the output dirpath.
            # Remove "$" to avoid interpolation when accessing the name.
            override_str = "-".join(args.overrides).replace("/", "_").replace("$", "")
            name = f"{config_path.stem}-{override_str}"
        config["_name"] = name

    # Get the name of the directory containing the configuration file (this can be
    # useful when generating the output dirname).
    config["_config_dir"] = config_path.parent.name

    # Add the current datetime to the config.
    config["_datetime"] = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    # Add the current working directory to the config (in case we move later).
    config["_original_dir"] = os.getcwd()

    if "_output_dir" in config:
        # Create an output directory and save the merged configuration.
        output_dir = Path(config["_output_dir"])
        output_dir.mkdir(parents=True)
        OmegaConf.save(config, output_dir / "_config.yml", resolve=True)

        # Optionally change the current working directory to the output directory.
        if config.get("_chdir", False):
            os.chdir(output_dir)

    # Configure the root logger (all loggers inherit from the root logger).
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
    container_types = (DictConfig, ListConfig, dict, list)

    # This allows us to effectively delete an instantiated object from a yaml config
    # hierarchy by setting it to null (None).
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
    return list(sorted(p.stem for p in Path(location).glob(f"*{ext}")))


def load_config(
    config_path: str | Path, to_container: bool = False
) -> DictConfig | ListConfig | dict | list:
    config = OmegaConf.load(config_path)
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
    for remove_specifier in config.pop("_remove", []):
        del defaults[remove_specifier]
    config = OmegaConf.merge(defaults, config)
    if to_container:
        config = OmegaConf.to_container(config, resolve=True)
    return config


def resolve_config_path(parent_path: str | Path, path: str) -> str | Path:
    return (Path(parent_path).parent / path) if path.startswith(".") else path
