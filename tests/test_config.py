#!/usr/bin/env python3

import pytest
from torch import nn

from stability.config import instantiate


def test_instantiate_dict():
    config = {"_target": "torch.nn.Linear", "in_features": 64, "out_features": 32}
    module = instantiate(config)
    assert isinstance(module, nn.Linear)
    assert module.in_features == config["in_features"]
    assert module.out_features == config["out_features"]


def test_instantiate_list():
    config = [
        {"_target": "torch.nn.Linear", "in_features": 64, "out_features": 32},
        {"_target": "torch.nn.ReLU"},
    ]
    modules = instantiate(config)
    assert isinstance(modules[0], nn.Linear)
    assert modules[0].in_features == config[0]["in_features"]
    assert modules[0].out_features == config[0]["out_features"]
    assert isinstance(modules[1], nn.ReLU)


def test_instantiate_recursive():
    config = {
        "_target": "torch.nn.Sequential",
        "_args": [
            {"_target": "torch.nn.Linear", "in_features": 64, "out_features": 32},
            {"_target": "torch.nn.ReLU"},
        ],
    }
    module_list = instantiate(config)
    assert isinstance(module_list, nn.Sequential)
    assert isinstance(module_list[0], nn.Linear)
    assert module_list[0].in_features == config["_args"][0]["in_features"]
    assert module_list[0].out_features == config["_args"][0]["out_features"]
    assert isinstance(module_list[1], nn.ReLU)


if __name__ == "__main__":
    pytest.main([__file__])
