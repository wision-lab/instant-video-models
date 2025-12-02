#!/usr/bin/env python3

import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stability.config import initialize_run, instantiate
from stability.helpers import prepare_item, prepare_time_step
from stability.transforms.base import initialize_transforms
from stability.utils import (
    add_hook_modules,
    best_pytorch_device,
    custom_collate,
    invoke_on_values,
    list_cuda_devices,
    set_random_seeds,
    verify_invocation_count,
)

logger = logging.getLogger(__name__)


@torch.no_grad
def main():
    device = best_pytorch_device()
    config = initialize_run()
    set_random_seeds(config.get("seed", 37))
    logger.info(f"CUDA devices: {list_cuda_devices()}")
    logger.info(f"Working directory: {os.getcwd()}")

    logger.info("Initializing dataset...")
    data = instantiate(config["dataset"])
    num_workers = config.get("num_workers", 0)
    loader = DataLoader(
        data,
        batch_size=config.get("batch_size", 1),
        num_workers=num_workers,
        collate_fn=custom_collate,
    )

    if "input_transform" in config:
        logger.info("Initializing input transform...")
        input_transform = instantiate(config["input_transform"])
    else:
        input_transform = None

    logger.info("Initializing model...")
    model = instantiate(config["model"]).to(device)
    model.eval()

    if "controllers" in config:
        logger.info("Initializing controllers...")
        controller_dict = add_hook_modules(model, config["controllers"], device=device)
        invoke_on_values(controller_dict, "enable")
    else:
        controller_dict = {}

    logger.info("Initializing stabilizers...")
    stabilizer_dict = add_hook_modules(model, config["stabilizers"], device=device)
    invoke_on_values(stabilizer_dict, "enable")

    logger.info("Initializing metrics...")
    metrics = instantiate(config["metrics"])

    if "postprocess_transform" in config:
        logger.info("Initializing postprocessing transform...")
        postprocess_transform = instantiate(config["postprocess_transform"])
    else:
        postprocess_transform = None

    if "override_weights_filepath" in config:
        override_weights_filepath = config["override_weights_filepath"]
        logger.info(f"Loading override weights from {override_weights_filepath}...")

        # Use assign=True in case any components have uninitialized parameters.
        model.load_state_dict(torch.load(override_weights_filepath), assign=True)

    def reset():
        invoke_on_values(controller_dict, "reset")
        invoke_on_values(stabilizer_dict, "reset")
        invoke_on_values(metrics, "reset")

    logger.info("Starting dataset iteration...")
    for item in tqdm(loader, ncols=0):
        reset()
        item = prepare_item(item, num_workers)
        for t, time_step in enumerate(zip(*item)):
            frame, ground_truth = prepare_time_step(time_step, input_transform, device)
            prediction = model(frame)
            if postprocess_transform is not None:
                prediction, ground_truth = postprocess_transform(
                    prediction, ground_truth
                )
            for metric in metrics:
                metric.update(prediction, ground_truth)
            verify_invocation_count(stabilizer_dict, t + 1)

    logger.info("Done!")
    if len(metrics) > 0:
        logger.info("Metrics:")
        with open("metrics.csv", "w") as metrics_file:
            for metric in metrics:
                value = metric.compute()
                logger.info(f"  {metric.get_name()}: {value:.4g}")
                print(f"{metric.get_name()},{value}", file=metrics_file)


if __name__ == "__main__":
    main()
