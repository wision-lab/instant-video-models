#!/usr/bin/env python3

import logging
import os
from pathlib import Path

import torch
from torch.nn.parameter import is_lazy
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from instant_video_models.config import initialize_run, instantiate
from instant_video_models.helpers import prepare_item, prepare_time_step
from instant_video_models.hooks import add_hook_modules
from instant_video_models.utils import (
    MeanValue,
    best_pytorch_device,
    custom_collate,
    invoke_on_values,
    list_cuda_devices,
    make_log_header,
    set_random_seeds,
)

logger = logging.getLogger(__name__)


def main():
    device = best_pytorch_device()
    config = initialize_run()
    set_random_seeds(config.get("seed", 37))
    logger.info(f"CUDA devices: {list_cuda_devices()}")
    logger.info(f"Working directory: {os.getcwd()}")

    tensorboard_writer = SummaryWriter(log_dir="tensorboard")
    num_workers = config.get("num_workers", 0)

    logger.info("Initializing training dataset...")
    train_data = instantiate(config["train_dataset"])
    if "train_repeats" in config:
        train_data = ConcatDataset([train_data for _ in range(config["train_repeats"])])
    train_loader = DataLoader(
        train_data,
        batch_size=config.get("train_batch_size", 1),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )

    logger.info("Initializing validation dataset...")
    val_data = instantiate(config["val_dataset"])
    val_loader = DataLoader(
        val_data,
        batch_size=config.get("val_batch_size", 1),
        shuffle=False,
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

    initial_params = list(model.parameters())
    freeze_initial_params = config.get("freeze_initial_params", False)
    if freeze_initial_params:
        for parameter in initial_params:
            parameter.requires_grad = False

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

    logger.info("Initializing loss...")
    loss = instantiate(config["loss"])
    mean_loss = MeanValue()

    logger.info("Initializing optimizer...")
    optimizer = instantiate(config["optimizer"], params=model.parameters())

    if "scheduler" in config:
        logger.info("Initializing learning rate scheduler...")
        scheduler = instantiate(config["scheduler"], optimizer=optimizer)
    else:
        scheduler = None

    if "postprocess_transform" in config:
        logger.info("Initializing postprocessing transform...")
        postprocess_transform = instantiate(config["postprocess_transform"])
    else:
        postprocess_transform = None

    # Find a checkpoint to resume from, if any
    checkpoint_path = None
    if "checkpoint_path" in config:
        checkpoint_path = Path(config["checkpoint_path"])
    elif "checkpoint_search_path" in config:
        checkpoint_options = list(Path(config["checkpoint_search_path"]).iterdir())
        checkpoint_options.sort(key=lambda p: p.stat().st_ctime)
        checkpoint_options.reverse()  # Look at most recent first
        for option in checkpoint_options:
            if option.is_dir() and (option / "epoch.txt").is_file():
                checkpoint_path = option
                break

    skip_epochs = 0
    if checkpoint_path is not None:
        logger.info(f"Resuming from checkpoint in directory {checkpoint_path}")

        if any(is_lazy(p) for p in model.parameters()):
            # Unlike in other scripts, where we use assign=True in case any lazy
            # parameters are uninitialized, in this script we pass a dummy input through
            # the model to perform lazy initialization. If we use assign=True, then the
            # parameters get overwritten by new objects. The optimizer maintains object
            # (not named) references to parameters, so overwriting them causes the
            # optimizer to lose track of them and stalls training. We run the input
            # twice to initialize stabilizers that don't know their input shape until
            # the second time step.
            item = prepare_item(next(iter(val_loader)), num_workers)
            time_step = next(zip(*item))
            frame, _ = prepare_time_step(time_step, input_transform, device)
            model.eval()  # To prevent batch norm exception with 1-element batch
            with torch.no_grad():
                for _ in range(2):
                    model(frame)
        model.load_state_dict(torch.load(checkpoint_path / "weights_last.pth"))

        optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer_last.pth"))
        scheduler_checkpoint = checkpoint_path / "scheduler_last.pth"
        if (scheduler is not None) and scheduler_checkpoint.is_file():
            scheduler.load_state_dict(torch.load(scheduler_checkpoint))

        epoch_filepath = checkpoint_path / "epoch.txt"
        if epoch_filepath.is_file():
            with open(epoch_filepath, "r") as epoch_file:
                skip_epochs = int(epoch_file.read()) + 1

    def reset():
        invoke_on_values(controller_dict, "reset")
        invoke_on_values(stabilizer_dict, "reset")
        invoke_on_values(metrics, "reset")
        loss.reset()

    def flush():
        invoke_on_values(controller_dict, "flush")
        invoke_on_values(stabilizer_dict, "flush")
        invoke_on_values(metrics, "flush")
        mean_loss.flush()
        loss.flush()

    def log_metrics(epoch_, long_name, short_name, log_to_tb=True):
        value = mean_loss.compute()
        logger.info(f"{long_name} loss: {value:.4g}")
        if log_to_tb:
            tensorboard_writer.add_scalar(f"Loss/{short_name}", value, epoch_)
        if len(metrics) > 0:
            logger.info(f"{long_name} metrics:")
            for metric in metrics:
                metric_name = metric.get_name()
                value = metric.compute()
                logger.info(f"  {metric_name}: {value:.4g}")
                if log_to_tb:
                    tensorboard_writer.add_scalar(
                        f"{metric_name}/{short_name}", value, epoch_
                    )

    @torch.no_grad
    def val_pass():
        logger.info("Starting validation pass...")
        model.eval()
        flush()
        for item in tqdm(val_loader, ncols=0):
            reset()
            item = prepare_item(item, num_workers)
            for t, time_step in enumerate(zip(*item)):
                frame, ground_truth = prepare_time_step(
                    time_step, input_transform, device
                )
                prediction = model(frame)
                if postprocess_transform is not None:
                    prediction, ground_truth = postprocess_transform(
                        prediction, ground_truth
                    )
                loss.update(prediction, ground_truth)
                if loss.available():
                    loss_value = loss.compute()
                    loss.flush()
                    mean_loss.update(loss_value.item())
                for metric in metrics:
                    metric.update(prediction, ground_truth)
        logger.info("Done!")

    def train_pass():
        logger.info("Starting training pass...")
        if scheduler is not None:
            for i, current_lr in enumerate(scheduler.get_last_lr()):
                logger.info(f"Learning rate {i}: {current_lr:.2e}")
                tensorboard_writer.add_scalar(f"LearningRate{i}", current_lr, epoch)
        if config.get("force_eval_mode", False):
            model.eval()
        elif freeze_initial_params:
            model.eval()
            for module_dict in stabilizer_dict, controller_dict:
                for item in module_dict.values():
                    item.train()
        else:
            model.train()
        flush()
        for item in tqdm(train_loader, ncols=0):
            reset()
            item = prepare_item(item, num_workers)
            final_t = min(len(x) for x in item) - 1
            optimizer.zero_grad()
            for t, time_step in enumerate(zip(*item)):
                frame, ground_truth = prepare_time_step(
                    time_step, input_transform, device
                )
                prediction = model(frame)
                if postprocess_transform is not None:
                    prediction, ground_truth = postprocess_transform(
                        prediction, ground_truth
                    )
                loss.update(prediction, ground_truth)
                if loss.available() and (t == final_t):
                    loss_value = loss.compute()
                    loss.flush()
                    mean_loss.update(loss_value.item())
                    if loss_value.requires_grad:
                        loss_value.backward()
                for metric in metrics:
                    metric.update(prediction.detach(), ground_truth)
            optimizer.step()
        logger.info("Done!")

    if (skip_epochs == 0) and (len(stabilizer_dict) > 0):
        logger.info(make_log_header("INITIAL VALIDATION PASS WITHOUT STABILIZERS"))
        invoke_on_values(controller_dict, "disable")
        invoke_on_values(stabilizer_dict, "disable")
        val_pass()
        log_metrics(-1, "Validation", "val")
        invoke_on_values(controller_dict, "enable")
        invoke_on_values(stabilizer_dict, "enable")

    logger.info(make_log_header("INITIAL VALIDATION PASS"))
    val_pass()
    log_metrics(0, "Validation", "val", log_to_tb=(skip_epochs == 0))
    best_val_loss = mean_loss.compute()

    for epoch in range(skip_epochs, config["epochs"]):
        logger.info(make_log_header(f"EPOCH {epoch + 1} OF {config['epochs']}"))
        train_pass()
        log_metrics(epoch + 1, "Training", "train")
        val_pass()
        log_metrics(epoch + 1, "Validation", "val")
        val_loss = mean_loss.compute()
        if val_loss < best_val_loss:
            logger.info("Saving new best model weights...")
            torch.save(model.state_dict(), "weights_best.pth")
            best_val_loss = val_loss
            with open("best_val_metrics.txt", "w") as metrics_file:
                metrics_file.write("Best validation metrics:\n")
                for metric in metrics:
                    metrics_file.write(
                        f"  {metric.get_name()}: {metric.compute():.4g}\n"
                    )
        if scheduler is not None:
            scheduler.step()
        torch.save(model.state_dict(), "weights_last.pth")
        torch.save(optimizer.state_dict(), "optimizer_last.pth")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), "scheduler_last.pth")
        with open("epoch.txt", "w") as epoch_file:
            epoch_file.write(str(epoch))


if __name__ == "__main__":
    main()
