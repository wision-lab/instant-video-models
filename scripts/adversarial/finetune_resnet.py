#!/usr/bin/env python3

import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from tqdm import tqdm

from stability.config import initialize_run
from stability.datasets.custom import CustomClassificationDataset
from stability.utils import set_random_seeds, MeanValue


def create_transforms(config):
    transform_list = [
        transforms.Resize(config["transforms"]["resize"]),
        transforms.ToTensor(),
    ]

    if config["transforms"].get("augmentations"):
        if config["transforms"]["augmentations"].get("random_flip"):
            transform_list.insert(0, transforms.RandomHorizontalFlip())
        if config["transforms"]["augmentations"].get("random_rotation_degrees"):
            degrees = config["transforms"]["augmentations"]["random_rotation_degrees"]
            transform_list.insert(0, transforms.RandomRotation(degrees))
        if config["transforms"]["augmentations"].get("color_jitter"):
            jitter_params = config["transforms"]["augmentations"]["color_jitter"]
            transform_list.insert(
                0,
                transforms.ColorJitter(
                    brightness=jitter_params["brightness"],
                    contrast=jitter_params["contrast"],
                    saturation=jitter_params["saturation"],
                    hue=jitter_params["hue"],
                ),
            )

    transform_list.append(
        transforms.Normalize(
            mean=config["transforms"]["normalize"]["mean"],
            std=config["transforms"]["normalize"]["std"],
        )
    )

    return transforms.Compose(transform_list)


# Load images from video paths
def load_images_from_videos(video_paths):
    images = []
    labels = []
    for video_path in video_paths:
        for image_file in os.listdir(video_path):
            if image_file.endswith(".png"):
                parts = image_file.split("_")
                class_label = 1 if "person" in parts[-2] else 0
                images.append(os.path.join(video_path, image_file))
                labels.append(class_label)
    return images, labels


def load_videos(root_directory):
    video_paths = []
    root_directory = Path(root_directory)
    for subdir in root_directory.iterdir():
        if subdir.is_dir():
            video_paths.append(subdir)
    return video_paths


# Main function
def main():
    config = initialize_run()
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    set_random_seeds(config.get("seed", 37))

    # Initialize and configure the model
    model = models.resnet50(pretrained=config["model"]["pretrained"])

    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config["model"]["num_classes"])

    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model.to(device)

    # Create training and testing transforms
    train_transform = create_transforms(config)
    test_transform = transforms.Compose(
        [
            transforms.Resize(config["transforms"]["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["transforms"]["normalize"]["mean"],
                std=config["transforms"]["normalize"]["std"],
            ),
        ]
    )

    # Load data
    video_paths = load_videos(config["dataset"]["path"])
    random.shuffle(video_paths)
    split_index = int(0.8 * len(video_paths))
    train_paths, test_paths = video_paths[:split_index], video_paths[split_index:]
    train_images, train_labels = load_images_from_videos(train_paths)
    test_images, test_labels = load_images_from_videos(test_paths)

    train_dataset = CustomClassificationDataset(
        train_images, train_labels, transform=train_transform
    )
    test_dataset = CustomClassificationDataset(
        test_images, test_labels, transform=test_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["hyperparameters"]["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["hyperparameters"]["batch_size"], shuffle=False
    )

    # Setup optimizer, loss, and scheduler
    optimizer = Adam(model.parameters(), lr=config["hyperparameters"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(
        optimizer,
        step_size=config["hyperparameters"]["step_size"],
        gamma=config["hyperparameters"]["gamma"],
    )

    # TensorBoard setup
    writer = SummaryWriter(str(Path("runs", "finetune_resnet")))

    # Output weight directory
    weights_dir = Path("weights", "resnet")
    weights_dir.mkdir(exist_ok=True)

    # Run training and validation
    train_and_validate(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        criterion,
        config["hyperparameters"]["epochs"],
        writer,
        weights_dir,
        device,
    )


# Training and validation function
def train_and_validate(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    epochs,
    writer,
    weights_dir,
    device,
):
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        mean_train_loss = MeanValue()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            mean_train_loss.update(loss.item())
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Learning Rate", current_lr, epoch)
        writer.add_scalar("Loss/Train", mean_train_loss.compute(), epoch)

        model.eval()
        mean_val_loss = MeanValue()
        mean_val_accuracy = MeanValue()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                mean_val_loss.update(loss.item())
                _, predicted = torch.max(outputs.data, dim=1)
                mean_val_accuracy.update((predicted == labels).sum().item())
        current_loss = mean_val_loss.compute()
        writer.add_scalar("Loss/Validation", current_loss, epoch)
        writer.add_scalar("Accuracy/Validation", mean_val_accuracy.compute(), epoch)

        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), weights_dir / "best.pth")
        scheduler.step()

    writer.close()


if __name__ == "__main__":
    main()
