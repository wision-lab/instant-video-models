#!/usr/bin/env python3

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -2.2, 2.7)
    return perturbed_image


def iterative_fgsm_attack(model, data, target, epsilon, iters=100):
    alpha = epsilon / iters
    perturbed_data = data.clone().detach()
    perturbed_data.requires_grad = True

    for _ in range(iters):
        output = model(perturbed_data)
        loss = nn.CrossEntropyLoss()(output, target)

        model.zero_grad()
        if perturbed_data.grad is not None:
            perturbed_data.grad.zero_()

        loss.backward()

        perturbed_data = perturbed_data + alpha * perturbed_data.grad.sign()
        perturbed_data = torch.clamp(perturbed_data, -2.2, 2.7)
        perturbed_data = perturbed_data.detach().clone()
        perturbed_data.requires_grad = True

    return perturbed_data


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


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


def random_noise_attack(image, epsilon):
    noise = 2 * epsilon * (torch.rand_like(image) - 0.5)
    perturbed_image = image + noise
    perturbed_image = torch.clamp(perturbed_image, -2.2, 2.7)
    return perturbed_image


def save_unnormalized_image(img_tensor, filename, mean, std):
    img = img_tensor.clone().detach()
    img = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    img = torch.clamp(img, 0, 1)
    save_image(img, filename)


def test_model_with_various_attacks(model, device, test_loader, epsilon):
    correct_fgsm = 0
    correct_noise = 0
    correct_iterative_fgsm = 0
    adv_examples_fgsm = []
    adv_examples_noise = []
    adv_examples_iterative_fgsm = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue

        loss = nn.CrossEntropyLoss()(output, target)
        data.grad = None
        loss.backward()
        data_grad = data.grad.data

        # FGSM
        perturbed_data_fgsm = fgsm_attack(data, epsilon, data_grad)
        output_fgsm = model(perturbed_data_fgsm)
        final_pred_fgsm = output_fgsm.max(1, keepdim=True)[1]
        if final_pred_fgsm.item() == target.item():
            correct_fgsm += 1
        if len(adv_examples_fgsm) < 5:
            adv_ex_fgsm = perturbed_data_fgsm.squeeze().detach().cpu().numpy()
            adv_examples_fgsm.append(
                (init_pred.item(), final_pred_fgsm.item(), adv_ex_fgsm, target.item())
            )

        # Random Noise
        perturbed_data_noise = random_noise_attack(data, epsilon)
        output_noise = model(perturbed_data_noise)
        final_pred_noise = output_noise.max(1, keepdim=True)[1]
        if final_pred_noise.item() == target.item():
            correct_noise += 1
        if len(adv_examples_noise) < 5:
            adv_ex_noise = perturbed_data_noise.squeeze().detach().cpu().numpy()
            adv_examples_noise.append(
                (init_pred.item(), final_pred_noise.item(), adv_ex_noise, target.item())
            )

        # Iterative FGSM
        perturbed_data_iterative = iterative_fgsm_attack(
            model, data, target, epsilon, iters=100
        )
        output_iterative = model(perturbed_data_iterative)
        final_pred_iterative = output_iterative.max(1, keepdim=True)[1]
        if final_pred_iterative.item() == target.item():
            correct_iterative_fgsm += 1
        if len(adv_examples_iterative_fgsm) < 5:
            adv_ex_iterative = perturbed_data_iterative.squeeze().detach().cpu().numpy()
            adv_examples_iterative_fgsm.append(
                (
                    init_pred.item(),
                    final_pred_iterative.item(),
                    adv_ex_iterative,
                    target.item(),
                )
            )

    print(
        f"Epsilon: {epsilon}\tFGSM Accuracy = {correct_fgsm} / {len(test_loader)} "
        f"={correct_fgsm / len(test_loader):.5f}"
    )
    print(
        f"Epsilon: {epsilon}\tNoise Accuracy = {correct_noise} / {len(test_loader)} "
        f"= {correct_noise / len(test_loader):.5f}"
    )
    print(
        f"Epsilon: {epsilon}\tIterative FGSM Accuracy "
        f"= {correct_iterative_fgsm} / {len(test_loader)} "
        f"= {correct_iterative_fgsm / len(test_loader):.5f}"
    )

    return adv_examples_fgsm, adv_examples_noise, adv_examples_iterative_fgsm


def main():
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # Update weights path as needed
    print("Loading Model")
    model.load_state_dict(torch.load("weights/resnet/best.pth"))
    model.eval()
    model.to("cuda")
    print("Model Loaded")

    print("Loading dataset")
    config = load_config("configs/finetune_resnet/defaults.yml")

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    video_paths = load_videos(config["dataset"]["path"])
    random.shuffle(video_paths)
    split_index = int(0.8 * len(video_paths))
    train_paths, test_paths = video_paths[:split_index], video_paths[split_index:]

    test_images, test_labels = load_images_from_videos(test_paths)

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = CustomDataset(test_images, test_labels, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("Dataset loaded")

    all_adv_examples = []
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    # epsilons = [0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.07]

    all_adv_examples_fgsm = []
    all_adv_examples_noise = []
    all_adv_examples_iterative_fgsm = []

    for eps in epsilons:
        print("Evaluating for epsilon:", eps)
        adv_examples_fgsm, adv_examples_noise, adv_examples_iterative_fgsm = (
            test_model_with_various_attacks(model, "cuda", test_loader, eps)
        )

        all_adv_examples_fgsm.append(adv_examples_fgsm)
        all_adv_examples_noise.append(adv_examples_noise)
        all_adv_examples_iterative_fgsm.append(adv_examples_iterative_fgsm)

    output_dir = "adversarial/images/"
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for eps, examples in zip(epsilons, all_adv_examples):
        print("\nEpsilon: ", eps)
        for i, (init_pred, final_pred, adv_ex, orig, true_label) in enumerate(examples):

            orig_filename = (
                f"{output_dir}/eps_{eps}_ex_{i}_orig_true_{true_label}"
                f"_pred_{init_pred}.png"
            )
            adv_filename = (
                f"{output_dir}/eps_{eps}_ex_{i}_adv_true_{true_label}"
                f"_pred_{final_pred}.png"
            )

            save_unnormalized_image(torch.tensor(orig), orig_filename, mean, std)
            save_unnormalized_image(torch.tensor(adv_ex), adv_filename, mean, std)


if __name__ == "__main__":
    main()
