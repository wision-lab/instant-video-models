import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from instant_video_models.hooks import add_hook_modules
from instant_video_models.utils import invoke_on_values


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


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


def load_videos(root_directory):
    video_paths = []
    root_directory = Path(root_directory)
    for subdir in root_directory.iterdir():
        if subdir.is_dir():
            video_paths.append(subdir)
    return video_paths


def load_images_from_videos(video_paths):
    video_data = {}
    for video_path in sorted(video_paths):
        object_groups = defaultdict(list)

        image_files = sorted(os.listdir(video_path))
        for image_file in image_files:
            if image_file.endswith(".png"):
                parts = image_file.split("_")
                frame_number = parts[0]
                video_name = parts[1]
                object_id = "_".join(parts[2:])
                object_id = object_id[:-4]

                object_groups[object_id].append(
                    {
                        "frame_number": int(frame_number),
                        "file_path": os.path.join(video_path, image_file),
                    }
                )

        # Store sorted lists of image paths for each object group
        for object_id, frames in object_groups.items():
            # Sort frames by frame number
            sorted_frames = sorted(frames, key=lambda x: x["frame_number"])
            sorted_image_paths = [frame["file_path"] for frame in sorted_frames]
            video_data.setdefault(video_path, []).append(
                (object_id, sorted_image_paths)
            )

    return video_data


def process_video_groups(
    model, device, video_data, transform, stabilizer_dict, epsilon
):
    model.eval()

    total_correct_predictions = 0
    total_images = 0

    for video_path, groups in video_data.items():
        for label, image_paths in groups:
            # reset(stabilizer_dict)
            # print(f"Starting {label} video in folder {video_path}")
            correct_predictions = 0
            video_images = len(image_paths)
            total_images += video_images

            for image_path in image_paths:
                # Extract true label from image filename
                parts = image_path.split("_")
                true_label = 1 if "person" in parts[-2] else 0
                true_tensor = torch.tensor([true_label]).cuda()

                # Load and preprocess the image
                image = Image.open(image_path).convert("RGB")
                image.requires_grad = True
                if transform:
                    image = transform(image)
                image = image.unsqueeze(0).to(device)

                perturbed_image = iterative_fgsm_attack(
                    model,
                    image,
                    true_tensor,
                    epsilon=epsilon,
                    iters=100,
                    stabilizer_dict=stabilizer_dict,
                    device=device,
                )

                output_iterative = model(perturbed_image)
                final_pred_iterative = output_iterative.max(1, keepdim=True)[1]

                # Compare predicted label with true label
                if final_pred_iterative == true_label:
                    correct_predictions += 1
                    total_correct_predictions += 1

            # Calculate and print accuracy for the current label group
            accuracy = correct_predictions / video_images
            print(
                f"Ended {label} video in folder {video_path} with accuracy: {accuracy:.4f}"
            )

    # Calculate and print total accuracy across all videos and labels
    total_accuracy = total_correct_predictions / total_images if total_images > 0 else 0
    print(f"Total accuracy across all images: {total_accuracy:.4f}")


def iterative_fgsm_attack(
    model, data, target, epsilon, iters=100, stabilizer_dict=None, device=None
):
    invoke_on_values(stabilizer_dict, "disable")
    alpha = epsilon / iters
    perturbed_data = data.clone().detach().to(device)
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
        perturbed_data = perturbed_data.detach().clone().to(device)
        perturbed_data.requires_grad = True

    invoke_on_values(stabilizer_dict, "enable")
    return perturbed_data


def reset(stabilizer_dict):
    invoke_on_values(stabilizer_dict, "reset")


def main(config_path, epsilon, decay):

    print("Loading configuration")
    config = load_config(config_path)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    print("Initializing model")
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load("weights/resnet/resnet_v6.pth"))
    model.eval()
    model.to(device)
    print("Model loaded")

    config["stabilizers"][0]["module"][
        "_target"
    ] = "stability.stabilizers.simple_fixed_stabilizer.SimpleFixedStabilizer"
    config["stabilizers"][0]["module"]["decay"] = decay

    print("Initializing Stabilizers")
    stabilizer_dict = add_hook_modules(model, config["stabilizers"], device=device)
    invoke_on_values(stabilizer_dict, "enable")

    print("Loading dataset")
    video_paths = load_videos(config["dataset"]["path"])
    random.shuffle(video_paths)
    split_index = int(0.8 * len(video_paths))
    train_paths, test_paths = video_paths[:split_index], video_paths[split_index:]

    test_data = load_images_from_videos(test_paths)
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    process_video_groups(
        model, device, test_data, test_transform, stabilizer_dict, epsilon
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model on video data with configurable settings."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    parser.add_argument(
        "--epsilon", type=float, required=True, help="Epsilon value for the attack."
    )
    parser.add_argument(
        "--decay", type=float, required=True, help="Decay value for the stabilizer."
    )
    args = parser.parse_args()

    main(args.config, args.epsilon, args.decay)


# #epsilons = [0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.07]
