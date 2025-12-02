import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import yaml
import torch
import torchvision.models as models
import torch.nn as nn
import yaml
import random
from pathlib import Path
import json
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Custom Dataset class
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
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        print(image)
        return image, label

def load_images_from_videos(video_paths):
    images = []
    labels = []
    for video_path in video_paths:
        for image_file in os.listdir(video_path):
            if image_file.endswith('.png'):
                parts = image_file.split('_')
                class_label = 1 if 'person' in parts[-2] else 0
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

# Define transformation for test dataset
def create_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Evaluate the model's accuracy
def evaluate_model(model, device, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')

# Main function to setup and run the test
def main():
    config = load_config('configs/finetune_resnet/config.yml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and prepare model
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load('weights/resnet/resnet_v5.pth'))
    model.to(device)
    
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    video_paths = load_videos(config['dataset']['path'])
    random.shuffle(video_paths)
    split_index = int(0.8 * len(video_paths))
    train_paths, test_paths = video_paths[:split_index], video_paths[split_index:]

    # Loading test images and labels
    test_images, test_labels = load_images_from_videos(test_paths)

    
    test_transform = create_transforms()
    test_dataset = CustomDataset(test_images, test_labels, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Evaluate the model
    evaluate_model(model, device, test_loader)

if __name__ == '__main__':
    main()
