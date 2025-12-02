import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from collections import defaultdict


model = models.resnet50(pretrained=True)
model.eval()


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_class_from_filename(filename):
    parts = filename.rsplit('_', 2)
    class_label = parts[-2]  # Get the class label
    return class_label

def process_folder(folder_path):
    # Dictionary to hold images grouped by their class labels
    class_images = defaultdict(list)

    # Group images by class
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            class_label = extract_class_from_filename(filename)
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image = transform(image).unsqueeze(0)  # Apply transformation and add batch dimension
            class_images[class_label].append(image)

    # Predict and average for each class
    class_predictions = {}
    for class_label, images in class_images.items():
        images_tensor = torch.cat(images, dim=0)
        with torch.no_grad():
            outputs = model(images_tensor)
            average_output = torch.mean(outputs, dim=0)
            predicted_index = torch.argmax(average_output).item()
            predicted_class = imagenet_labels[predicted_index]  # Using ImageNet labels
            class_predictions[class_label] = predicted_class

    return class_predictions

def load_imagenet_labels(filename):
    with open(filename, 'r') as file:
        labels = {i: line.strip() for i, line in enumerate(file)}
    return labels

# Load ImageNet labels
imagenet_labels = load_imagenet_labels('scripts/imagenet_labels.txt')


#folder_path = 'data/davis_2017/DAVIS/CroppedImages/bear'
folder_path = 'data/davis_2017/DAVIS/CroppedImages/skate-park'
predictions = process_folder(folder_path)
for class_label, predicted_class in predictions.items():
    print(f"Class: {class_label}, Predicted as: {predicted_class}")
