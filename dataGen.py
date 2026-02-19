"""
This script processes a raw dataset of images by performing the following operations:
1. Resizes all images to 256x256 and converts them to grayscale.
2. Applies data augmentation techniques (e.g., flips, crops, color jitter, blur) to generate augmented samples.
3. Saves the augmented samples to a specified folder.
4. Creates a CSV manifest file listing all processed images and their corresponding classes.
"""

import os
import csv
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from itertools import cycle

# Paths for input, output, and augmented data
input_root = r""  # Raw dataset folder path
output_root = r""  # Resized dataset folder path
aug_output = r""  # Augmented samples folder path
os.makedirs(output_root, exist_ok=True)
os.makedirs(aug_output, exist_ok=True)

# Step 1: Resize and preprocess images
# Resize all images to 256x256 and convert to grayscale
print("Resizing and preprocessing images...")
target_size = (256, 256)
for class_name in os.listdir(input_root):
    class_input = os.path.join(input_root, class_name)
    class_output = os.path.join(output_root, class_name)
    if os.path.isdir(class_input):
        os.makedirs(class_output, exist_ok=True)
        for fname in os.listdir(class_input):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                img_path = os.path.join(class_input, fname)
                img = Image.open(img_path).convert("L")  # Convert to grayscale
                img = img.resize(target_size)  # Resize to 256x256
                img.save(os.path.join(class_output, fname))
print("Resizing complete. Processed images saved to:", output_root)

# Step 2: Define augmentation pipeline
# Augmentations include flips, random crops, color jitter, and Gaussian blur
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip
    # Random crop and resize at 256x256
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    # Adjust brightness and contrast
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(
        0.1, 1.0)),  # Apply Gaussian blur
    transforms.ToTensor(),  # Convert image to tensor
    # Normalize grayscale to [-1, 1]
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Step 3: Apply augmentations and save augmented samples
print("Applying augmentations and saving augmented samples...")
# Ensure transform is applied
base_dataset = ImageFolder(root=output_root, transform=augmentation)

# DataLoader to iterate over the dataset
dataloader = DataLoader(base_dataset, batch_size=16, shuffle=True)
num_augmented_to_save = 100  # Number of augmented samples to save
saved = 0
for images, labels in cycle(dataloader):
    batch_size = images.size(0)
    for i in range(batch_size):
        label_idx = labels[i].item()
        # Get class name from label index
        label = base_dataset.classes[label_idx]
        img = images[i] * 0.5 + 0.5  # Unnormalize from [-1, 1] to [0, 1]
        img_np = img.numpy().transpose((1, 2, 0))  # Convert to NumPy array
        plt.imsave(os.path.join(
            aug_output, f"{label}_{saved}.png"), img_np.squeeze(), cmap="gray")
        saved += 1
        if saved >= num_augmented_to_save:
            break
    if saved >= num_augmented_to_save:
        break
print(f"Augmented samples saved to {aug_output}: {saved} files")

# Step 4: Save dataset manifest as a CSV file
print("Saving dataset manifest...")
csv_path = os.path.join(output_root, "dataset_manifest.csv")
rows = []
for class_name in os.listdir(output_root):
    class_folder = os.path.join(output_root, class_name)
    if not os.path.isdir(class_folder):
        continue
    for fname in os.listdir(class_folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            rows.append({
                "filename": os.path.join(class_name, fname),
                "class": class_name
            })

with open(csv_path, "w", newline="") as csvfile:
    fieldnames = ["filename", "class"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
print(f"Dataset manifest saved at {csv_path}")
