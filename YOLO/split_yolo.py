from torch.utils.data import random_split, DataLoader
import os

from torchvision.transforms import ToPILImage
from ultralytics import YOLO
from YOLO.YoloDataset import YOLODataset
from dataset.code.database import ImagesDatabase
from dataset.code.database import AnnotationsDatabase
import numpy as np
from dataset.code.datasets import ObjectTrackingDataset

# Define the directory path where processed images are stored
images_dir = '/Users/serenatrovalusci/Documents/UNI/data/processed_images_folder'
annotations_dir = '/Users/serenatrovalusci/Documents/UNI/data/processed_annotations_folder'

images_db = ImagesDatabase(images_dir)
annotations_db = AnnotationsDatabase(annotations_dir)

ids = images_db.get_ids()

# Create Dataset and DataLoader
dataset = YOLODataset(images_db, annotations_db, ids)

# Determine the sizes for train, validation, and test sets
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

# Perform the split
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Paths to save the datasets
base_dir = '/Users/serenatrovalusci/Documents/UNI/data/dataset'
train_images_dir = os.path.join(base_dir, 'train/images')
train_labels_dir = os.path.join(base_dir, 'train/labels')
val_images_dir = os.path.join(base_dir, 'val/images')
val_labels_dir = os.path.join(base_dir, 'val/labels')
test_images_dir = os.path.join(base_dir, 'test/images')
test_labels_dir = os.path.join(base_dir, 'test/labels')

# Create directories if they do not exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# Function to save datasets
def save_dataset(dataset, images_dir, labels_dir):
    to_pil = ToPILImage()
    for i, (image, annotation) in enumerate(dataset):
        # Save image
        image = to_pil(image)
        image_filename = f'image_{i}.jpg'
        image_path = os.path.join(images_dir, image_filename)
        image.save(image_path)

        # Save annotation
        label_filename = f'image_{i}.txt'
        label_path = os.path.join(labels_dir, label_filename)
        with open(label_path, 'w') as f:
            for bbox in annotation:
                bbox_str = ' '.join(map(str, bbox.tolist()))
                f.write(f"{bbox_str}\n")


# Save train dataset
save_dataset(train_dataset, train_images_dir, train_labels_dir)

# Save validation dataset
save_dataset(val_dataset, val_images_dir, val_labels_dir)

# Save test dataset
save_dataset(test_dataset, test_images_dir, test_labels_dir)

print("Datasets saved successfully.")
