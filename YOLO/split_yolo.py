import os
import yaml

from torch.utils.data import random_split
from torchvision.transforms import ToPILImage

from YOLO.YoloDataset import YOLODataset
from data.database import AnnotationsDatabase
from data.database import ImagesDatabase

script_dir = os.path.dirname(os.path.abspath(__file__))


def save_dataset(dataset, images_dir, labels_dir):
    """
    Save dataset to disk.
    :param dataset: dataset to be saved.
    :param images_dir: directory to save images.
    :param labels_dir: directory to save labels.
    """
    to_pil = ToPILImage()
    for i, (image, annotation) in enumerate(dataset):
        # Converti l'immagine in formato numpy array
        image = to_pil(image)

        # Salva l'immagine
        image_filename = f'image_{i}.jpg'
        image_path = os.path.join(images_dir, image_filename)
        image.save(image_path)

        # Salva l'annotazione
        label_filename = f'image_{i}.txt'
        label_path = os.path.join(labels_dir, label_filename)
        with open(label_path, 'w') as f:
            for bbox in annotation:
                bbox_str = ' '.join(map(str, bbox.tolist()))
                f.write(f"{bbox_str}\n")


def create_yolo_dataset(raw_images_dir: str, images_annotations_dir: str, images_dir: str):
    """
    Transforms the dataset to a format valid for YOLO.
    """
    # Define the directory path where processed images are stored

    images_db = ImagesDatabase(raw_images_dir)
    annotations_db = AnnotationsDatabase(images_annotations_dir)

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
    train_images_dir = os.path.join(images_dir, 'train/images')
    train_labels_dir = os.path.join(images_dir, 'train/labels')
    val_images_dir = os.path.join(images_dir, 'val/images')
    val_labels_dir = os.path.join(images_dir, 'val/labels')
    test_images_dir = os.path.join(images_dir, 'test/images')
    test_labels_dir = os.path.join(images_dir, 'test/labels')

    # Create directories if they do not exist
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)

    # Save train dataset
    save_dataset(train_dataset, train_images_dir, train_labels_dir)

    # Save validation dataset
    save_dataset(val_dataset, val_images_dir, val_labels_dir)

    # Save test dataset
    save_dataset(test_dataset, test_images_dir, test_labels_dir)

    # Creating YAML file
    file_content = {
        'train': train_images_dir,
        'val': val_images_dir,
        'nc': 14,
        'names': ['Dumpster', 'Door', 'Prop', 'Push_Pulled_Object', 'Person', 'Animal', 'Construction_Vehicle',
                  'Construction_Barrier', 'Vehicle', 'Tree', 'Parking_Meter', 'Bike', 'Articulated_Infrastructure',
                  'Other']
    }

    with open(os.path.join(script_dir, 'virat_dataset.yaml'), 'w') as file:
        yaml.dump(file_content, file)
        print("Datasets saved successfully.")
