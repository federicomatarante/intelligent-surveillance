from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.database import ImagesDatabase
from data.preprocess import preprocess_frame


class YOLODataset(Dataset):
    def __init__(self, images_db, annotations_db, ids):
        self.images_db: ImagesDatabase = images_db
        self.annotations_db = annotations_db
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image = self.images_db.read(image_id)
        old_dimensions = self.images_db.get_dimension(image_id)
        new_image = preprocess_frame(image, target_width=400, target_height=250, max_size=400)
        new_image = np.array(new_image)
        new_dimensions = (new_image.shape[0], new_image.shape[1])
        annotation = self.annotations_db.read(image_id)
        # Convert image to tensor and normalize
        new_image = torch.from_numpy(new_image).permute(2, 0, 1).float() / 255.0
        # Convert annotations to YOLO format
        yolo_annotation = self.convert_annotation_to_yolo_format(annotation, old_dimensions, new_dimensions)

        return new_image, yolo_annotation

    # To train a YOLO model, you need to convert this detailed annotation structure into the format expected by YOLO.
    # Here’s how you can proceed:
    def convert_annotation_to_yolo_format(self, annotation, old_dimensions: Tuple[int, int],
                                          new_dimensions: Tuple[int, int]):
        """
        Converts annotation from VIRAT format to yolo format.
        :param annotation: the annotation from VIRAT format.
        :param old_dimensions: the old dimensions in the format (height, width).
        :param new_dimensions: the new dimensions in the format (height, width).
        :return: 
        """
        yolo_annotation = []

        # Calculate scaling factors (estimation of the original size of the image)

        # these are the dimensions of the images of the entire dataset

        width_scale = new_dimensions[1] / old_dimensions[1]
        height_scale = new_dimensions[0] / old_dimensions[0]

        for obj in annotation.tracked_objects:
            for bbox in obj.bounding_boxes:
                # Scale bounding box coordinates
                x1 = bbox.x1 * width_scale
                y1 = bbox.y1 * height_scale
                x2 = bbox.x2 * width_scale
                y2 = bbox.y2 * height_scale

                # Calculate center coordinates and dimensions
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                # Normalize to be between 0 and 1
                x_center /= new_dimensions[1]
                y_center /= new_dimensions[0]
                width /= new_dimensions[1]
                height /= new_dimensions[0]

                yolo_annotation.append([obj.label, x_center, y_center, width, height])

        return torch.tensor(yolo_annotation)
