import torch
from torch.utils.data import Dataset


class YOLODataset(Dataset):
    def __init__(self, images_db, annotations_db, ids):
        self.images_db = images_db
        self.annotations_db = annotations_db
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image = self.images_db.read(image_id)
        annotation = self.annotations_db.read(image_id)

        # Convert image to tensor and normalize
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Convert annotations to YOLO format
        yolo_annotation = self.convert_annotation_to_yolo_format(annotation, image.shape[1], image.shape[2])

        return image, yolo_annotation

    # To train a YOLO model, you need to convert this detailed annotation structure into the format expected by YOLO.
    # Here’s how you can proceed:
    def convert_annotation_to_yolo_format(self, annotation, img_height, img_width):
        yolo_annotation = []

        # Calculate scaling factors (estimation of the original size of the image)

        # these are the dimensions of the images of the entire dataset

        original_width = 1280
        original_height = 720

        width_scale = img_width / original_width
        height_scale = img_height / original_height

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
                x_center /= img_width
                y_center /= img_height
                width /= img_width
                height /= img_height

                yolo_annotation.append([obj.label, x_center, y_center, width, height])

        return torch.tensor(yolo_annotation)
