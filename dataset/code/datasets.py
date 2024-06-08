import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from dataset.code.database import AnnotationsDatabase, VideosDatabase, ImagesDatabase


class ActionRecognitionDataset(Dataset):

    def __init__(self, videos_dir: str, annotations_dir: str, transform=None):
        annotation_names = os.listdir(annotations_dir)
        # Only keeping common
        self._to_tensor = ToTensor()
        self._transform = transform
        self.annotations_database = AnnotationsDatabase(annotations_dir)
        self.videos_database = VideosDatabase(videos_dir)
        self.video_ids = list(self.annotations_database.annotation_paths.keys())

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        """
        :param idx: index of the video to retrieve.
        :return: a tuple (video, label) in which
            - video is a torch tensor of shape (frames,channels,height,width)
            - label is a torch tensor of shape (1), which is the label of the video.
        """
        video_id = self.video_ids[idx]
        frames = self.videos_database.read(video_id)
        annotations = self.annotations_database.read(video_id)
        if self._transform:
            frames = [self._transform(frame) for frame in frames]

        frames = [self._to_tensor(frame) for frame in frames]
        label = annotations.events[0].event_type
        return frames, torch.tensor(label)


class ObjectTrackingDataset(Dataset):

    def __init__(self, images_dir: str, annotations_dir: str):
        self.images_db = ImagesDatabase(images_dir)
        self.annotations_db = AnnotationsDatabase(annotations_dir)
        self.images_ids = list(self.annotations_db.annotation_paths.keys())
        self._to_tensor = ToTensor()

    def __len__(self):
        return len(self.images_ids)

    def __getitem__(self, idx):
        """
        :param idx: index of the video to retrieve.
        :return: a tuple (image, track) in which
            - image is a torch tensor of shape (channels,height,width)
            - track is a torch tensor of shape (num_objects,5)
                where the last dimension corresponds to (x1,y1,x2,y2,label) of the bounding box.
        """
        image_id = self.images_ids[idx]
        image = self.images_db.read(image_id)
        annotations = self.annotations_db.read(image_id)
        tracks = annotations.tracked_objects
        annotation_info = []
        for track in tracks:
            bbox = track.bounding_boxes[0]
            annotation_info.append(
                torch.tensor([bbox.x1, bbox.y1, bbox.x2, bbox.y2, track.label])
            )
        return self._to_tensor(image), torch.tensor(annotation_info)
