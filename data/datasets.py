import os
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from data.annotations import tracked_object_labels, event_types
from data.database import AnnotationsDatabase, VideosDatabase, ImagesDatabase


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
        :return: a tuple (video, objects, label) in which
            - video is a torch tensor of shape (channels RGB,frames,height,width)
            - objects is a torch tensor of shape (objects)
                where 'objects' is the number of total objects and contains the count of
                  each object in each posizion.
            - label is a torch tensor of shape (N), where each index is one label of the video.
        """
        video_id = self.video_ids[idx]
        # Preparing frames
        frames = self.videos_database.read(video_id)
        frames = [self._to_tensor(frame) for frame in frames]

        if self._transform:
            frames = [self._transform(frame) for frame in frames]
        frames = torch.stack(frames)
        frames.permute(1, 0, 2, 3)
        annotations = self.annotations_database.read(video_id)

        # Preparing loading object
        tracked_objects = annotations.events[0].tracked_objects
        objects_involved = torch.zeros(len(tracked_object_labels))
        for tracked_object in tracked_objects:
            objects_involved[tracked_object.label] += 1

        # Preparing labels
        label = annotations.events[0].event_type
        labels = torch.zeros(len(event_types))
        labels[label] = 1
        return frames, objects_involved, labels


class ReducedActionRecognitionDataset(ActionRecognitionDataset):
    """
    A reduced version of ActionRecognitionDataset.
    This dataset allows to keep only a restricted list of classes from the dataset.
    """

    def __init__(self, videos_dir: str, annotations_dir: str, classes_to_keep: List[int], transform=None):
        """
        :param videos_dir: path to the directory containing the videos.
        :param annotations_dir: path to the directory containing the annotations.
        :param classes_to_keep: list of classes to keep in the dataset.
            The new classes will change ID after this, with the new ID being the index in the List.
            For instance:

                classes_to_keep = [5,6,8,1]
                The new IDs of the classes becomes:
                    - 5 -> 0
                    - 6 -> 1
                    - 8 -> 2
                    - 1 -> 3
        :param transform: the transform to apply to the video frames.
        """
        super().__init__(videos_dir, annotations_dir, transform)
        self.classes_to_keep = classes_to_keep
        self._filter_database()

    def _filter_database(self):
        """
        It filters the database by only keeping the classes that are in classes_to_keep.
        It updates the variable "video_ids" by keeping only the allowed classes.
        """
        new_video_ids = []
        for video_id in self.video_ids:
            annotations = self.annotations_database.read(video_id)
            event_type = annotations.events[0].event_type
            if event_type in self.classes_to_keep:
                new_video_ids.append(video_id)
        self.video_ids = new_video_ids

    def __getitem__(self, idx):
        """
        Does the same thing as ActionRecognitionDataset.
        The difference is that it changes the ID of the label as specified in __init__.
        :return three tensors:
            - frames: a tensor of shape (frames,channels RGB,height,width)
            - objects: a tensor of shape (objects)
            where 'objects' is the number of total objects and contains the count of
            each object in each position.
            - label: a tensor of shape (N), where each index is one label of the video.
        """
        frames, objects_involved, labels = super().__getitem__(idx)
        curr_index = torch.argmax(labels).item()
        new_index = self.classes_to_keep.index(curr_index)
        new_labels = torch.zeros(len(self.classes_to_keep))
        new_labels[new_index] = 1
        return frames, objects_involved, new_labels


class VideoCollater:

    def __init__(self, max_frames):
        self.max_frames = max_frames

    def __call__(self, batch):
        videos = []
        objects = []
        labels = []
        # Find the maximum sequence length in the batch
        max_len = min(max(video.size(0) for video, _, _ in batch), self.max_frames)

        # Pad or truncate the videos to the maximum length
        for video, obj, label in batch:
            if video.size(0) < max_len:
                padding = torch.zeros(max_len - video.size(0), *video.size()[1:])
                padded_video = torch.cat([video, padding], dim=0)
            elif video.size(0) > self.max_frames:
                padded_video = video[:self.max_frames]
            else:
                padded_video = video

            videos.append(padded_video)
            objects.append(obj)
            labels.append(label)
        # Stack the padded videos, objects, and labels into tensors
        videos = torch.stack(videos)
        objects = torch.stack(objects)
        labels = torch.stack(labels)
        return videos, objects, labels


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
            - image is a torch tensor of shape (channels in RGB,height,width)
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
        return self._to_tensor(image), torch.stack(annotation_info)

