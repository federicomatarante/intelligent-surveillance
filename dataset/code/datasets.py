import msvcrt
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from dataset.code.annotations import tracked_object_labels, Event, event_types
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
        :return: a tuple (video, objects, label) in which
            - video is a torch tensor of shape (frames,channels RGB,height,width)
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
        return self._to_tensor(image), torch.tensor(annotation_info)



def allign_frames(batch):
    # TODO this should normalize vidoes with different frames, check for bugs
    videos = []
    objects = []
    labels = []

    # Find the maximum sequence length in the batch
    max_len = max(video.size(0) for video, _, _ in batch)

    # Pad or truncate the videos to the maximum length
    for video, obj, label in batch:
        if video.size(0) < max_len:
            padding = torch.zeros(max_len - video.size(0), *video.size()[1:])
            padded_video = torch.cat([video, padding], dim=0)
        elif video.size(0) > max_len:
            padded_video = video[:max_len]
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
