import os
from typing import List, Dict

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from annotations import Annotations


class _DatabaseExtractor:
    def __init__(self, videos_dir: str, annotations_dir: str, verbose=False):
        """
        :param videos_dir: the directory of videos. It will look for videos also in its subdirectories.
        :param annotations_dir: the directory of annotations. It will look for annotations also in its subdirectories.
        :param verbose: enable verbose mode ( default False). If True it will inform the user about the data it's extracting.
        """
        self.videos_dir = videos_dir
        self.annotations_dir = annotations_dir
        self.verbose = verbose

    @staticmethod
    def _read_annotations(annotation_file: str, keys: List[str], group_last: bool) -> List[Dict[str, int]]:
        """
        Reads annotations from a file.
        :param annotation_file: the file to read.
        :param keys: the keys of the annotations.
        :param group_last: whether to group last X annotations in the last key.
        :return: a list of annotations. The structure is:
          [
            {"key1": annotation1, "key2": annotation2, ...},
            {"key1": annotation1, "key2": annotation2, ...},
            ...
          ]
        """
        annotations: List[Dict[str, int]] = []
        with open(annotation_file, 'r') as file:
            for line in file:
                values = line.strip().split()
                annotation = {}
                if group_last:
                    if len(values) < len(keys):
                        raise ValueError(
                            f"Annotation file {annotation_file} has {len(values)} keys, but at least {len(keys)} had to be given")
                    for index, key in enumerate(keys[:-1]):
                        annotation[key] = int(values[index])
                    last_index = len(keys) - 2
                    annotation[keys[-1]] = [int(value) for value in values[last_index:]]
                else:
                    if len(values) != len(keys):
                        raise ValueError(
                            f'Annotation file {annotation_file} has {len(values)} keys, but {len(keys)} were given')
                    for index, key in enumerate(keys):
                        annotation[key] = int(values[index])

                annotations.append(annotation)
        return annotations

    def get_annotations(self) -> Dict[str, Dict[str, List[Dict[str, int | List[int]]]]]:
        """
        Reads the annotations file from  the specified directory.
        :return: A dictionary of annotations.
        One example of the structure is the following:

        "video_name_x": {
                "events": {
                    "eventId": 1,
                    "eventType": 2,
                    "duration": 10,
                    "startFrame": 20,
                    "endFrame": 30,
                    "currentFrame": 25,
                    "bbox_lefttop_x": 100,
                    "bbox_lefttop_y": 150,
                    "bbox_width": 50,
                    "bbox_height": 70
                },
                "objects": {
                    "objectID": 1,
                    "objectDuration": 5,
                    "currentFrame": 22,
                    "bbox_lefttop_x": 120,
                    "bbox_lefttop_y": 160,
                    "bbox_width": 40,
                    "bbox_height": 60,
                    "objectType": 3
                },
                "mapping": {
                    "eventID": 1,
                    "eventType": 2,
                    "event_duration": 10,
                    "startFrame": 20,
                    "endFrame": 30,
                    "number_of_obj": 2,
                    "relationsToObject": [1,0,1,0,0,1]
                }
            },
        """
        if self.verbose:
            print(f"Reading annotations from {self.annotations_dir}")
        annotations = {}
        for file in os.listdir(self.annotations_dir):
            if self.videos_dir:
                print("Reading annotation from file: " + file)
            file_path = os.path.join(self.annotations_dir, file)
            tokens = file.split('.')
            if len(tokens) != 4:
                continue
            video_name = tokens[0]
            annotation_type = tokens[2]
            if annotation_type == "events":
                columns = ["eventID", "eventType", "duration", "startFrame", "endFrame", "currentFrame",
                           "bbox_lefttop_x", "bbox_lefttop_y", "bbox_width", "bbox_height"]
            elif annotation_type == "objects":
                columns = ["objectID", "objectDuration", "currentFrame", "bbox_lefttop_x", "bbox_lefttop_y",
                           "bbox_width", "bbox_height", "objectType"]
            elif annotation_type == "mapping":
                columns = ["eventID", "eventType", "event_duration", "startFrame", "endFrame", "number_of_obj",
                           "relationsToObject"]
            else:
                continue
            if video_name not in annotations:
                annotations[video_name] = {}
            annotations[video_name][annotation_type] = self._read_annotations(file_path, columns,
                                                                              annotation_type == "mapping")
        return annotations

    def get_video_paths(self) -> List[str]:
        """
        Retrieve paths of video files within the specified directory.
        :return: A list of strings representing paths to video files.
        """
        video_paths = []
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpeg']
        if self.verbose:
            print(f"Extracting video paths from {self.videos_dir}")
        for file in os.listdir(self.videos_dir):
            if os.path.splitext(file)[1].lower() in video_extensions:
                file_path = os.path.join(self.videos_dir, file)
                video_paths.append(file_path)
        return video_paths

    def load_video(self, video_name) -> torch.tensor:
        """
        Load a video file and convert it into a tensor containing its frames.
        :param video_name: A string representing the name of the video file to load.
        :return: A torch tensor containing the frames of the loaded video.
        The shape of tensor should be (num_frames, height, width, channels)
        """
        if self.verbose:
            print(f"Loading video {video_name} from directory {self.videos_dir}")
        video_path = str(os.path.join(self.videos_dir, video_name))
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        video_capture.release()
        video_tensor = torch.tensor(np.array(frames))

        return video_tensor


class ViratDataset(Dataset):

    def __init__(self, videos_dir: str, annotations_dir: str, verbose=False):
        self.database_extractor = _DatabaseExtractor(videos_dir, annotations_dir, verbose)
        self.annotations = self.database_extractor.get_annotations()
        video_paths = self.database_extractor.get_video_paths()
        self.video_names = [os.path.basename(video) for video in video_paths]
        self.pure_video_names = [video_name.split('.')[0] for video_name in self.video_names]
        self.annotations = {video_name: self.annotations[video_name] for video_name in self.pure_video_names}

    def __len__(self):
        return len(self.annotations)

    # TODO it may not be the right way for dataloader. May create a dataset/collate_fn for each training
    # TODO part
    def __getitem__(self, index) -> dict[str, Annotations | Tensor]:
        video_name = self.video_names[index]
        pure_video_name = self.pure_video_names[index]
        video = self.database_extractor.load_video(video_name)
        annotations_json = self.annotations[pure_video_name]
        return {
            "frames": video,
            "annotations": Annotations.from_json(annotations_json)
        }
