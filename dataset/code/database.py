import os
import pickle
from typing import List, Tuple

import cv2
import numpy
import numpy as np

from dataset.code.annotations import Annotations


def show_video(video: List[np.ndarray]):
    """
    Shows the video on the screen.
    :param video: the video to show ( a list of np.ndarray in which each element is a frame )
    """
    for frame in video:
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


class AnnotationsDatabase:
    """
    Class to manage the annotations database

    """

    def __init__(self, database_path: str):
        """
        :param database_path: the path to the folder where the annotations database is located.
        """
        self.database_path = database_path
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
        self.annotation_paths = {
            ".".join(file.split(".")[:-1]): os.path.join(database_path, file) for file in os.listdir(database_path)
        }

    def read(self, video_id: str) -> Annotations:
        """
        Reads the annotations for a given video ID.
        :param video_id: video ID. Is the first part of a file name; for instance:
            "VIRAT_S_040005_00_000003_000058.tracking.0_1_3_7_5000_5001_10_5002_12_5003_5004.605.annotations"
        Has the ID:
            "VIRAT_S_040005_00_000003_000058.tracking.0_1_3_7_5000_5001_10_5002_12_5003_5004.605"
        :return: the annotations for the given video ID.
        """
        if video_id not in self.annotation_paths:
            raise KeyError("Video id not found in annotations database")
        annotation_file = self.annotation_paths[video_id]
        with open(annotation_file, 'rb') as f:
            annotations_data = pickle.load(f)

        return Annotations.from_json(annotations_data)

    def save(self, video_id: str, annotations_id: str, annotations: Annotations):
        annotations_path = os.path.join(self.database_path, fr"{video_id}.{annotations_id}.annotations")
        with open(annotations_path, 'wb') as f:
            pickle.dump(annotations.to_json(), f)
        self.annotation_paths[video_id] = annotations_path

    def delete(self, video_id: str):
        annotations_path = self.annotation_paths[video_id]
        os.remove(annotations_path)
        del self.annotation_paths[video_id]

    def get_ids(self):
        """
        :return: The list of the IDs in the database, meaning the elements possible to retrieve from the database.
        """
        return list(self.annotation_paths.keys())


class VideosDatabase:
    """
    Class to manage the videos database.
    """

    def __init__(self, database_path: str):
        """
        :param database_path: The path to the folder where the video database is located.
        """
        self.database_path = database_path
        self.video_paths = {
            ".".join(file.split(".")[:-1]): os.path.join(database_path, file) for file in os.listdir(database_path)
        }

    def read(self, video_id: str) -> List[numpy.ndarray]:
        """
        Read a video from the database.
        :param video_id: The ID of the video to read.
        :return: A list of NumPy arrays representing the frames of the video.
        :rtype: List[numpy.ndarray]
        """
        video_path = self.video_paths[video_id]
        cap = cv2.VideoCapture(video_path)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        return frames

    def save(self, video: List[numpy.ndarray], video_id: str, fps=30, size=(480, 640), extension='mp4'):
        """
        Save a video to the database.
        :param video: A list of NumPy arrays representing the frames of the video.
        :type video: List[numpy.ndarray]
        :param video_id: The ID to assign to the video.
        :param fps: The frames per second of the video (default: 30).
        :param size: The resolution of the video (default: (480, 640)).
        :param extension: The file extension of the video (default: 'mp4').
        """
        video_path = os.path.join(self.database_path, fr"{video_id}.{extension}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, size)
        for frame in video:
            out.write(frame)
        self.video_paths[video_id] = video_path

    def delete(self, video_id: str):
        """
        Remove a video from the database.
        :param video_id: The ID of the video to remove.
        """
        video_path = self.video_paths[video_id]
        os.remove(video_path)
        del self.video_paths[video_id]

    def get_ids(self):
        """
        :return: The list of the IDs in the database, meaning the elements possible to retrieve from the database.
        """
        return list(self.video_paths.keys())


class ImagesDatabase:
    """
    Class to manage the images database.
    """

    def __init__(self, database_path: str):
        """
        database_path (str): The path to the folder where the image database is located.
        """
        self.database_path = database_path
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
        self.image_paths = {
            ".".join(file.split(".")[:-1]): os.path.join(database_path, file) for file in os.listdir(database_path)
        }

    def read(self, image_id: str) -> np.ndarray:
        """
        Read an image from the database
        :param image_id: The ID of the image to read
        :return numpy.ndarray: The read image as a NumPy array.
        """
        image_path = self.image_paths[image_id]
        return cv2.imread(image_path)

    def save(self, image_id: str, image: numpy.ndarray):
        """
        Save an image to the database.
        :param image_id: The ID to assign to the image.
        :param image: The image to save as a NumPy array.
        """
        image_path = os.path.join(self.database_path, fr"{image_id}.jpg")
        cv2.imwrite(image_path, image)
        self.image_paths[image_id] = image_path

    def delete(self, image_id: str):
        """
        Remove an image from the database.
        :param image_id: The ID of the image to remove.
        """
        image_path = self.image_paths[image_id]
        os.remove(image_path)
        del self.image_paths[image_id]

    def get_ids(self):
        """
        :return: The list of the IDs in the database, meaning the elements possible to retrieve from the database.
        """
        return list(self.image_paths.keys())

    def get_dimension(self, image_id: str) -> tuple[int, ...]:
        """
        Get the dimensions of the image from the database.
        :param image_id: The ID of the image to get dimensions for.
        :return: A tuple of (height, width, channels) for the image.
        """
        image= self.read(image_id)
        return image.shape[0], image.shape[1]

    def get_all_dimensions(self) -> list[tuple[int, ...]]:
        image_ids = self.get_ids()
        dimensions = []

        for image_id in image_ids:
            dimensions.append(self.get_dimension(image_id))

        return dimensions




