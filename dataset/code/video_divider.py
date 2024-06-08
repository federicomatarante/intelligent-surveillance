import os
from typing import Dict, List, Tuple, Any, Set

import cv2

from annotations import Annotations, BoundingBox
from dataset.code.annotations import TrackedObject, Event
from dataset.code.annotations_parser import AnnotationsReader
from dataset.code.database import AnnotationsDatabase


# pip install moviepy


class _VideoSegmentator:
    """
    It's a class responsible to analyze the annotations of a video and extract segmentations useful
    for extracting relevant frames and clips to create a performant dataset for object tracking
    and action recognition.
    """

    def __init__(self, annotations: Annotations):
        """
        :param annotations: the annotations to analyez.
        """
        self.annotations = annotations

    @staticmethod
    def _compare_tracked_objects(tracked_objects_1: Set[TrackedObject], tracked_objects_2: Set[TrackedObject]) -> bool:
        """
        It compares two sets of tracked objects and returns true if
        they are the same objects and if they didn't move a lot.
        Note: the tracked_object needs to contain only one bounding box!
        :param tracked_objects_1: the first set of tracked objects
        :param tracked_objects_2: the second set of tracked objects
        :return: true if they are the same objects and if they didn't move a lot.'
        """

        def did_object_move(bb1: BoundingBox, bb2: BoundingBox, delta) -> bool:
            return abs(bb1.x1 - bb2.x1) + (abs(bb1.y1 - bb2.y1) + abs(bb1.x2 - bb2.x2)) + abs(bb1.y2 - bb2.y2) > delta

        for tracked_object in tracked_objects_1:
            if len(tracked_object.bounding_boxes) != 1:
                raise ValueError("The tracked object needs to contain only one bounding box!")
            if tracked_object not in tracked_objects_2:
                return False
        for tracked_object in tracked_objects_2:
            if len(tracked_object.bounding_boxes) != 1:
                raise ValueError("The tracked object needs to contain only one bounding box!")
            if tracked_object not in tracked_objects_1:
                return False
        tracked_objects_1_map = {
            tracked_object.track_id: tracked_object for tracked_object in tracked_objects_1
        }
        tracked_objects_2_map = {
            tracked_object.track_id: tracked_object for tracked_object in tracked_objects_2
        }

        for track_id in tracked_objects_1_map:
            bbox1 = tracked_objects_1_map[track_id].bounding_boxes[0]
            bbox2 = tracked_objects_2_map[track_id].bounding_boxes[0]
            if did_object_move(bbox1, bbox2, delta=0):
                return False
        return True

    def get_object_tracking_frames(self, sampling_rate: int) -> dict[int, set[TrackedObject]]:
        """
        Returns a mapping  of frame number to the tracked objects of that frame.
        It samples the frames with a certain sampling rate to reduce memory usage.
        Before sampling, it removes consecutive frames which are similair.
        :param sampling_rate: the sampling rate.
        :return: a mapping of frame number to the tracked objects of that frame.
        The format is the following:
         {
            frame_number: list of tracked objects of that frame.
         }
        """
        tracked_objects = self.annotations.tracked_objects
        frames: Dict[int, Set[TrackedObject]] = {}
        for tracked_object in tracked_objects:
            for bbox in tracked_object.bounding_boxes:
                frame = bbox.frame
                if frame not in frames:
                    frames[frame] = set()
                new_tracked_object = TrackedObject(
                    track_id=tracked_object.track_id,
                    bounding_boxes=[bbox],
                    label=tracked_object.label
                )
                frames[frame].add(new_tracked_object)
        frames_numbers = sorted(list(frames.keys()))
        frames_to_remove = []
        for i in range(len(frames_numbers) - 1):
            track_1 = frames[frames_numbers[i]]
            track_2 = frames[frames_numbers[i + 1]]
            if self._compare_tracked_objects(track_1, track_2):
                frames_to_remove.append(frames_numbers[i])
        frames_numbers = [frame_number for frame_number in frames_numbers if frame_number not in frames_to_remove]

        return {
            frame: frames[frame] for frame in sorted(frames_numbers)[::sampling_rate]
        }

    def get_events_tracking_segmentation(self) -> list[tuple[Event, dict[str, Any]]]:
        """
        Returns a mapping of each event to the coordinates of the part of the video
        belonging to the tracked object.
        :return: a list of tuples where each tuple is of the form
        (event, {
            x1: top-left corner x-coordinate,
            y1: top-left corner y-coordinate,
            x2: bottom-right corner x-coordinate,
            y2: bottom-right corner y-coordinate,
        })
        """
        event_tracks: List[Tuple[Event, Dict[str, Any]]] = []

        for event in self.annotations.events:
            bounding_boxes = event.bounding_boxes
            min_x1 = min(bounding_boxes, key=lambda x: x.x1).x1
            min_y1 = min(bounding_boxes, key=lambda x: x.y1).y1
            max_x2 = max(bounding_boxes, key=lambda x: x.x2).x2
            max_y2 = max(bounding_boxes, key=lambda x: x.y2).y2
            event_tracks.append((
                event, {"x1": min_x1, "y1": min_y1, "x2": max_x2, "y2": max_y2},
            ))
        return event_tracks


class _VideoCutter:
    """
    A class for cutting the videos in sub-chunks of frames, with the possibility to only keep spatial patches.
    It also allows to keep single frames.
    It needs to be opened and closed, for example:
     with VideoCutter(video_path) as video_cutter:
        Code related.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.video_capture = None

    @property
    def video_length(self):
        """
        Returns the total number of frames in the video.
        """
        if self.video_capture is None:
            raise ValueError("Video capture is not initialized.")

        total_frames = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(total_frames)

    def cut_video(self, output_video_path, start_frame=0, end_frame=None, crop_info: Dict[str, int] = None):
        """
        Crops a spatial portion of a video and saves it as a new video file.
        Allows specifying a range of frames to include in the output video.

        Args:

            output_video_path (str): Path to the output cropped video file.
            start_frame (int, optional): The frame number to start from (inclusive). Default is 0.
            end_frame (int, optional): The frame number to end at (exclusive). Default is None (end of video).
            crop_info (Dict[str, int], optional): spatial information about the crop.
                x1 (int): X-coordinate of the top-left corner of the ROI.
                y1 (int): Y-coordinate of the top-left corner of the ROI.
                x2 (int): X-coordinate of the bottom-right corner of the ROI.
                y2 (int): Y-coordinate of the bottom-right corner of the ROI.
        """

        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if crop_info:
            x1, y1, x2, y2 = crop_info['x1'], crop_info['y1'], crop_info['x2'], crop_info['y2']
        else:
            x1, y1, x2, y2 = 0, 0, width, height
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out = cv2.VideoWriter(output_video_path, codec, fps, (x2 - x1, y2 - y1))
        for frame_num in range(start_frame, end_frame):
            ret, frame = self.video_capture.read()
            if not ret:
                break
            roi = frame[y1:y2, x1:x2]
            out.write(roi)
        out.release()

    def sample_frame(self, output_image_path, frame: int):
        """
        Samples a frame from the video and saves it as a new image file.
        :param output_image_path: Path to the output image file.
        :param frame: the frame number to sample.
        """
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame = self.video_capture.read()
        if not ret:
            return
        cv2.imwrite(output_image_path, frame)

    def __enter__(self):
        self.video_capture = cv2.VideoCapture(self.video_path)
        if not self.video_capture.isOpened():
            raise FileNotFoundError(f'Cannot open video {self.video_path}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.video_capture.release()


class VideosDivider:
    """
    A class that divides the videos into multiple chunks of frames or images according to the annotations linked to it.
    For every event it generates a video file only showing it and a new annotation file is created.
    For every object tracking it generates a list of images showing a bunch of it and a new annotation file is created.
    """

    def __init__(self, videos_folder: str, tracking_folder: str, events_folder: str, tracking_annotations_folder: str,
                 events_annotations_folder: str, sampling_rate: int = 10, image_extension="jpg", frames_offset=10):
        """
        :param videos_folder: the folder containing the video files to be divided.
        :param tracking_folder: the folder that will contain the images of "Object Tracking" category.
        :param events_folder: the folder that will contain the divided videos kf "Events" category.
        :param tracking_annotations_folder: the folder that will contain the annotation of "Object Tracking" images.
        :param events_annotations_folder: the folder that will contain the annotation of "Events" videos.
        :param sampling_rate: the sampling rate to extract frames for Object Tracking. Default is 10.
        :param image_extension: the image extension to extract frames for Object Tracking. Default is "jpg".
        :param frames_offset: the frames to keep before and after an event for video segmentation. Default is 10.
        """
        self.tracking_folder = tracking_folder
        self.events_folder = events_folder
        self.video_paths = self._load_video_names(videos_folder)
        self.sampling_rate = sampling_rate
        self.image_extension = image_extension
        self.frames_offset = frames_offset

        # Create folders if not exist
        os.makedirs(self.events_folder, exist_ok=True)
        os.makedirs(self.tracking_folder, exist_ok=True)
        os.makedirs(tracking_annotations_folder, exist_ok=True)
        os.makedirs(events_annotations_folder, exist_ok=True)
        self.tracking_annotations_database = AnnotationsDatabase(tracking_annotations_folder)
        self.events_annotations_database = AnnotationsDatabase(events_annotations_folder)

    @staticmethod
    def _load_video_names(videos_folder: str) -> Dict[str, str]:
        """
        Loads the video names from the video folder and returns them as a dictionary mapping the video id to the video path.
        """
        video_paths: Dict[str, str] = {}
        for file in os.listdir(videos_folder):
            if not file.endswith('.mp4'):
                continue
            video_id = file.split('.')[0]
            video_paths[video_id] = os.path.join(videos_folder, file)
        return video_paths

    def _divide_video(self, video_id: str, annotations: Annotations):
        """
        It divides a single video in multiple chunks of videos according to the annotations linked to it.

        :param video_id:
        :param annotations:
        :return:
        """
        video_path = self.video_paths[video_id]
        video_extension = video_path.split('.')[-1]
        segmentator = _VideoSegmentator(annotations)
        with _VideoCutter(video_path) as video_cutter:
            frames: Dict[int, Set[TrackedObject]] = segmentator.get_object_tracking_frames(self.sampling_rate)
            for frame, tracked_objects in frames.items():
                track_ids = '_'.join([str(track.track_id) for track in tracked_objects])
                new_video_name = f'{video_id}.tracking.{track_ids}.{frame}.{self.image_extension}'
                new_file_path = os.path.join(self.tracking_folder, new_video_name)
                video_cutter.sample_frame(new_file_path, frame)
                new_annotations_name = f'tracking.{track_ids}.{frame}'
                new_annotations = Annotations(
                    events=[],
                    tracked_objects=list(tracked_objects),
                )
                self.tracking_annotations_database.save(video_id, new_annotations_name, new_annotations)

            segmentations: List[Tuple[Event, Dict[str, Any]]] = segmentator.get_events_tracking_segmentation()
            for event, segmentation_info in segmentations:
                new_video_name = f'{video_id}.event.{event.event_id}.{video_extension}'
                new_video_path = os.path.join(self.events_folder, new_video_name)
                video_cutter.cut_video(
                    output_video_path=new_video_path,
                    start_frame=max(0, event.start_frame - self.frames_offset),
                    end_frame=min(event.end_frame + self.frames_offset, video_cutter.video_length - 1),
                    crop_info=segmentation_info,
                )
                new_annotations_name = f'event.{event.event_id}'
                new_annotations = Annotations(
                    tracked_objects=[],
                    events=[Event(
                        event_id=event.event_id,
                        event_type=event.event_type,
                        start_frame=event.start_frame,
                        end_frame=event.end_frame,
                        objects={}
                    )]
                )
                self.events_annotations_database.save(
                    video_id, new_annotations_name, new_annotations
                )

    def divide_videos(self, annotations: Dict[str, Annotations]):
        """
        Divides all the videos given in input in multiple chunks of videos according to the annotations linked to it and
        multiple images. Also new annotations files are created.
        :param annotations: A dictionary where the keys are video file ids and the values are annotations.
        Like: {
            video_id: annotations related to video
        }

        """
        for video_id, annotations in annotations.items():
            if video_id not in self.video_paths:
                continue
            self._divide_video(video_id, annotations)

