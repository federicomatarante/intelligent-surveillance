import os
import random
from typing import Dict, List, Tuple, Any, Set

import cv2

from annotations import Annotations, BoundingBox
from dataset.code.annotations import TrackedObject, Event
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

    def get_events_tracking_segmentation(self, video_size: Tuple[int, int], offset: int = 5, max_duration: int = None) -> list[
        tuple[Event, dict[str, Any]]]:
        """
        Returns a mapping of each event to the coordinates of the part of the video
        belonging to the tracked object.
        It also edits "objects" field of the event, putting the right (start_frame, end_frame) and the right bounding
        boxes coordinates.
        Notice! The bounding boxes coordinates are wrong! It should not be a problem.

        :param offset: the offset of pixels to add in each direction. Default is 5.
        :param video_size: the size of the video in the format (width, height).
        :param max_duration: the maximum duration of the events in frames.
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
            max_frame = event.start_frame if max_duration is None else event.start_frame + max_duration - 1
            bounding_boxes = [bbox for bbox in event.bounding_boxes if bbox.frame <= max_frame] # TODO test

            min_x1 = min(bounding_boxes, key=lambda x: x.x1).x1
            min_y1 = min(bounding_boxes, key=lambda x: x.y1).y1
            max_x2 = max(bounding_boxes, key=lambda x: x.x2).x2
            max_y2 = max(bounding_boxes, key=lambda x: x.y2).y2
            min_x1 = max(0, min_x1 - offset)
            min_y1 = max(0, min_y1 - offset)
            max_x2 = min(max_x2 + offset, video_size[0] - 1)
            max_y2 = min(max_y2 + offset, video_size[1] - 1)
            event_tracks.append((
                event.copy(), {"x1": min_x1, "y1": min_y1, "x2": max_x2, "y2": max_y2},
            ))

        for event, new_coords in event_tracks:
            new_objects = {}
            for tracked_object, (start_frame, end_frame) in event.objects.items():
                new_tracked_object = tracked_object.copy()
                new_tracked_object.bounding_boxes = [
                    bb for bb in tracked_object.bounding_boxes if bb.frame >= event.start_frame
                ]
                for bbox in new_tracked_object.bounding_boxes:
                    bbox.x1 = max(0, bbox.x1 - new_coords["x1"])
                    bbox.y1 = max(0, bbox.y1 - new_coords["y1"])
                    bbox.x2 = max(0, bbox.x2 - new_coords["x2"])
                    bbox.y2 = max(0, bbox.y2 - new_coords["y2"])
                    bbox.x1 = min(video_size[0] - 1, bbox.x1)
                    bbox.y1 = min(video_size[1] - 1, bbox.y1)
                    bbox.x2 = min(video_size[0] - 1, bbox.x2)
                    bbox.y2 = min(video_size[1] - 1, bbox.y2)
                    bbox.frame -= event.start_frame
                new_objects[new_tracked_object] = (start_frame - event.start_frame, end_frame - event.start_frame)
            event.objects = new_objects

        return event_tracks

    def get_empty_events_tracking_segmentation(self, minimum_frames: int, video_size: Tuple[int, int],
                                               min_zoom_size: Tuple[int, int] = (30, 30)) \
            -> List[Dict[str, int]]:
        """
        Reads the annotations and gives back  alist of video frame ranges without events happening
        :param minimum_frames: the minimum number of frames required for a video portion to be considered valid.
        :param video_size: the size of the video in the format (width, height).
        :param min_zoom_size: the minimum size of the video portion required for a video portion to be considered valid.
        :return: a list of dicts in the following format:
            {
                start_frame: the first frame of the empty event,
                end_frame: the last frame of the empty event,
                x1: top-left corner x-coordinate,
                y1: top-left corner y-coordinate,
                x2: bottom-right corner x-coordinate,
                y2: bottom-right corner y-coordinate,
            }
        """

        empty_events = []
        event_frames = [(event.start_frame, event.end_frame) for event in self.annotations.events]
        event_frames = sorted(event_frames, key=lambda x: x[0])
        for i in range(len(event_frames) - 1):
            previous_end_frame = event_frames[i][1] + 1
            next_start_frame = event_frames[i + 1][0] - 1
            if next_start_frame - previous_end_frame >= minimum_frames:
                record = {
                    "start_frame": event_frames[i][0],
                    "end_frame": event_frames[i + 1][0],
                    "x1": random.randint(0, video_size[0] - min_zoom_size[0] - 1),
                    "y1": random.randint(0, video_size[1] - min_zoom_size[1] - 1),
                }
                record.update({
                    "x2": random.randint(record["x1"] + min_zoom_size[0], video_size[0] - 1),
                    "y2": random.randint(record["y1"] + min_zoom_size[1], video_size[1] - 1),
                })
                empty_events.append(record)

        return empty_events


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

    @property
    def video_size(self) -> Tuple[int, int]:
        """
        Returns the size of the video.
        :return: A tuple in the following format:
            (width, height)
        """
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

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

        width, height = self.video_size
        if crop_info:
            x1, y1, x2, y2 = crop_info['x1'], crop_info['y1'], crop_info['x2'], crop_info['y2']
        else:
            x1, y1, x2, y2 = 0, 0, width - 1, height - 1
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
                 events_annotations_folder: str, images_per_video: int = 10, image_extension="jpg", frames_offset=10,
                 min_zoom_size=(30, 30), minimum_frames=30, event_window_offset=5, max_event_duration=100):
        """
        If the folders "tracking_folder", "events folder","tracking_annotations_folder" and "events_annotations_folder"
        don't exist, they will be created.'
        :param videos_folder: the folder containing the video files to be divided.
        :param tracking_folder: the folder that will contain the images of "Object Tracking" category.
        :param events_folder: the folder that will contain the divided videos kf "Events" category.
        :param tracking_annotations_folder: the folder that will contain the annotation of "Object Tracking" images.
        :param events_annotations_folder: the folder that will contain the annotation of "Events" videos.
        :param images_per_video: the maximum number of images to extract from a video. Default is 10.
        :param image_extension: the image extension to extract frames for Object Tracking. Default is "jpg".
        :param frames_offset: the frames to keep before and after an event for video segmentation. Default is 10.
        :param min_zoom_size: the minimum size of the video patch to extract "Empty" events.
        :param minimum_frames: the minimum number of frames to extract "Empty" events.
        :param event_window_offset: the offset to add to each direction for the event window. Default is 5.
        :param max_event_duration: the maximum duration of the event window in frames. Default is 30.
        """
        self.max_event_duration = max_event_duration
        self.min_zoom_size = min_zoom_size
        self.minimum_frames = minimum_frames
        self.tracking_folder = tracking_folder
        self.events_folder = events_folder
        self.video_paths = self._load_video_names(videos_folder)
        self.images_per_video = images_per_video
        self.image_extension = image_extension
        self.frames_offset = frames_offset
        self.event_window_offset = event_window_offset
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

            sampling_rate = max(1, int(video_cutter.video_length / self.images_per_video))
            frames: Dict[int, Set[TrackedObject]] = segmentator.get_object_tracking_frames(sampling_rate)
            for frame, tracked_objects in frames.items():
                if len(tracked_objects) > 8:
                    track_ids = '_'.join([str(track.track_id) for track in list(tracked_objects)[:8]])
                    track_ids += '_more'
                else:
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

            events_segmentations: List[Tuple[Event, Dict[str, Any]]] = segmentator.get_events_tracking_segmentation(
                video_cutter.video_size, offset=self.event_window_offset, max_duration=self.max_event_duration)
            for event, segmentation_info in events_segmentations:
                new_video_name = f'{video_id}.event.{event.event_id}.{video_extension}'
                new_video_path = os.path.join(self.events_folder, new_video_name)
                event_duration = event.duration + 2 * self.frames_offset
                start_frame = event.start_frame if event_duration > self.max_event_duration else max(0,
                                                                                                     event.start_frame - self.frames_offset)
                end_frame = min(event.end_frame + self.frames_offset,
                                video_cutter.video_length - 1,
                                event.start_frame + self.max_event_duration
                                )
                video_cutter.cut_video(
                    output_video_path=new_video_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    crop_info=segmentation_info,
                )
                new_annotations_name = f'event.{event.event_id}'
                new_annotations = Annotations(
                    tracked_objects=[],
                    events=[Event(
                        event_id=event.event_id,
                        event_type=event.event_type,
                        start_frame=0,
                        end_frame=video_cutter.video_length - 1,
                        objects=event.objects
                    )]
                )
                self.events_annotations_database.save(
                    video_id, new_annotations_name, new_annotations
                )

            empty_events_segmentations: List[Dict[str, int]] = segmentator.get_empty_events_tracking_segmentation(
                minimum_frames=self.minimum_frames,
                video_size=video_cutter.video_size,
                min_zoom_size=self.min_zoom_size
            )

            for i, record in enumerate(empty_events_segmentations):
                new_video_name = f'{video_id}.event.none_{i}.{video_extension}'
                new_video_path = os.path.join(self.events_folder, new_video_name)
                video_cutter.cut_video(
                    output_video_path=new_video_path,
                    start_frame=max(0, record["start_frame"] - self.frames_offset),
                    end_frame=min(record["end_frame"] + self.frames_offset, video_cutter.video_length - 1),
                    crop_info=record,
                )
                new_annotations_name = f'event.none_{i}'
                new_annotations = Annotations(
                    tracked_objects=[],
                    events=[Event(
                        event_id=i,
                        event_type=0,
                        start_frame=record["start_frame"],
                        end_frame=record["end_frame"],
                        objects={}
                    )]
                )
                self.events_annotations_database.save(
                    video_id, new_annotations_name, new_annotations
                )

    def divide_videos(self, annotations: Dict[str, Annotations], verbose_level=0):
        """
        Divides all the videos given in input in multiple chunks of videos according to the annotations linked to it and
        multiple images. Also new annotations files are created.
        :param verbose_level: the verbose level desired. 0 is silent, 1 prints the progress and 2 prints the
         details of the videos.
        :param annotations: A dictionary where the keys are video file ids and the values are annotations.
        Like: {
            video_id: annotations related to video
        }

        """
        i = 1
        tot_videos = len(annotations)
        for video_id, annotations in annotations.items():
            if verbose_level > 0:
                print(f"Dividing video {i}/{tot_videos}")
            if verbose_level > 1:
                print(f"\tVideo id: {video_id}")
            i += 1
            if video_id not in self.video_paths:
                continue
            self._divide_video(video_id, annotations)
