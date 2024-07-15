import statistics
import sys
from collections import defaultdict
from dataclasses import field, dataclass
from typing import List, Tuple, Dict, Set

import numpy as np
import torch
from scipy.spatial.distance import squareform, pdist
from ultralytics import YOLO

from YOLO.tracking import ImageTracker
from bbox_viewer import show_tracked_video
from data.annotations import Annotations, TrackedObject, Event, BoundingBox, tracked_object_labels
from data.database import VideosDatabase
from data.preprocess import preprocess_frames
from resnet.resnet import ResNet2P1DClassifier, ResNet2Plus1D, ReducedResNet2P1DClassifier


class _TrackedObjectsAnalyzer:
    """
    A class used to analyze tracked objects during detection.
    Read the methods for further information.
    """

    def __init__(self, min_frames: int, max_movement: float, box_offset: int, cluster_threshold=15.0):
        """
        :param min_frames (int): the minimum number of frames for a tracked object to be considered part of the window.
        max_movement (int): the maximum number of pixels ( in x and y direction ) of ( average ) movement of a tracked
            object to be considered still.
        :param box_offset (int): the offset added to the box after the zoom-in in the video for the event classification.
        :param cluster_threshold (float): the threshold for the clustering algorithm ( the maximum dinstance from the cluster center ).
        """
        self.box_offset = box_offset
        self.max_movement = max_movement
        self.min_frames = min_frames
        self.cluster_threshold = cluster_threshold

    def get_box(self, tracked_objects: List[TrackedObject], start_frame: int, end_frame: int, video_width: int,
                video_height: int) -> Tuple[int, int, int, int]:
        """
        Given a list of tracked objects, returns the bounding box coordinates of the biggest window that contains them
            all. A "box_offset" is added to each direction.
        :param tracked_objects: the list of tracked objects to consider
        :param start_frame: the starting frame to consider.
        :param end_frame: the ending frame to consider.
        :param video_height: the height of the video.
        :param video_width: the width of the video.
        :return: The coordinates of the biggest window that contains all the objects.
            (x1,y1,x2,y2)
        """
        all_bboxes = sum([tracked_object.bounding_boxes for tracked_object in tracked_objects], [])
        all_bboxes = [bbox for bbox in all_bboxes if start_frame <= bbox.frame <= end_frame]
        min_x1 = max(min([bbox.x1 for bbox in all_bboxes]) - self.box_offset, 0)
        min_y1 = max(min([bbox.y1 for bbox in all_bboxes]) - self.box_offset, 0)
        min_x2 = min(max([bbox.x2 for bbox in all_bboxes]) + self.box_offset, video_width - 1)
        min_y2 = min(max([bbox.y2 for bbox in all_bboxes]) + self.box_offset, video_height - 1)
        return min_x1, min_y1, min_x2, min_y2

    def _cluster_objects(self, tracked_objects_centers: Dict[TrackedObject, Tuple[float, float]]) -> List[
        List[TrackedObject]]:
        """
        It clusters the objects based on the average distance between their centers.
        :param tracked_objects_centers: a dictionary in which each key is a tracked object and the value is a tuple
            (x,y) where x and y are the centers of the objects.
        :return: a list of groups of objects.
        """
        objects = list(tracked_objects_centers.keys())
        points = np.array(list(tracked_objects_centers.values()))

        if len(objects) == 0:
            return []

        if len(objects) == 1:
            return [objects]

        distances = squareform(pdist(points))

        clusters = [{i} for i in range(len(objects))]

        while True:
            min_dist = float('inf')
            min_pair = None

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = np.mean([distances[a][b] for a in clusters[i] for b in clusters[j]])
                    if dist < min_dist:
                        min_dist = dist
                        min_pair = (i, j)

            if min_dist > self.cluster_threshold:
                break

            i, j = min_pair
            clusters[i] = clusters[i].union(clusters[j])
            clusters.pop(j)

        return [[objects[i] for i in cluster] for cluster in clusters]

    def group_tracked_objects(self, tracked_objects: List[TrackedObject], start_frame: int, end_frame: int) -> List[
        List[TrackedObject]]:
        """
        It groups tracked objects based on their average centers.
        If an object is moving too much ( above max_movement ) then it will not be considered "still" and will be
            added alone in a group.
        :param tracked_objects: the list of tracked objects to be grouped.
        :param start_frame: the start frame to consider to compute the center.
        :param end_frame: the end frame to consider to compute the center.
        :return: a list of groups of tracked objects.
        """
        track_groups = []
        still_tracked_objects_centers: Dict[TrackedObject, Tuple[float, float]] = {}
        for tracked_object in tracked_objects:
            x_centers = []
            y_centers = []

            for bb in tracked_object.bounding_boxes:
                if bb.frame < start_frame:
                    continue
                if bb.frame > end_frame:
                    break
                x_centers.append((bb.x1 + bb.x2) / 2)
                y_centers.append((bb.y1 + bb.y2) / 2)
            # If object is moving too much I don't group it with anybody
            x_variance, y_variance = statistics.stdev(x_centers), statistics.stdev(y_centers)
            if x_variance > self.max_movement or y_variance > self.max_movement:
                track_groups.append([tracked_object])
            else:
                mean_center = (statistics.mean(x_centers), statistics.mean(y_centers))
                still_tracked_objects_centers[tracked_object] = mean_center
        track_groups.extend(self._cluster_objects(still_tracked_objects_centers))
        return track_groups

    def filter(self, tracked_objects: List[TrackedObject], start_frame: int, end_frame: int) -> List[TrackedObject]:
        """
        It filters the tracked objects according to the given start and end frames.
        Basically it keeps the objects only if the number of frames inside the window [start_frame, end_frame]
        is more than the threshold "min_frames".
        :param tracked_objects: the tracked objects to filter. It's not modified.
        :param start_frame: the start frame of the window.
        :param end_frame: the end frame of the window.
        :return: a list of filtered tracked objects.
        """
        filtered_objects = []
        for tracked_object in tracked_objects:
            max_start = max(start_frame, tracked_object.start_frame)
            min_end = min(end_frame, tracked_object.end_frame)
            common_frames = min_end - max_start
            if common_frames > self.min_frames:
                filtered_objects.append(tracked_object)
        return filtered_objects


class _EventMerger:
    """
    It's a class that merges all the similar events.
    """

    def __init__(self, object_similarity_threshold: float = 0.8,
                 iou_threshold: float = 0.5,
                 frames_threshold: int = 5):
        """
        :param object_similarity_threshold: The threshold for object similarity between events.
        :param iou_threshold: The threshold for bounding box IoU between events.
        :param frames_threshold: The minimum number of frames that two events must have in common to be considered for merging.
        """
        if iou_threshold < 0 or iou_threshold > 1:
            raise ValueError('IoU threshold must be between 0 and 1.')
        if object_similarity_threshold < 0 or object_similarity_threshold > 1:
            raise ValueError('Object similarity threshold must be between 0 and 1.')
        self.object_similarity_threshold = object_similarity_threshold
        self.bbox_iou_threshold = iou_threshold
        self.frames_threshold = frames_threshold

    def merge_similar_events(self, events: Dict[int, List[Event]]) -> List[Event]:
        """
        Merge similar events based on the defined criteria.

        :param events: A dictionary of event lists, keyed by event type.
        :return: A list of merged events.
        """
        merged_events = []
        for event_list in events.values():
            event_list = sorted(event_list, key=lambda x: x.start_frame)
            unvalidated_events = set()
            for i in range(len(event_list)):
                event_1 = event_list[i]
                if event_1 in unvalidated_events:
                    continue
                for j in range(i + 1, len(event_list)):
                    event_2 = event_list[j]
                    if event_2 in unvalidated_events:
                        continue
                    if self.is_similar(event_1, event_2):
                        event_1.end_frame = event_2.end_frame
                        for event2obj, (start2, end2) in event_2.objects.items():
                            if event2obj in event_1.objects:
                                start1, end1 = event_1.objects[event2obj]
                                event_1.objects[event2obj] = (min(start1, start2), max(end1, end2))
                            else:
                                event_1.objects[event2obj] = (start2, end2)
                        unvalidated_events.add(event_2)
                merged_events.append(event_1)
        return merged_events

    def is_similar(self, event1: Event, event2: Event) -> bool:
        """
        The criteria for merging is:
            - The events have at least "frames_threshold" frames in common.
            - The events have at least "object_similarity_threshold"% objects in common.
            - The last frame's BBOX of the first event and the first frame's BBOX of the second event must have a
            IoU >= "iou_threshold".
        :param event1: The first event to compare.
        :param event2: The second event to compare.
        :return: True if the events are similar, False otherwise.
        """
        frames_in_common = min(event1.end_frame, event2.end_frame) - max(event1.start_frame, event2.start_frame)
        if frames_in_common < self.frames_threshold:
            return False

        objects1: Set = set(event1.objects.keys())
        objects2: Set = set(event2.objects.keys())

        intersection = objects1.intersection(objects2)
        union = objects1.union(objects2)
        object_similarity = len(intersection) / len(union)

        if object_similarity < self.object_similarity_threshold:
            return False

        last_bbox = event1.bounding_boxes[-1]
        first_bbox = event2.bounding_boxes[0]
        iou = BoundingBox.calculate_iou(last_bbox, first_bbox)

        return iou >= self.bbox_iou_threshold


@dataclass
class AnalyzerSettings:
    """
    It defines the analyzer settings.
        stride (int): the stride of the moving window.
        window_size (int): the window size of the moving window.
        min_frames (int): the minimum number of frames for a tracked object to be considered part of the window.
        max_movement (int): the maximum number of pixels ( in x and y direction ) of ( average ) movement of a tracked
            object to be considered still.
        box_offset (int): the offset added to the box after the zoom-in in the video for the event classification.
        cluster_threshold (float): the threshold for the clustering algorithm ( the maximum dinstance from the cluster center ).
        object_similarity_threshold(float): what percentage of common objects two events must have to be considered similair.
        iou_threshold (float): the IoU ( Intersection Over Union ) threshold two events ( the last and first bounding
            box respectively ) must have to be considered similar.
        frames_threshold (int): the minimum  frames in common two events must have to be considered similair.
    """
    stride: int = field(default=10)
    window_size: int = field(default=60)
    min_frames: int = field(default=10)
    max_movement: int = field(default=20)
    box_offset: int = field(default=10)
    cluster_threshold: float = field(default=15.0)
    object_similarity_threshold: float = field(default=0.8)
    iou_threshold: float = field(default=0.5)
    frames_threshold: int = field(default=5)


class VideoAnalyzer:
    """
    A class to analyze the videos to create Annotations.
    The algorithm used is the following:
        1. The video is analyzed to find the tracked objects.
        2. A sliding window is used to analyze independently portions of the video.
        3. For each portion, cluster of objects are created. Moving objects are considered independent clusters.
        4. For each cluster, the video is zoomed-in and the action is classified.
        5. All the "similar" events are merged to compensate for the sliding window discontinuity.
    """

    def __init__(self, action_classifier: ResNet2P1DClassifier,
                 tracker: ImageTracker, settings: AnalyzerSettings):
        """
        :param action_classifier: the action classifier for Event Classification.
        :param tracker: the tracker for ObjectDetection and ObjectTracking.
        :param settings: the settings to customize the analyzer behaviour.
        """
        self.objects_analyzer = _TrackedObjectsAnalyzer(
            min_frames=settings.min_frames,
            max_movement=settings.max_movement,
            box_offset=settings.box_offset,
            cluster_threshold=settings.cluster_threshold,
        )
        self.event_merger = _EventMerger(
            object_similarity_threshold=settings.object_similarity_threshold,
            iou_threshold=settings.iou_threshold,
            frames_threshold=settings.frames_threshold

        )
        self.stride = settings.stride
        self.window_size = settings.window_size
        self.classifier = action_classifier
        self.tracker = tracker

    @staticmethod
    def _cut_video(video: List[np.ndarray] | np.ndarray, x1, y1, x2, y2) -> List[np.ndarray]:
        """
        It cuts the video into a new sub-box.
        :param video: the video to be cut.
        :param x1: the left-top x coordinate of the new sub-box.
        :param y1: the left-top y coordinate of the new sub-box.
        :param x2: the right-bottom x coordinate of the new sub-box.
        :param y2: the right-bottom y coordinate of the new sub-box.
        :return: the new cut video.
        """
        cut_video = []
        for frame in video:
            cut_video.append(frame[y1:y2, x1:x2])
        return cut_video

    def analyze(self, video: List[np.ndarray] | np.ndarray) -> Annotations:
        """
        Given a video, it detects tracked objects and events within it.
        :param video: the video to be analyzed. Each element in the list is a numpy array with shape
            (channels, height, width).
        :return: the annotations for the video.
        """
        if isinstance(video, list):
            video = np.stack(video)
        tracked_objects: List[TrackedObject] = self.tracker.track(video)
        events: Dict[int, List[Event]] = defaultdict(list)
        video_width = video[0].shape[1]
        video_height = video[0].shape[0]
        event_id = 0
        for i in range(0, len(video), self.stride):
            start_frame = i
            end_frame = min(i + self.window_size, len(video) - 1)
            filtered_objects: List[TrackedObject] = self.objects_analyzer.filter(tracked_objects, start_frame,
                                                                                 end_frame)
            grouped_objects: List[List[TrackedObject]] = self.objects_analyzer.group_tracked_objects(filtered_objects,
                                                                                                     start_frame,
                                                                                                     end_frame)
            for group in grouped_objects:
                x1, y1, x2, y2 = self.objects_analyzer.get_box(group, start_frame, end_frame, video_width,
                                                               video_height)
                cut_video = self._cut_video(video, x1, y1, x2, y2)
                preprocessed_video = preprocess_frames(cut_video)
                event_type = self.classifier.classify(
                    preprocessed_video, group
                )
                event: Event = Event(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    event_id=event_id,
                    event_type=event_type,
                    objects={obj: (max(start_frame, obj.start_frame), min(end_frame, obj.end_frame))
                             for obj in group}
                )
                event_id += 1
                events[event_type].append(event)
        merged_events = self.event_merger.merge_similar_events(events)
        return Annotations(events=merged_events, tracked_objects=tracked_objects)


def main():
    if len(sys.argv) != 2:
        print(f"Missing argument! Right command is: python {sys.argv[0]} <video path>")

    tracker_model = YOLO(r".\YOLO\models\train\weights\best.pt",
                         verbose=False)
    action_weights = r'.\resnet\models\model_ResNet2P1-120.pt'
    state_dict = torch.load(action_weights)
    action_recognition_model = ResNet2Plus1D(
        num_classes=12,
        num_objects=len(tracked_object_labels),
        hidden_size=32,
        unfrozen_layers=6
    )
    action_recognition_model.load_state_dict(state_dict)
    action_classifier = ReducedResNet2P1DClassifier(
        model=action_recognition_model,
        classes_ids=[0, 9, 26, 38, 37, 31, 35, 15, 19, 8, 20, 12]
    )
    settings = AnalyzerSettings(
        stride=60,
        window_size=120,
        iou_threshold=0.01,
        frames_threshold=5

    )
    video_analyzer = VideoAnalyzer(
        action_classifier=action_classifier,
        tracker=ImageTracker(tracker_model),
        settings=settings

    )
    print("Reading video: ", sys.argv[1])
    video = VideosDatabase.read_video(sys.argv[1])
    print("Analyzing video...")
    annotations = video_analyzer.analyze(video)
    print("Video analyzed!")
    print("Press enter to visualize the results...")
    input()
    print("Showing tracked objects... ( press q to exit the video )")
    show_tracked_video(video, tracked_objects=annotations.tracked_objects)
    print("Showing tracked events... ( press q to exit the video )")
    show_tracked_video(video, events=annotations.events)


if __name__ == '__main__':
    main()
