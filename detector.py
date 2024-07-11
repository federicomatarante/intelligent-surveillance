import random
import statistics
from typing import List, Tuple, Dict

import numpy as np
from scipy.spatial.distance import squareform, pdist

from bbox_viewer import show_tracked_video
from dataset.code.annotations import Annotations, TrackedObject, Event, BoundingBox
from dataset.code.annotations_parser import AnnotationsReader
from dataset.code.database import VideosDatabase
from dataset.code.preprocess import preprocess_frames
from resnet_model.resnet import ResNet3DClassifier


class _TrackedObjectsAnalyzer:

    def __init__(self, min_frames: int, max_movement: float, window_offset: int, cluster_threshold=15.0):
        """
        :param min_frames: the minimum number of frames required for a tracked object to be accepted
            in a window ( used for filtering )
        :param min_frames: the maximum movement a bounding box can do to be considered "still".
        :param window_offset: the offset to add to the window when calculating the bounding box
            containing all the objects.
        :param cluster_threshold: the threshold to use for clustering the objects.
        """
        self.window_offset = window_offset
        self.max_movement = max_movement
        self.min_frames = min_frames
        self.cluster_threshold = cluster_threshold

    def get_window(self, tracked_objects: List[TrackedObject], start_frame: int, end_frame: int, video_width: int,
                   video_height: int) -> Tuple[
        int, int, int, int]:
        """
        Given a list of tracked objects, returns the bounding box coordinates of the biggest window that
            contains them all. A "window_offset" is added to each direction.
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
        min_x1 = max(min([bbox.x1 for bbox in all_bboxes]) - self.window_offset, 0)
        min_y1 = max(min([bbox.y1 for bbox in all_bboxes]) - self.window_offset, 0)
        min_x2 = min(max([bbox.x2 for bbox in all_bboxes]) + self.window_offset, video_width - 1)
        min_y2 = min(max([bbox.y2 for bbox in all_bboxes]) + self.window_offset, video_height - 1)
        return min_x1, min_y1, min_x2, min_y2

    def _cluster_objects(self, tracked_objects_centers: Dict[TrackedObject, Tuple[float, float]]) -> List[
        List[TrackedObject]]:
        """
        Raggruppa gli oggetti basandosi sulla distanza media tra i loro centri.

        :param tracked_objects_centers: un dizionario dove ogni chiave è un TrackedObject
            e ogni valore è una tupla (x,y) dove x e y sono le coordinate del centro dell'oggetto.
        :param threshold: la distanza massima tra i centri di due oggetti per considerarli nello stesso gruppo.
        :return: una lista di gruppi di TrackedObjects raggruppati.
        """
        objects = list(tracked_objects_centers.keys())
        points = np.array(list(tracked_objects_centers.values()))

        if len(objects) == 0:
            return []

        if len(objects) == 1:
            return [objects]

        # Calcola la matrice delle distanze tra tutti i punti
        distances = squareform(pdist(points))

        # Inizializza i cluster
        clusters = [{i} for i in range(len(objects))]

        # Unisci i cluster finché non ci sono più fusioni possibili
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


class VideoAnalyzer:

    def __init__(self, action_classifier: ResNet3DClassifier, stride=10,
                 event_duration=60, min_frames=10, max_movement=20,
                 window_offset=10, cluster_threshold=15.0):  # TODO here the models ( all the parameters )
        self.objects_analyzer = _TrackedObjectsAnalyzer(
            min_frames=min_frames,
            max_movement=max_movement,
            window_offset=window_offset,
            cluster_threshold=cluster_threshold
        )
        self.stride = stride
        self.event_duration = event_duration
        self.action_classifier = action_classifier

    @staticmethod
    def _cut_video(video: List[np.ndarray], x1, y1, x2, y2) -> List[np.ndarray]:
        cut_video = []
        for frame in video:
            cut_video.append(frame[y1:y2, x1:x2])
        return cut_video

    def analyze(self, video: List[np.ndarray]) -> Annotations:
        """
        Given a video, it detects tracked objects and events within it.
        :param video: the video to be analyzed. Each element in the list is a numpy array with shape
            (channels, height, width).
        :return: the annotations for the video.
        """
        tracked_objects: List[TrackedObject] = []  # TODO to predict with YOLO
        events: List[Event] = []
        video_width = video[0].shape[1]
        video_height = video[0].shape[0]
        event_id = 0
        for i in range(0, len(video), self.stride):
            start_frame = i
            end_frame = max(i + self.event_duration, len(video) - 1)
            filtered_objects: List[TrackedObject] = self.objects_analyzer.filter(tracked_objects, start_frame,
                                                                                 end_frame)
            grouped_objects: List[List[TrackedObject]] = self.objects_analyzer.group_tracked_objects(filtered_objects,
                                                                                                     start_frame,
                                                                                                     end_frame)
            for group in grouped_objects:
                x1, y1, x2, y2 = self.objects_analyzer.get_window(group, start_frame, end_frame, video_width,
                                                                  video_height)
                cut_video = self._cut_video(video, x1, y1, x2, y2)
                preprocessed_video = preprocess_frames(cut_video)
                event_type = self.action_classifier.classify(
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
                events.append(event)
            # TODO maybe group same events?
        return Annotations(events=events, tracked_objects=tracked_objects)


def test_1():
    videos_dir = r'C:\Users\feder\PycharmProjects\intelligent-surveillance\dataset\videos'
    annotations_dir = r'C:\Users\feder\PycharmProjects\intelligent-surveillance\dataset\kml_annotations\all'
    print("Starting!")
    annotations_reader = AnnotationsReader(annotations_dir)
    videos_database = VideosDatabase(videos_dir)
    print("Let's go!!")
    random_id = None

    while random_id not in videos_database.get_ids():
        random_id = random.choice(annotations_reader.annotation_ids)
    print("Random Id: ", random_id)
    video = videos_database.read(random_id)
    video_width = video[0].shape[1]
    video_height = video[0].shape[0]
    print(f"Video length: {len(video)}")
    annotations = annotations_reader.read_one_annotation(random_id)
    print("Annotations read! Showing video")
    show_tracked_video(video, tracked_objects=annotations.tracked_objects)
    analyzer = _TrackedObjectsAnalyzer(10, 30, 5)
    groups = analyzer.group_tracked_objects(annotations.tracked_objects, 0, len(video) - 1)
    for group in groups:
        start_frame = min(set([item.start_frame for item in group]))
        end_frame = max(set([item.end_frame for item in group]))

        x1, y1, x2, y2 = analyzer.get_window(group, 0, len(video) - 1, video_width, video_height)
        track = TrackedObject(track_id=0, bounding_boxes=[
            BoundingBox(x1, y1, x2, y2, frame, 0) for frame in range(0, len(video) - 1)
        ], label=0)
        print("Track: ", str(track))
        print("Group: ", [{"Name: ", item.label_name, "ID: ", item.track_id} for item in group])
        items_to_show = [track]
        items_to_show.extend(annotations.tracked_objects)
        sub_video = video[start_frame:end_frame]
        show_tracked_video(sub_video, tracked_objects=items_to_show)


if __name__ == '__main__':
    print("WE")
    test_1()
