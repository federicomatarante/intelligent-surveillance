from typing import Dict, List

from dataset.code.database import VideosDatabase, AnnotationsDatabase, ImagesDatabase


class DatasetAnalyzer:

    def __init__(self, videos_database: VideosDatabase, videos_annotations_database: AnnotationsDatabase,
                 images_database: ImagesDatabase, images_annotations_database: AnnotationsDatabase):
        """
        :param videos_database: the database containing all videos for event recognition dataset.
        :param videos_annotations_database: the database containing all annotations for event recognition dataset videos.
        :param images_database: the database containing all images for object tracking dataset.
        :param images_annotations_database: the database containing all annotations for object tracking dataset videos.
        """
        self.videos_annotations_database = videos_annotations_database
        self.videos_database = videos_database
        self.images_database = images_database
        self.images_annotations_database = images_annotations_database
        self._load_events_analysis(videos_annotations_database)
        self._load_tracking_analysis(images_annotations_database)

    def _load_events_analysis(self, videos_annotations_database: AnnotationsDatabase):
        # A dict containing a mapping { event_label: count of videos with that event }
        self.event_labels_distribution: Dict[int, int] = {}
        # A dict containing a mapping { event_label: list of IDS of annotations with that event }
        self.event_labels_ids: Dict[int, List[int]] = {}
        for annotation_id in videos_annotations_database.get_ids():
            annotations = videos_annotations_database.read(annotation_id)
            event_type = annotations.events[0].event_type
            if event_type not in self.event_labels_distribution:
                self.event_labels_distribution[event_type] = 0
                self.event_labels_ids[event_type] = []

            self.event_labels_distribution[event_type] += 1
            self.event_labels_ids[event_type].append(annotation_id)

    def _load_tracking_analysis(self, images_annotations_database: AnnotationsDatabase):
        # A dict containing a mapping { event_label: count of videos with that event }
        self.tracking_labels_distribution: Dict[int, int] = {}
        # A dict containing a mapping { event_label: list of IDS of annotations with that event }
        self.tracking_labels_idS: Dict[int, List[int]] = {}
        for annotation_id in images_annotations_database.get_ids():
            annotations = images_annotations_database.read(annotation_id)
            tracking_labels = [track.label for track in annotations.tracked_objects]
            for label in tracking_labels:
                if label not in self.tracking_labels_distribution:
                    self.tracking_labels_distribution[label] = 0
                    self.tracking_labels_idS[label] = []
                self.tracking_labels_distribution[label] += 1
                self.tracking_labels_idS[label].append(annotation_id)

    def get_event_labels_distribution(self):
        return self.event_labels_distribution

    @property
    def events_count(self):
        return sum(self.event_labels_distribution.values())

    def get_tracking_labels_distribution(self):
        return self.tracking_labels_distribution

    @property
    def tracked_objects_count(self):
        return sum(self.tracking_labels_distribution.values())
