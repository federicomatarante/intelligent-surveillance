from typing import Dict, List, Optional

from data.annotations import event_types, tracked_object_labels
from data.database import AnnotationsDatabase


class DatasetAnalyzer:

    def __init__(self, videos_annotations_database: Optional[AnnotationsDatabase],
                 images_annotations_database: Optional[AnnotationsDatabase]):
        """
        When initialized, it loads in memory all the data needed for the analysis from the annotation databases.
        If the parameters are None, it will not load any data from the respective database.
        :param videos_annotations_database: the database containing all annotations for event recognition dataset videos.
        :param images_annotations_database: the database containing all annotations for object tracking dataset videos.
        """
        self.videos_annotations_database = videos_annotations_database
        self.images_annotations_database = images_annotations_database
        self._load_events_analysis(videos_annotations_database)
        self._load_tracking_analysis(images_annotations_database)

    def _load_events_analysis(self, videos_annotations_database: AnnotationsDatabase):
        if videos_annotations_database is None:
            self.events_analysis = None
            self.event_labels_ids = None
            return
        # A dict containing a mapping { event_label: count of videos with that event }
        self.event_labels_distribution: Dict[int, int] = {key: 0 for key in event_types.keys()}
        # A dict containing a mapping { event_label: list of IDS of annotations with that event }
        self.event_labels_ids: Dict[int, List[int]] = {key: [] for key in event_types.keys()}
        for annotation_id in videos_annotations_database.get_ids():
            annotations = videos_annotations_database.read(annotation_id)
            event_type = annotations.events[0].event_type

            self.event_labels_distribution[event_type] += 1
            self.event_labels_ids[event_type].append(annotation_id)

    def _load_tracking_analysis(self, images_annotations_database: AnnotationsDatabase):
        if images_annotations_database is None:
            self.tracking_labels_distribution = None
            self.tracking_labels_ids = None
            return
        # A dict containing a mapping { event_label: count of videos with that event }
        self.tracking_labels_distribution: Dict[int, int] = {key: 0 for key in tracked_object_labels.keys()}
        # A dict containing a mapping { event_label: list of IDS of annotations with that event }
        self.tracking_labels_idS: Dict[int, List[int]] = {key: [] for key in tracked_object_labels.keys()}
        for annotation_id in images_annotations_database.get_ids():

            annotations = images_annotations_database.read(annotation_id)
            tracking_labels = [track.label for track in annotations.tracked_objects]
            for label in tracking_labels:
                self.tracking_labels_distribution[label] += 1
                self.tracking_labels_idS[label].append(annotation_id)

    def get_event_labels_distribution(self):
        return {key: value for key, value in
                sorted(self.event_labels_distribution.items(), key=lambda x: x[1], reverse=True)}

    @property
    def events_count(self):
        return sum(self.event_labels_distribution.values())

    def get_tracking_labels_distribution(self):
        return {
            key: value for key, value in
            sorted(self.tracking_labels_distribution.items(), key=lambda x: x[1], reverse=True)
        }
        return self.tracking_labels_distribution

    @property
    def tracked_objects_count(self):
        return sum(self.tracking_labels_distribution.values())
