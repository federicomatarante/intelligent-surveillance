import os
import re
from typing import Dict, List, Any, Tuple

import yaml

from dataset.code.annotations import BoundingBox, TrackedObject, Event, Annotations


class _KMLFileReader:
    """
    A class for reading KML files, which are the annotation files in the Virat Dataset.
    Example of usage:

        reader = KMLFileReader()
        filename = 'KML/KML.kml'
        #   Reading KML file and converted in a list of dictionaries containing
        # information about the KML lines, one line per element in a list:
        annotations = reader.read(filename)
        # Removing metas from the files:
        annotations = reader.remove_metas(annotations)


    """

    @staticmethod
    def remove_metas(record: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove all the meta elements from the KML file.
        A meta element is one of the following:
         { meta: "Something something something... ", ... }
        :param record: A list of KML files.
        :return: The list of KML files without meta elements.
        """
        return [kml for kml in record if "meta" not in kml]

    @staticmethod
    def _split_numbers(record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforms each numbers-only string of a record to a Tuple of number.
        For instance '"nums": 1 2 2 3' becomes '"num": (1,2,2,3)'.
        :param record: the record to transform.
        :return: the transformed record.
        """
        for key, value in record.items():

            if isinstance(value, dict):
                record[key] = _KMLFileReader._split_numbers(value)
                continue
            elif not isinstance(value, str):
                continue

            pattern = r'^\d+[a-zA-Z]?(\s+\d+[a-zA-Z]?)*$'
            if re.match(pattern, value):
                record[key] = tuple(int(number.rstrip("f")) for number in value.split(' '))
        return record

    @staticmethod
    def read(filename: str) -> List[Dict[str, Any]]:
        """
        Read the KML file.
        :param filename: the file path to read.
        :return: the list of lines in a Dict format.
        """
        with open(filename, 'r') as f:
            lines = f.readlines()
            lines = [line.rstrip("\n").lstrip("- ") for line in lines]
            jsons = [yaml.load(line, yaml.SafeLoader) for line in lines]
            jsons = [_KMLFileReader._split_numbers(record) for record in jsons]
        return jsons


class _GeomParser:
    """
    A class for parsing 'geometry' files in the Virat Dataset annotations.
    Usage example:

    filename = 'KML/KML.kml'
    geom_records = KMLFileReader().read(filename)
    parsed_records = _GeomParser.parse(geom_records)

    """

    @staticmethod
    def parse(records: List[Dict[str, Any]]) -> Dict[int, list[BoundingBox]]:
        """
        Processes a "geometry" dictionary returning a list of BoundingBoxes.
        :param records: a list of dict in the following format:
            {   "geom": {
                    "ts0": int,
                    "id1": int,
                    "g0": (int,int,int,int),
                    "occlusion": ["partially" | "mostly"]  // Optional
                }
            }
        :return: A dictionary in the following format:
         { track_id: [bounding_boxes], ...}
        """
        tracks: Dict[int, List[BoundingBox]] = {}
        for record in records:

            keys = ["ts0", "id1", "g0"]
            if "geom" not in record or any(key not in record["geom"] for key in keys):
                raise ValueError("The current record does not seem to be a geom record\n",
                                 f"'geom': '{record["geom"]}'.\n",
                                 f"It should have the following keys:\n",
                                 f"{keys}")
            content = record["geom"]
            frame = content["ts0"]
            track_id = content["id1"]
            x1, y1, x2, y2 = content["g0"]
            occlusion = content["occlusion"] if "occlusion" in content else None
            if not occlusion:
                occlusion = 0
            elif occlusion == 'partially':
                occlusion = 1
            elif occlusion == 'mostly':
                occlusion = 2
            else:
                raise ValueError(f"Invalid occlusion field: {occlusion}\n"
                                 f"Possible values are 'partially' and 'mostly'.")

            bbox = BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2, frame=frame, occlusion=occlusion
            )
            if track_id not in tracks:
                tracks[track_id] = []

            tracks[track_id].append(bbox)

        return tracks


class _TypesParser:
    """
    A class for parsing 'types' files in the Virat Dataset annotations.

    Usage example:
    filename = 'KML/KML.kml'
    type_records = KMLFileReader().read(filename)
    parsed_records = _TypesParser.parse(type_records)

    """

    @staticmethod
    def parse(records: List[Dict[str, Any]]) -> Dict[int, int]:
        """
        Reads and return the most common class type of a tracking from the Types KML file.
        :param records: a list of records, each a dict in the following format:
         {
            "types": {
                "id1": int,
                "cset3": {
                    "lablel1": float,
                    "label2": float,
                    ...
                }
            }
         }

        :return: a dictionary in the following format:
        { track_id: [type], ...}
        """
        track_labels: Dict[int, int] = {}
        for record in records:
            keys = ["id1", "cset3"]

            if "types" not in record or any(key not in record["types"] for key in keys):
                raise ValueError("The current record does not seem to be a types record\n",
                                 f"'types': '{record["types"]}'.\n",
                                 f"It should have the following keys:\n",
                                 f"{keys}")
            content = record["types"]
            track_id = content["id1"]
            labels = content["cset3"]
            label = max(labels, key=labels.get)
            label_id = TrackedObject.get_label_number(label)

            track_labels[track_id] = label_id

        return track_labels


class _ActivitiesParser:
    """
    A class for parsing 'activities' files in the Virat Dataset annotations.
    Usage example:
    filename = 'KML/KML.kml'
    type_records = KMLFileReader().read(filename)
    parsed_records = _ActivitiesParser.parse(type_records)

    """

    @staticmethod
    def parse(records: List[Dict[str, Any]]) -> List[tuple[Event, dict[int, tuple[int, int]]]]:
        """
        Processes an "activities" dictionary.
        :param records: a list of dict in the following format:
            { act:
                { act2: {
                    vehicle_moving: 1.0
                    },
                id2: 66,
                timespan: [{tsr0: [3291, 3407]}],
                src: truth,
                actors: [{
                    id1: 20 ,
                    timespan: [{tsr0: [3291, 3407 ]}],
                    },{
                    id1: 3 ,
                    timespan: [{tsr0: [2920, 2950 ]}]
                    }
                ]}
            }

        :return: A list of tuples in the following format:
         (Event1, {
            actor1id: (start_frame, end_frame),
            actor2id: (start_frame, end_frame),
            })
         ...
         }

        """
        activities: List[Tuple[Event, Dict[int, Tuple[int, int]]]] = []
        for record in records:
            keys = ["act2", "id2", "timespan", "src", "actors"]

            if "act" not in record or any(key not in record["act"] for key in keys):
                raise ValueError("The current record does not seem to be a types record\n",
                                 f"'types': '{record["act"]}'.\n",
                                 f"It should have the following keys:\n",
                                 f"{keys}")
            content = record["act"]

            # Extracting label of action
            labels = content["act2"]
            label = max(labels, key=labels.get)
            # Extracting the ID of the event
            event_id = content["id2"]
            # Extracts start and end frame of event
            start_frame, end_frame = content["timespan"][0]["tsr0"]

            actors: Dict[int, Tuple[int, int]] = {}
            for actor_record in content["actors"]:
                actor_id = actor_record["id1"]
                actor_start_frame, actor_end_frame = actor_record["timespan"][0]["tsr0"]
                actors[actor_id] = (actor_start_frame, actor_end_frame)

            event = Event(
                start_frame=start_frame,
                end_frame=end_frame,
                event_id=event_id,
                event_type=Event.get_label_number(label),
                objects={}
            )
            activities.append(
                (event, actors)
            )
        return activities


class _AnnotationsAssembler:
    """
    A class for assembling Annotations objects given the parsed 'geometry', 'activities' and 'types' information.
    Usage example:
    # Files containing the annotations of ONE video
    geom_file = 'KML/GeomKML.kml'
    types_file = 'KML/TypesKML.kml'
    activities_file = 'KML/ACtivitiesKML.kml'

    geom_records = KMLFileReader().read(filename)
    type_records = KMLFileReader().read(types_file)
    activities_records = KMLFileReader().read(activities_file)

    parsed_geom_records = _AnnotationsParser.parse(geom_records)
    parsed_type_records = _TypesParser.parse(type_records)
    parsed_activities_records = _ActivitiesParser.parse(activities_records)

    assembler = AnnotationsAssembler(
        bboxes: parsed_geom_records,
        track_labels: parsed_type_records,
        activities: parsed_activities_records,
    )

    annotations = assembler.assemble()
    # Now I have the annotation of a video
    """

    def __init__(self, bboxes: Dict[int, list[BoundingBox]], track_labels: Dict[int, int],
                 activities: List[tuple[Event, dict[int, tuple[int, int]]]]):
        """
        The parameters are parsed information about the annotation of one video.
        :param bboxes: A dictionary of bounding boxes in the following format: { track_id: [bounding_boxes], ...}
        :param track_labels: A dictionary linking the track_id to the label of the object in the following format: { track_id: [type], ...}
        :param activities: A list of tuples linking each event with its participating actorsin the following format:
         (Event1, {
            actor1id: (start_frame, end_frame),
            actor2id: (start_frame, end_frame),
         })
        """
        self.bboxes = bboxes
        self.track_labels = track_labels
        self.activities = activities

    def assemble(self) -> Annotations:
        """
        Assembles the annotations about the video returning an "Annotations" object.
        :return: An Annotations object containing the annotations of the video.
        """
        assembled_tracked_objects: Dict[int, TrackedObject] = {}
        for track_id, boundingBoxes in self.bboxes.items():
            if track_id not in self.track_labels:
                raise ValueError(f"Can't find a track label for track {track_id}")
            label = self.track_labels[track_id]
            tracked_object = TrackedObject(
                track_id=track_id,
                bounding_boxes=boundingBoxes,
                label=label
            )
            assembled_tracked_objects[track_id] = tracked_object

        for event, tracked_objects in self.activities:
            for tracked_object_id, (start_frame, end_frame) in tracked_objects.items():
                event.objects[assembled_tracked_objects[tracked_object_id]] = (start_frame, end_frame)

        return Annotations(
            tracked_objects=list(assembled_tracked_objects.values()),
            events=list([activity[0] for activity in self.activities]),
        )


class AnnotationsReader:
    """
    A class to read the annotations of the VIRAT dataset.
    Usage example:
    annotations_directory = 'KML/Annotations'
    reader = AnnotationsReader(annotations_directory)
    annotations = reader.read()
    """

    def __init__(self, annotations_dir: str):
        """
        It loads in memory all the file names of the folder, throwing an error if they're not valid.
        :param annotations_dir: the path to the directory containing the annotations of the
        VIRAT Dataset.
        """
        self.annotations_dir = annotations_dir
        self.annotation_files = self._get_annotation_files()

    @property
    def annotation_ids(self) -> List[str]:
        """
        :return: the IDS of all the readable annotations.
        """
        return list(self.annotation_files.keys())

    def _get_annotation_files(self) -> Dict[str, Dict[str, str]]:
        """
        Reads the annotations file names in the annotations directory and returns a dictionary
        with the following format:
         { [Video ID]: {
            'types': [File containing "types"],
            'geom': [File containing "geometry"]
            'activities': [File containing "activities"]
         }
        """
        annotation_files: Dict[str, Dict[str, str]] = {}
        for filename in os.listdir(self.annotations_dir):
            tokens = filename.split('.')
            if not filename.endswith(".yml") or len(tokens) != 3:
                raise ValueError(
                    f"Invalid annotation file name {filename}. \nIt should have the following format: [video_id].["
                    f"annotation_type].yml")

            video_id = tokens[0]
            annotation_type = tokens[1]

            if video_id not in annotation_files:
                annotation_files[video_id] = {}

            annotation_files[video_id][annotation_type] = os.path.join(self.annotations_dir, filename)
        return annotation_files

    def read_one_annotation(self, video_id: str) -> Annotations:
        """
        Given the video ID, returns the annotations of the video.
        NB: the video ID is the first part of the file name, for example, given:
            "VIRAT_S_000001.mp4" and "VIRAT_S_000007.geom.yml"
        The video ID are:
            "VIRAT_S_000001" and "VIRAT_S_000007"
        """
        annotations = self.annotation_files[video_id]
        if 'geom' not in annotations or 'activities' not in annotations or 'types' not in annotations:
            raise ValueError(
                f"Insufficient annotations for video {video_id}. \nIt should have the following annotation files:\
                    '{video_id}.geom.yml', '{video_id}.activities.yml' and '{video_id}.types.yml'"
            )
        geom_file = annotations["geom"]
        activities_file = annotations["activities"]
        types_file = annotations["types"]

        geom_kml = _KMLFileReader.read(geom_file)
        geom_kml = _KMLFileReader.remove_metas(geom_kml)
        activities_kml = _KMLFileReader.read(activities_file)
        activities_kml = _KMLFileReader.remove_metas(activities_kml)
        types_kml = _KMLFileReader.read(types_file)
        types_kml = _KMLFileReader.remove_metas(types_kml)

        bboxes = _GeomParser.parse(geom_kml)
        types = _TypesParser.parse(types_kml)
        activities = _ActivitiesParser.parse(activities_kml)

        assembler = _AnnotationsAssembler(
            bboxes=bboxes,
            track_labels=types,
            activities=activities
        )
        return assembler.assemble()

    def read(self, verbose_level: int = 0) -> Dict[str, Annotations]:
        """
        Reads the files in the directory and returns the read annotations.
        :param verbose_level: the level of verbosity. 0 is silent, 1 is verbose, 2 is very verbose.
        :return: a dictionary with the following format:
         { [Video ID]: [Video Anotation] }
        """
        assembled_annotations = {}
        i = 0
        for video_id, annotations in self.annotation_files.items():
            i += 1
            if verbose_level > 0:
                print(f"Reading annotation {i}/{len(annotations)}")
                if verbose_level > 1:
                    print("\tVideo id: ", video_id)
            if 'geom' not in annotations or 'activities' not in annotations or 'types' not in annotations:
                raise ValueError(
                    f"Insufficient annotations for video {video_id}. \nIt should have the following annotation files:\
                    '{video_id}.geom.yml', '{video_id}.activities.yml' and '{video_id}.types.yml'"
                )
            geom_file = annotations["geom"]
            activities_file = annotations["activities"]
            types_file = annotations["types"]

            geom_kml = _KMLFileReader.read(geom_file)
            geom_kml = _KMLFileReader.remove_metas(geom_kml)
            activities_kml = _KMLFileReader.read(activities_file)
            activities_kml = _KMLFileReader.remove_metas(activities_kml)
            types_kml = _KMLFileReader.read(types_file)
            types_kml = _KMLFileReader.remove_metas(types_kml)

            bboxes = _GeomParser.parse(geom_kml)
            types = _TypesParser.parse(types_kml)
            activities = _ActivitiesParser.parse(activities_kml)

            assembler = _AnnotationsAssembler(
                bboxes=bboxes,
                track_labels=types,
                activities=activities
            )
            annotation = assembler.assemble()
            assembled_annotations[video_id] = annotation
        return assembled_annotations
