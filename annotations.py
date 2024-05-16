from dataclasses import dataclass
from typing import List, Dict


@dataclass
class BoundingBox:
    """
    BoundingBox defines the current bounding box of an object.

    Args:
        x (int): x coordinate of the left-top point of the bounding box.
        y (int): y coordinate of the left-top point of the bounding box.
        width (int): width of the bounding box.
        height (int): height of the bounding box.
        frame (int): frame number of the current bounding box relative to the video beginning.

    """
    x: int
    y: int
    width: int
    height: int
    frame: int


@dataclass
class TrackedObject:
    """
    Object defines a track of an object in a video.

    Args:
        object_id (int): unique id of the object ( in the current video clip ).
        duration (int): duration of the object tracking ( in frame number ).
        bounding_boxes (List[BoundingBox]): list of all the bounding boxes of the object, from the first frame of the tracking
            to the last frame of the tracking. len(bounding_boxes) should be equal to "duration".
        object_type(int): type of the object. The possible values are:
            - 1: person
            - 2: car              (usually passenger vehicles such as sedan, truck)
            - 3: vehicles         (vehicles other than usual passenger cars. Examples include construction vehicles)
            - 4: object           (neither car or person, usually carried objects)
            - 5: bike, bicylces   (may include engine-powered auto-bikes)
    """
    object_id: int
    duration: int
    bounding_boxes: List[BoundingBox]
    object_type: int


@dataclass
class Event:
    """
    Event defines a track of an event.

    Args:
        event_id (int): unique id of the event ( in the current video clip ).
        start_frame (int): frame number of the start of the event.
        end_frame (int): frame number of the end of the event.
        duration (int): duration of the event tracking ( in frame number ).
            In theory should be equal to "start_frame" - "end_frame" + 1.
        bounding_boxes (List[BoundingBox]): list of all the bounding boxes of the event, from the first frame of the tracking
            to the last frame of the tracking. len(bounding_boxes) should be equal to "duration".
        objects (List[TrackedObject]): list of all the objects involved in the event.
        event_type (int): type of the event. The possible values are:
            - 1: Person loading an Object to a Vehicle
            - 2: Person Unloading an Object from a Car/Vehicle
            - 3: Person Opening a Vehicle/Car Trunk
            - 4: Person Closing a Vehicle/Car Trunk
            - 5: Person getting into a Vehicle
            - 6: Person getting out of a Vehicle
            - 7: Person gesturing
            - 8: Person digging
            - 9: Person carrying an object
            - 10: Person running
            - 11: Person entering a facility
            - 12: Person exiting a facility

    """
    event_id: int
    duration: int
    start_frame: int
    end_frame: int
    bounding_boxes: List[BoundingBox]
    event_type: int
    objects: List[TrackedObject]


class _AnnotationsReader:
    def __init__(self, json: Dict):
        self.json = json

    def extract_objects(self) -> Dict[int, TrackedObject]:
        json_objects = self.json["objects"]
        tracked_objects_tmp: Dict[int, TrackedObject] = {}
        for json_object in json_objects:
            object_id = json_object["objectID"]
            duration = json_object["objectDuration"]
            frame = json_object["currentFrame"]
            x = json_object["bbox_lefttop_x"]
            y = json_object["bbox_lefttop_y"]
            width = json_object["bbox_width"]
            height = json_object["bbox_height"]
            object_type = json_object["objectType"]
            if object_id not in tracked_objects_tmp:
                tracked_object = TrackedObject(
                    object_id=object_id,
                    duration=duration,
                    object_type=object_type,
                    bounding_boxes=[],
                )
                tracked_objects_tmp[object_id] = tracked_object
            bounding_box = BoundingBox(
                x=x, y=y, width=width, height=height, frame=frame,
            )
            tracked_object = tracked_objects_tmp[object_id]
            tracked_object.bounding_boxes.append(bounding_box)
        return tracked_objects_tmp

    def extract_events(self) -> Dict[int, Event]:
        json_events = self.json["events"]
        events_tmp: Dict[int, Event] = {}
        for json_event in json_events:
            event_id = json_event["eventID"]
            event_type = json_event["eventType"]
            duration = json_event["duration"]
            start_frame = json_event["startFrame"]
            end_frame = json_event["endFrame"]
            current_frame = json_event["currentFrame"]
            x = json_event["bbox_lefttop_x"]
            y = json_event["bbox_lefttop_y"]
            width = json_event["bbox_width"]
            height = json_event["bbox_height"]

            if event_id not in events_tmp:
                events_tmp[event_id] = Event(
                    event_id=event_id,
                    duration=duration,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    bounding_boxes=[],
                    event_type=event_type,
                    objects=[]
                )

            bounding_box = BoundingBox(
                x=x, y=y, width=width, height=height, frame=current_frame
            )

            events_tmp[event_id].bounding_boxes.append(bounding_box)

        return events_tmp

    def get_mapping(self) -> Dict[int, List[bool]]:
        """
        :return: A mapping between event ID and the objects which are present in the video.
        One example is:
         {
            321: [True,False,False,True,False]
         }
         This would mean that the event ID 321 has the objects of index 0 and 3 involved
        """
        json_mapping = self.json["mapping"]
        mapping: Dict[int, List[bool]] = {}
        for json_mapping in json_mapping:
            event_id = json_mapping["eventID"]
            relations_to_objects = json_mapping["relationsToObject"]
            boolean_relations_to_objects = [
                bool(relation) for relation in relations_to_objects
            ]
            mapping[event_id] = boolean_relations_to_objects

        return mapping


@dataclass
class Annotations:
    """
    Annotations defines the annotations relative to a video.

    Args:
        - tracked_objects ( List[TrackedObject]) : List of the tracked objects in the video.
        - events ( List[Event]) : List of the events in a video. Each event could contain a list of
            objects present in the video, the same in "tracked_objects"
    """
    tracked_objects: List[TrackedObject]
    events: List[Event]

    @staticmethod
    def from_json(json: dict) -> 'Annotations':
        reader = _AnnotationsReader(json)
        tracked_objects_map = reader.extract_objects()
        tracked_objects = [tracked_objects_map[object_id] for object_id in tracked_objects_map]
        events_map = reader.extract_events()
        mappings = reader.get_mapping()
        for event_id, mappings in mappings.items():
            event = events_map[event_id]
            for index, mapping in enumerate(mappings):
                if mapping:
                    tracked_object = tracked_objects[index]
                    event.objects.append(tracked_object)

        events = [events_map[event_id] for event_id in events_map]
        return Annotations(tracked_objects=tracked_objects, events=events)
