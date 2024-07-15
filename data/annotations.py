from dataclasses import dataclass
from typing import List, Dict, Tuple, Any


@dataclass
class BoundingBox:
    """
    BoundingBox defines the current bounding box of an object or event.

    Args:
        x1 (int): x coordinate of the left-top point of the bounding box.
        y1 (int): y coordinate of the left-top point of the bounding box.
        x2 (int): x coordinate of the right-bottom point of the bounding box.
        y2 (int): y coordinate of the right-bottom point of the bounding box.
        frame (int): frame number of the current bounding box relative to the video beginning.
        occlusion (int): occlusion level of the bounding box. Possible values are:
            - 0: 'none'
            - 1: 'partially'
            - 2: 'mostly'
    """
    x1: int
    y1: int
    x2: int
    y2: int
    frame: int
    occlusion: int

    @staticmethod
    def get_occlusion_literal(level: int) -> str:
        occlusion_levels = {
            0: 'none',
            1: 'partially',
            2: 'mostly',
        }
        return occlusion_levels[level]

    @property
    def width(self) -> int:
        """
        :return: The width of the bounding box.
        """
        return self.x2 - self.x1 + 1

    @property
    def height(self) -> int:
        """
        :return: The height of the bounding box.
        """
        return self.y2 - self.y1 + 1

    @staticmethod
    def calculate_iou(box1: 'BoundingBox', box2: 'BoundingBox'):
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        union_area = box1_area + box2_area - intersection_area
        if union_area == 0:
            return 0
        return intersection_area / union_area

    def to_json(self):
        """
        Converts the current BoundingBox object in a dict containint its properties.
        :return: A dict in the following structure:
            {
                'x1': int,
                'x2': int,
                'y1': int,
                'y2': int,
                'frame': int,
                'occlusion': int

            }
        """
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'frame': self.frame,
            'occlusion': self.occlusion,
        }

    @classmethod
    def from_json(cls, bbox_json: Dict[str, Any]) -> 'BoundingBox':
        """
        Creates a BoundingBox object from a JSON dictionary containing bounding box information.
        :param bbox_json: A Dictionary which has to contain:
            - 'x1': the x coordinate of the left-top point of the bounding box.
            - 'y1': the y coordinate of the left-top point of the bounding box.
            - 'x2': the x coordinate of the right-bottom point of the bounding box.
            - 'y2': the y coordinate of the right-bottom point of the bounding box.
            - 'frame': the frame number of the current bounding box.
            - 'occlusion': the occlusion level of the bounding box.
            All integers.
        :return: A BoundingBox object.
        """
        return BoundingBox(
            x1=bbox_json['x1'],
            y1=bbox_json['y1'],
            x2=bbox_json['x2'],
            y2=bbox_json['y2'],
            frame=bbox_json['frame'],
            occlusion=bbox_json['occlusion'],
        )

    def copy(self):
        """
        Returns an exact copy of the object.
        """
        return BoundingBox.from_json(self.to_json())


tracked_object_labels = {
    0: 'Dumpster',
    1: 'Door',
    2: 'Prop',
    3: 'Push_Pulled_Object',
    4: 'Person',
    5: 'Animal',
    6: 'Construction_Vehicle',
    7: 'Construction_Barrier',
    8: 'Vehicle',
    9: 'Tree',
    10: 'Parking_Meter',
    11: 'Bike',
    12: 'Articulated_Infrastructure',
    13: 'Other',
}


@dataclass
class TrackedObject:
    """
    Object defines a track of an object in a video.

    Args:
        track_id (int): unique id of the object ( in the current video clip ).
        bounding_boxes (List[BoundingBox]): list of all the bounding boxes of the object, from the first frame of the tracking
            to the last frame of the tracking. len(bounding_boxes) should be equal to "duration".
        label(int): type of the object. The possible values are:
            - 0: Dumpster
            - 1: Door
            - 2: Prop
            - 3: Push_Pulled_Object
            - 4: Person
            - 5: Animal
            - 6: Construction_Vehicle
            - 7: Construction_Barrier
            - 8: Vehicle
            - 9: Tree
            - 10: Parking_Meter
            - 11: Bike
            - 12: Articulated_Infrastructure
            - 13: Other
    """
    track_id: int
    bounding_boxes: List[BoundingBox]
    label: int

    def __post_init__(self):
        self.bounding_boxes = sorted(self.bounding_boxes, key=lambda x: x.frame)

    def __hash__(self):
        return hash(self.track_id)

    def __eq__(self, other):
        if isinstance(other, TrackedObject):
            return self.track_id == other.track_id
        return False

    @property
    def label_name(self):
        return self.get_label_literal(self.label)

    @staticmethod
    def get_label_literal(label: int) -> str:
        """
        Converts the label integer to a label literal.
        """
        return tracked_object_labels[label]

    @staticmethod
    def get_label_number(level: str) -> int:
        """
        :param level: the literal label.
        :return: the corresponding literal number.
        """
        for key, label in tracked_object_labels.items():
            if label == level:
                return key
        raise ValueError(f'Label {level} is not valid')

    @property
    def start_frame(self):
        """
        The first frame of the tracking.
        :return:
        """
        return self.bounding_boxes[0].frame

    @property
    def end_frame(self):
        """
        The last frame of the tracking.
        """
        return self.bounding_boxes[-1].frame

    @property
    def duration(self) -> int:
        """
        :return: the duration of the track in frames.
        """
        first_bb = self.bounding_boxes[0]
        last_bb = self.bounding_boxes[-1]
        return last_bb.frame - first_bb.frame + 1

    def get_bounding_box(self, frame: int) -> BoundingBox:
        """
        Gets the bounding box of the object relative to the video beginning.
        :param frame: frame number of the current bounding box relative to the video beginning.
        :return: BoundingBox object of the current bounding box.
        """
        for bounding_box in self.bounding_boxes:
            if bounding_box.frame == frame:
                return bounding_box
        raise ValueError(f'Frame {frame} is not valid')

    def to_json(self):
        """
        Converts the current TrackedObject into a dict containing its properties.
        :return: A dict with the following structure:
         {
            'track_id': int,
            'label': int,
            'bounding_boxes': [
                bb1_json,bb2_json,... ( See informationa bout BoundingBox Json )
            ]

         }
        """
        bounding_boxes_json = []
        for bbox in self.bounding_boxes:
            bounding_boxes_json.append(bbox.to_json())
        return {
            'track_id': self.track_id,
            'label': self.label,
            'bounding_boxes': bounding_boxes_json
        }

    @staticmethod
    def from_json(json_data: Dict[str, Any]):
        """
        Creates a TrackedObject given a JSON containing its properties.
        :param json_data: A dict with the following structure:
         {
            'track_id': int,
            'label': int,
            'bounding_boxes': [
                bb1_json,bb2_json,... ( See informationa bout BoundingBox Json )
            ]

         }
        """
        bboxes = []
        for bbox_json in json_data['bounding_boxes']:
            bbox = BoundingBox.from_json(bbox_json)
            bboxes.append(bbox)
        return TrackedObject(
            track_id=json_data["track_id"],
            label=json_data["label"],
            bounding_boxes=bboxes
        )

    def copy(self):
        """
        Returns an exact copy of the object.
        """
        return TrackedObject.from_json(self.to_json())


event_types = {
    0: 'None',
    1: 'activity_running',
    2: 'specialized_miscellaneous',
    3: 'Drop',
    4: 'Transport_HeavyCarry',
    5: 'specialized_throwing',
    6: 'specialized_talking_phone',
    7: 'activity_crouching',
    8: 'Opening',
    9: 'vehicle_moving',
    10: 'Person_Person_Interaction',
    11: 'Riding',
    12: 'Interacts',
    13: 'Closing_Trunk',
    14: 'Talking',
    15: 'vehicle_turning_right',
    16: 'Open_Trunk',
    17: 'specialized_texting_phone',
    18: 'Exiting',
    19: 'vehicle_turning_left',
    20: 'Closing',
    21: 'activity_gesturing',
    22: 'Entering',
    23: 'activity_sitting',
    24: 'specialized_using_tool',
    25: 'Misc',
    26: 'activity_walking',
    27: 'PickUp',
    28: 'Loading',
    29: 'DropOff_Person_Vehicle',
    30: 'vehicle_u_turn',
    31: 'activity_carrying',
    32: 'Push',
    33: 'Unloading',
    34: 'PickUp_Person_Vehicle',
    35: 'vehicle_starting',
    36: 'specialized_umbrella',
    37: 'vehicle_stopping',
    38: 'activity_standing',
    39: 'Object_Transfer',
    40: 'SetDown',
    41: 'Pull',
}


@dataclass
class Event:
    """
    Event defines a track of an event.
    Args:
        event_id (int): unique id of the event ( in the current video clip ).
        start_frame (int): frame number of the start of the event.
        end_frame (int): frame number of the end of the event.
        objects Dict[TrackedObject, Tuple[int, int]]: list of all the objects involved in the event.
            One example of record is:
                [tracked_object]: (start_frame, end_frame)
            Where:
                - start_frame: first frame number of the video in which there is the Object.
                - end_frame: last frame number of the video in which there is the Object.
        event_type (int): type of the event. The possible values are:
            - 0: Pull
            - 1: activity_running
            - 2: specialized_miscellaneous
            - 3: Drop
            - 4: Transport_HeavyCarry
            - 5: specialized_throwing
            - 6: specialized_talking_phone
            - 7: activity_crouching
            - 8: Opening
            - 9: vehicle_moving
            - 10: Person_Person_Interaction
            - 11: Riding
            - 12: Interacts
            - 13: Closing_Trunk
            - 14: Talking
            - 15: vehicle_turning_right
            - 16: Open_Trunk
            - 17: specialized_texting_phone
            - 18: Exiting
            - 19: vehicle_turning_left
            - 20: Closing
            - 21: activity_gesturing
            - 22: Entering
            - 23: activity_sitting
            - 24: specialized_using_tool
            - 25: Misc
            - 26: activity_walking
            - 27: PickUp
            - 28: Loading
            - 29: DropOff_Person_Vehicle
            - 30: vehicle_u_turn
            - 31: activity_carrying
            - 32: Push
            - 33: Unloading
            - 34: PickUp_Person_Vehicle
            - 35: vehicle_starting
            - 36: specialized_umbrella
            - 37: vehicle_stopping
            - 38: activity_standing
            - 39: Object_Transfer
            - 40: SetDown
    """
    event_id: int
    start_frame: int
    end_frame: int
    event_type: int
    objects: Dict[TrackedObject, Tuple[int, int]]

    @property
    def tracked_objects(self) -> List[TrackedObject]:
        """
        :return: list of all the objects involved in the event.
        """
        return list(self.objects.keys())

    @staticmethod
    def get_label_number(label: str):
        """
        Converts a label to its numeric value.
        :param label: the label to convert.
        :return: the converted label.
        """
        for number, event_type in event_types.items():
            if label == event_type:
                return number
        raise ValueError(f"{label} is not a valid label.")

    @staticmethod
    def get_label_name(event_type: int):
        """
        Converts a label to its string value.
        :param event_type: the label to convert.
        :return: the converted label.
        """
        return event_types[event_type]

    @property
    def label_name(self):
        """
        The name of the label.
        """
        return event_types[self.event_type]

    @property
    def duration(self) -> int:
        """
        The duration of the event in frames.
        """
        return self.end_frame - self.start_frame

    @property
    def bounding_boxes(self) -> List[BoundingBox]:
        """
        A list of the bounding boxes of the event.
        It's computed through the bounding boxes of the TrackedObjects in the event.
        """
        objects_bboxes_data = {}  #
        # Mapping each frame for its bounding boxes
        for track, (obj_first_frame, obj_last_frame) in self.objects.items():
            for bbox in track.bounding_boxes:
                if bbox.frame < obj_first_frame:
                    continue
                if bbox.frame > obj_last_frame:
                    break
                if bbox.frame not in objects_bboxes_data:
                    objects_bboxes_data[bbox.frame] = []
                objects_bboxes_data[bbox.frame].append(bbox)
        # Finding largest width and height
        max_bb_width = 0
        max_bb_height = 0
        for bboxes in objects_bboxes_data.values():
            min_x1 = min(bboxes, key=lambda b: b.x1).x1
            max_x2 = max(bboxes, key=lambda b: b.x2).x2
            min_y1 = min(bboxes, key=lambda b: b.y1).y1
            max_y2 = max(bboxes, key=lambda b: b.y2).y2
            width = max_x2 - min_x1
            height = max_y2 - min_y1
            if width > max_bb_width:
                max_bb_width = width
            if height > max_bb_height:
                max_bb_height = height
        bounding_boxes: List[BoundingBox] = []
        # Creating new bounding boxes
        for frame, bboxes in objects_bboxes_data.items():
            min_x1 = min(bboxes, key=lambda b: b.x1).x1
            min_y1 = min(bboxes, key=lambda b: b.y1).y1
            x2 = min_x1 + max_bb_width
            y2 = min_y1 + max_bb_height
            bounding_boxes.append(
                BoundingBox(frame=frame, x1=min_x1, y1=min_y1, x2=x2, y2=y2, occlusion=0)
            )
        return bounding_boxes

    def __hash__(self):
        return hash(self.event_id)

    @staticmethod
    def from_json(json: Dict[Any, Any]) -> 'Event':
        """
        Creates an 'Event' object from a JSON.
        :param json: a dictionary in the following format:
        {
            'event_id': int,
            'start_frame': int,
            'end_frame': int,
            'event_type': int,
            'objects': [
                Json_obj_1, Json_obj_2, ... ( See JSON of TrackedObject )
            ]
        }
        :return: an 'Event' object.
        """
        objects = {}
        for track, (start_frame, end_frame) in json['objects']:
            objects.update(
                {
                    TrackedObject.from_json(track): (start_frame, end_frame)
                }
            )
        return Event(
            event_id=json["event_id"],
            start_frame=json['start_frame'],
            end_frame=json['end_frame'],
            event_type=json['event_type'],
            objects=objects,
        )

    def to_json(self) -> Dict[Any, Any]:
        """
        Converts the 'Event' object in a JSON containing all its properties.
        :return: a dictionary containing all the properties of the 'Event' object:
            {
                'event_id': int,
                'start_frame': int,
                'end_frame': int,
                'event_type': int,
                'objects': [
                    Json_obj_1, Json_obj_2, ... ( See JSON of TrackedObject )
                ]
            }
            """
        objects_jsons = tuple(
            (obj.to_json(), (start, end)) for obj, (start, end) in self.objects.items()
        )
        event_json = {
            'event_id': self.event_id,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'event_type': self.event_type,
            'objects': objects_jsons,
        }
        return event_json

    def copy(self):
        """
        Returns an exact copy of the object.
        """
        return Event.from_json(self.to_json())


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

    def to_json(self):
        """
        Converts the 'Annotations' object in a JSON containing all its properties.
        :return: a dictionary of the following format:
        {
            'tracked_objects': [
                tracked_object_1, tracked_object_2, ... ( See JSON of TrackedObject )
            ],
            'events': [
                event_1, event_2, ... ( See JSON of Event )
            ]
         }
        """
        return {
            'tracked_objects': [
                obj.to_json() for obj in self.tracked_objects
            ],
            'events': [
                event.to_json() for event in self.events
            ],
        }

    @staticmethod
    def from_json(json: Dict[Any, Any]) -> 'Annotations':
        """
        Creates a 'Annotations' object from a JSON containing all its properties.
        :param json: a dictionary of the following format:
         {
            'tracked_objects': [
                tracked_object_1, tracked_object_2, ... ( See JSON of TrackedObject )
            ],
            'events': [
                event_1, event_2, ... ( See JSON of Event )
            ]
         }
        :return:
        """
        return Annotations(
            tracked_objects=[
                TrackedObject.from_json(obj_json) for obj_json in json["tracked_objects"]
            ],
            events=[
                Event.from_json(ev_json) for ev_json in json["events"]
            ]
        )

    def copy(self):
        """
        Returns an exact copy of the object.
        """
        return Annotations.from_json(self.to_json())
