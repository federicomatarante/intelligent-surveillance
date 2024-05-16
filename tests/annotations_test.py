import random
import unittest
from typing import List, Tuple, Dict

from annotations import Annotations, TrackedObject, BoundingBox, Event
from dataset import _DatabaseExtractor


class TestAnnotations(unittest.TestCase):
    ANNOTATIONS_DIR = r"test_dataset/annotations"

    @staticmethod
    def _generate_objects(count: int) -> tuple[list[TrackedObject], list[dict[str, int]]]:
        tracked_objects = []
        for i in range(count):
            duration = random.randint(1, 100)
            tracking_start_frame = random.randint(0, 500)

            tracked_object = TrackedObject(
                object_id=i,
                duration=duration,
                bounding_boxes=TestAnnotations._generate_bounding_boxes(duration, tracking_start_frame),
                object_type=random.randint(1, 5)
            )
            tracked_objects.append(tracked_object)

        tracked_objects_jsons = []
        for tracked_object in tracked_objects:
            current_objects_jsons = [
                {
                    "objectID": tracked_object.object_id,
                    "objectType": tracked_object.object_type,
                    "objectDuration": tracked_object.duration,
                    "currentFrame": box.frame,
                    "bbox_lefttop_x": box.x,
                    "bbox_lefttop_y": box.y,
                    "bbox_width": box.width,
                    "bbox_height": box.height
                } for box in tracked_object.bounding_boxes
            ]
            tracked_objects_jsons.extend(current_objects_jsons)
        return tracked_objects, tracked_objects_jsons

    @staticmethod
    def _generate_events(count: int) -> tuple[list[Event], list[dict[str, int]]]:
        event_list = []
        for i in range(count):
            duration = random.randint(1, 300)
            start_frame = random.randint(0, 500)
            event = Event(
                event_id=i,
                duration=duration,
                start_frame=start_frame,
                end_frame=duration + start_frame,
                bounding_boxes=TestAnnotations._generate_bounding_boxes(duration, start_frame),
                event_type=random.randint(1, 12),
                objects=[]
            )
            event_list.append(event)

        event_jsons = []
        for event in event_list:
            current_event_jsons = [
                {
                    "eventID": event.event_id,
                    "eventType": event.event_type,
                    "duration": event.duration,
                    "startFrame": event.start_frame,
                    "endFrame": event.end_frame,
                    "currentFrame": box.frame,
                    "bbox_lefttop_x": box.x,
                    "bbox_lefttop_y": box.y,
                    "bbox_width": box.width,
                    "bbox_height": box.height

                } for box in event.bounding_boxes
            ]
            event_jsons.extend(current_event_jsons)
        return event_list, event_jsons

    @staticmethod
    def _randomly_generate_mappings(event_list, event_jsons, tracked_object_list, tracked_object_jsons, max_objects):
        mapping_jsons = []
        for event in event_list:
            relations_to_objects = []
            related_objects = 0
            for i in range(len(tracked_object_list)):
                if related_objects == max_objects:
                    break
                related = random.randint(0, 1)
                if related == 1:
                    related_objects += 1
                    related_object = tracked_object_list[i]
                    if related_object not in event.objects:
                        event.objects.append(related_object)

                relations_to_objects.append(related)


            mapping_json = {
                "eventID": event.event_id,
                "eventType": event.event_type,
                "event_duration":event.duration,
                "startFrame": event.start_frame,
                "endFrame": event.end_frame,
                "number_of_obj": related_objects,
                "relationsToObject": relations_to_objects

            }

            mapping_jsons.append(mapping_json)
        return mapping_jsons

    @staticmethod
    def _generate_bounding_boxes(count: int, offset: int) -> List[BoundingBox]:
        bounding_boxes = []
        for j in range(count):
            bounding_box = BoundingBox(
                x=random.randint(0, 2024),
                y=random.randint(0, 2024),
                width=random.randint(0, 500),
                height=random.randint(0, 500),
                frame=offset + j
            )
            bounding_boxes.append(bounding_box)
        return bounding_boxes

    def test_annotations(self):
        configurations = [
            (10, 20, 5),
            (0, 0, 0),
            (20, 0, 0),
            (30, 10, 0)
        ]

        for tot_events, tot_objects, max_objects in configurations:
            events, event_jsons = TestAnnotations._generate_events(tot_events)
            tracked_objects, tracked_object_jsons = TestAnnotations._generate_objects(tot_objects)
            mapping_jsons = TestAnnotations._randomly_generate_mappings(events, event_jsons, tracked_objects,
                                                                        tracked_object_jsons, max_objects)
            annotations_json = {
                "events": event_jsons,
                "objects": tracked_object_jsons,
                "mapping": mapping_jsons
            }
            annotations = Annotations.from_json(annotations_json)
            self.assertEqual(len(annotations.events), tot_events)
            self.assertEqual(len(annotations.tracked_objects), tot_objects)
            for event in annotations.events:
                self.assertLessEqual(len(event.objects), max_objects)

            for event in annotations.events:
                real_event = [e for e in events if event.event_id == e.event_id][0]

                self.assertEqual(event.event_id, real_event.event_id)
                self.assertEqual(event.duration, real_event.duration)
                self.assertEqual(event.start_frame, real_event.start_frame)
                self.assertEqual(event.end_frame, real_event.end_frame)
                self.assertEqual(event.event_type, real_event.event_type)
                self.assertEqual(event.bounding_boxes, real_event.bounding_boxes)
                self.assertEqual(event.objects, real_event.objects)
                self.assertEqual(event,real_event)

    def test_database_annotations(self):
        database_extractor = _DatabaseExtractor(videos_dir="", annotations_dir=self.ANNOTATIONS_DIR)
        annotation_jsons = database_extractor.get_annotations()
        for annotation_json in annotation_jsons.values():
            Annotations.from_json(annotation_json)