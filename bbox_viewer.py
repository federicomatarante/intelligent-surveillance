import random
from typing import List

import cv2
import numpy as np

from dataset.code.annotations import TrackedObject, Event, Annotations
from dataset.code.annotations_parser import AnnotationsReader
from dataset.code.database import VideosDatabase


def _draw_text_with_newlines(frame, text, x, y, color, font_face, font_scale, thickness):
    words = text.split(' ')
    num_lines = (len(words) + 3 - 1) // 3
    lines = [' '.join(words[i * 3:(i + 1) * 3]) for i in range(num_lines)]

    line_spacing = int(font_scale * 30)  # Adjust this value to change the line spacing
    y = y - line_spacing * len(lines)
    for i, line in enumerate(lines):
        text_y = y + i * line_spacing
        cv2.putText(frame, line, (x, text_y), font_face, font_scale, color, thickness, cv2.LINE_AA)


def _check_overlap(bbox1, bbox2, overlap_threshold=0.9):
    """
    Calcola l'area di sovrapposizione tra due bounding box e restituisce True
    se l'area di sovrapposizione supera la soglia specificata o se un bounding box
    è completamente contenuto nell'altro.
    """
    x1 = max(bbox1["x1"], bbox2["x1"])
    y1 = max(bbox1["y1"], bbox2["y1"])
    x2 = min(bbox1["x2"], bbox2["x2"])
    y2 = min(bbox1["y2"], bbox2["y2"])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    bbox1_area = (bbox1["x2"] - bbox1["x1"] + 1) * (bbox1["y2"] - bbox1["y1"] + 1)
    bbox2_area = (bbox2["x2"] - bbox2["x1"] + 1) * (bbox2["y2"] - bbox2["y1"] + 1)

    overlap_area = intersection_area / max(0.1, float(bbox1_area + bbox2_area - intersection_area))

    # Verifica se un bounding box è completamente contenuto nell'altro
    if (bbox1["x1"] >= bbox2["x1"] and bbox1["x2"] <= bbox2["x2"] and
            bbox1["y1"] >= bbox2["y1"] and bbox1["y2"] <= bbox2["y2"]):
        return True
    if (bbox2["x1"] >= bbox1["x1"] and bbox2["x2"] <= bbox1["x2"] and
            bbox2["y1"] >= bbox1["y1"] and bbox2["y2"] <= bbox1["y2"]):
        return True

    return overlap_area >= overlap_threshold


def _merge_overlapping_bboxes_objects(tracked_objects: List[TrackedObject], overlap_threshold=0.9):
    bboxes_data = {}
    colors = {}

    for tracked_object in tracked_objects:
        for bbox in tracked_object.bounding_boxes:
            frame = bbox.frame
            if frame not in bboxes_data:
                bboxes_data[frame] = []

            existing_bbox = None
            for existing in bboxes_data[frame]:
                if _check_overlap(existing, {
                    "x1": bbox.x1,
                    "x2": bbox.x2,
                    "y1": bbox.y1,
                    "y2": bbox.y2
                }, overlap_threshold):
                    existing_bbox = existing
                    break

            if existing_bbox:
                existing_bbox["x1"] = min(existing_bbox["x1"], bbox.x1)
                existing_bbox["x2"] = max(existing_bbox["x2"], bbox.x2)
                existing_bbox["y1"] = min(existing_bbox["y1"], bbox.y1)
                existing_bbox["y2"] = max(existing_bbox["y2"], bbox.y2)
                existing_bbox["label"] += f", {tracked_object.label_name}"
                existing_bbox["id"] += f", {tracked_object.track_id}"
                colors[existing_bbox["id"]] = colors.get(existing_bbox["id"], (
                    random.randrange(256), random.randrange(256), random.randrange(256)))
            else:
                new_bbox = {
                    "x1": bbox.x1,
                    "x2": bbox.x2,
                    "y1": bbox.y1,
                    "y2": bbox.y2,
                    "id": str(tracked_object.track_id),
                    "label": tracked_object.label_name
                }
                bboxes_data[frame].append(new_bbox)
                colors[new_bbox["id"]] = (random.randrange(256), random.randrange(256), random.randrange(256))
    return bboxes_data, colors


def merge_overlapping_bboxes_events(events, overlap_threshold=0.9):
    bboxes_data = {}
    colors = {}

    for event in events:
        for bbox in event.bounding_boxes:
            frame = bbox.frame
            if frame not in bboxes_data:
                bboxes_data[frame] = []

            existing_bbox = None
            for existing in bboxes_data[frame]:
                if _check_overlap(existing, {
                    "x1": bbox.x1,
                    "x2": bbox.x2,
                    "y1": bbox.y1,
                    "y2": bbox.y2
                }, overlap_threshold):
                    existing_bbox = existing
                    break

            if existing_bbox:
                existing_bbox["x1"] = min(existing_bbox["x1"], bbox.x1)
                existing_bbox["x2"] = max(existing_bbox["x2"], bbox.x2)
                existing_bbox["y1"] = min(existing_bbox["y1"], bbox.y1)
                existing_bbox["y2"] = max(existing_bbox["y2"], bbox.y2)
                existing_bbox["label"] += f", {event.label_name}"
                existing_bbox["id"] += f", {event.event_id}"
                colors[existing_bbox["id"]] = colors.get(existing_bbox["id"], (
                    random.randrange(256), random.randrange(256), random.randrange(256)))
            else:
                new_bbox = {
                    "x1": bbox.x1,
                    "x2": bbox.x2,
                    "y1": bbox.y1,
                    "y2": bbox.y2,
                    "id": str(event.event_id),
                    "label": event.label_name
                }
                bboxes_data[frame].append(new_bbox)
                colors[new_bbox["id"]] = (random.randrange(256), random.randrange(256), random.randrange(256))
    return bboxes_data, colors


def show_tracked_image(image: np.ndarray, events: List[Event] = None, tracked_objects: List[TrackedObject] = None):
    """
    Given an image, it shows all the tracked objects in it as bounding boxes.
    :param image: the image to be displayed.
    :param events: the list of events to be displayed as bounding boxes.
    :param tracked_objects: the list of objects to be displayed as bounding boxes.
    """

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    bboxes_data, colors = {}, {}

    if events:
        events_bboxes_data, events_colors = merge_overlapping_bboxes_events(events, 0.70)
        bboxes_data.update(events_bboxes_data)
        colors.update(events_colors)

    if tracked_objects:
        objects_bboxes_data, objects_colors = _merge_overlapping_bboxes_objects(tracked_objects, 0.70)
        bboxes_data.update(objects_bboxes_data)
        colors.update(objects_colors)

    for bbox in bboxes_data.values():
        for box in bbox:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            color = colors[box['id']]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            _draw_text_with_newlines(image, "Labels: " + box["label"], x1, y1, color, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.putText(image, "IDs: " + str(box["id"]), (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_tracked_video(video_frames: np.ndarray | List[np.ndarray], events: List[Event] = None,
                       tracked_objects: List[TrackedObject] = None, fps=30):
    """
    Given a video, it shows all the tracked objects in it as bounding boxes.
    Press q to close a running video.
    :param video_frames: the video to be displayed ( In RGB format ).
    :param events: the list of events to be displayed as bounding boxes.
    :param tracked_objects: the list of objects to be displayed as bounding boxes.
    :param fps: the FPS of the video to be displayed.
    """
    bboxes_data, colors = {}, {}
    if events:
        events_bboxes_data, events_colors = merge_overlapping_bboxes_events(events, 0.70)
        bboxes_data.update(events_bboxes_data)
        colors.update(events_colors)
    if tracked_objects:
        objects_bboxes_data, objects_colors = _merge_overlapping_bboxes_objects(tracked_objects, 0.70)
        bboxes_data.update(objects_bboxes_data)
        colors.update(objects_colors)

    total_frames = len(video_frames)
    frame_idx = 0
    while frame_idx < total_frames:
        frame = video_frames[frame_idx]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if frame_idx in bboxes_data:
            for bbox in bboxes_data[frame_idx]:
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                color = colors[bbox['id']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                _draw_text_with_newlines(frame, "Labels: " + bbox["label"], x1, y1, color, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.5,
                                         2)
                cv2.putText(frame, "IDs: " + str(bbox["id"]), (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Frame: {frame_idx}/{total_frames}"
        cv2.putText(frame, text, (10, 30), font, 1, 255, 2, cv2.LINE_AA)
        cv2.imshow('Video', frame)
        key = cv2.waitKey(fps)
        if key == ord('q'):
            break

        frame_idx += 1

    cv2.destroyAllWindows()


def main():  # TODO remove all
    annotations_path = r'C:\Users\feder\PycharmProjects\intelligent-surveillance\dataset\kml_annotations\all'
    videos_path = r'C:\Users\feder\PycharmProjects\intelligent-surveillance\dataset\videos'
    reader = AnnotationsReader(annotations_path)
    videos_db = VideosDatabase(videos_path)
    while True:
        random_id = random.choice(reader.annotation_ids)  # VIRAT_S_040103_06_000836_000909
        random_id = "VIRAT_S_000200_00_000100_000171"
        if random_id in videos_db.get_ids():
            break
    print(random_id)  # VIRAT_S_000200_00_000100_000171
    annotations: Annotations = reader.read_one_annotation(random_id)
    video = videos_db.read(random_id)

    show_tracked_video(video)
    show_tracked_video(video, tracked_objects=annotations.tracked_objects)
    show_tracked_video(video, events=annotations.events)


if __name__ == '__main__':
    main()
