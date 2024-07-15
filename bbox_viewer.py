import random
from typing import Dict

import cv2
import numpy as np

from data.annotations import TrackedObject, Event


def _draw_text_with_newlines(frame, text, x, y, color, font_face, font_scale, thickness):
    """
    Function used to draw text on a frame with multi lines text.
    :param frame: frame to draw text on.
    :param text: text to draw.
    :param x: x coordinate of text in frame.
    :param y: y coordinate of text in frame.
    :param color: color of text.
    :param font_face: font face.
    :param font_scale: font scale.
    :param thickness: thickness of text.
    """
    words = text.split(' ')
    num_lines = (len(words) + 3 - 1) // 3
    lines = [' '.join(words[i * 3:(i + 1) * 3]) for i in range(num_lines)]

    line_spacing = int(font_scale * 30)  # Adjust this value to change the line spacing
    y = y - line_spacing * len(lines)
    for i, line in enumerate(lines):
        text_y = y + i * line_spacing
        cv2.putText(frame, line, (x, text_y), font_face, font_scale, color, thickness, cv2.LINE_AA)


def _check_overlap(bbox1: Dict[str, int], bbox2: Dict[str, int], overlap_threshold: float = 0.9):
    """
    Computes the IoU of the two boxes and checks if they overlap.
    :param bbox1: bounding box 1, a dict with format {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        where (x1,y1) are the top left corner of the bounding box and (x2,y2) are the bottom right corner of the bounding box.
    :param bbox2: bounding box 2 with the same format.
    :param overlap_threshold: threshold to be considered a overlap. Must be between 0 and 1.
    """
    x1 = max(bbox1["x1"], bbox2["x1"])
    y1 = max(bbox1["y1"], bbox2["y1"])
    x2 = min(bbox1["x2"], bbox2["x2"])
    y2 = min(bbox1["y2"], bbox2["y2"])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    bbox1_area = (bbox1["x2"] - bbox1["x1"] + 1) * (bbox1["y2"] - bbox1["y1"] + 1)
    bbox2_area = (bbox2["x2"] - bbox2["x1"] + 1) * (bbox2["y2"] - bbox2["y1"] + 1)

    overlap_area = intersection_area / max(0.1, float(bbox1_area + bbox2_area - intersection_area))

    if (bbox1["x1"] >= bbox2["x1"] and bbox1["x2"] <= bbox2["x2"] and
            bbox1["y1"] >= bbox2["y1"] and bbox1["y2"] <= bbox2["y2"]):
        return True
    if (bbox2["x1"] >= bbox1["x1"] and bbox2["x2"] <= bbox1["x2"] and
            bbox2["y1"] >= bbox1["y1"] and bbox2["y2"] <= bbox1["y2"]):
        return True

    return overlap_area >= overlap_threshold


from typing import List, Dict, Any, Union, Tuple


def _merge_overlapping_bboxes(
        items: Union[List[TrackedObject], List[Event]],
        overlap_threshold: float = 0.9,
        is_event: bool = False
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, Tuple[int, int, int]]]:
    """
    Merges all the overlapping bounding boxes for tracked objects or events.

    :param items: List of tracked objects or events to check.
    :param overlap_threshold: Threshold of IoU to be considered an overlap. Must be between 0 and 1.
    :param is_event: Boolean flag to indicate if items are events (True) or tracked objects (False).
    :return: A tuple (bboxes_data, colors) where bboxes_data contains the merged bounding boxes
             and colors contains the color for each unique ID.
    """
    bboxes_data: Dict[int, List[Dict[str, Any]]] = {}
    colors: Dict[str, Tuple[int, int, int]] = {}

    for item in items:
        for bbox in item.bounding_boxes:
            frame = bbox.frame
            if frame not in bboxes_data:
                bboxes_data[frame] = []

            new_bbox = {
                "x1": bbox.x1,
                "x2": bbox.x2,
                "y1": bbox.y1,
                "y2": bbox.y2,
                "id": str(item.event_id if is_event else item.track_id),
                "label": item.label_name
            }

            existing_bbox = next(
                (existing for existing in bboxes_data[frame]
                 if _check_overlap(existing, new_bbox, overlap_threshold)),
                None
            )

            if existing_bbox:
                existing_bbox.update({
                    "x1": min(existing_bbox["x1"], new_bbox["x1"]),
                    "x2": max(existing_bbox["x2"], new_bbox["x2"]),
                    "y1": min(existing_bbox["y1"], new_bbox["y1"]),
                    "y2": max(existing_bbox["y2"], new_bbox["y2"]),
                    "label": f"{existing_bbox['label']}, {new_bbox['label']}",
                    "id": f"{existing_bbox['id']}, {new_bbox['id']}"
                })
                bbox_id = existing_bbox["id"]
            else:
                bboxes_data[frame].append(new_bbox)
                bbox_id = new_bbox["id"]

            if bbox_id not in colors:
                colors[bbox_id] = (random.randrange(256), random.randrange(256), random.randrange(256))

    return bboxes_data, colors


def _merge_overlapping_bboxes_objects(tracked_objects: List[TrackedObject], overlap_threshold=0.9):
    """
    Merges all the overlapping objects into one object.
    :param tracked_objects: list of tracked objects to check.
    :param overlap_threshold: threshold of IoU to be considered an overlap. Must be between 0 and 1.
    :return: a tuple (Dict,colors) where the first element contains the merged objects.
    """

    return _merge_overlapping_bboxes(tracked_objects, overlap_threshold, is_event=False)


def _merge_overlapping_bboxes_events(events, overlap_threshold=0.9):
    """
    Merges all the overlapping events into one event.
    :param events: list of events to check.
    :param overlap_threshold: threshold of IoU to be considered an overlap. Must be between 0 and 1.
    :return: a tuple (Dict,colors) where the first element contains the merged objects.
    """
    return _merge_overlapping_bboxes(events, overlap_threshold, is_event=True)


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
        events_bboxes_data, events_colors = _merge_overlapping_bboxes_events(events, 0.70)
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
        events_bboxes_data, events_colors = _merge_overlapping_bboxes_events(events, 0.70)
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
