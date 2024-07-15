from typing import List

import numpy as np
import torch
from ultralytics import YOLO

from data.annotations import TrackedObject, BoundingBox


class ImageTracker:
    def __init__(self, model: YOLO):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self.model = model

    def track(self, video: List[np.ndarray] | np.ndarray) -> List[TrackedObject]:
        track_history = {}
        for i, frame in enumerate(video):
            results = self.model.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes.xyxy.int().cpu()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                track_ids = []
            labels = results[0].boxes.cls.int().cpu().tolist()
            frame_results = {
                track_id: (tuple(box.numpy().tolist()), label) for box, track_id, label in zip(boxes, track_ids, labels)
            }
            for new_id in track_ids:
                if new_id not in track_history:
                    track_history[new_id] = []
            for track_id, track in track_history.items():
                if track_id in track_ids:
                    track.append({
                        'box': frame_results[track_id][0],
                        'label': frame_results[track_id][1],
                        'frame': i
                    })

        tracked_objects = []
        for track_id, track in track_history.items():
            new_tracks = []
            for i in range(len(track) - 1):
                frame_1 = track[i]["frame"]
                frame_2 = track[i + 1]["frame"]
                new_tracks.append(track[i])
                for j in range(frame_1 + 1, frame_2):
                    new_tracks.append({
                        'box': track[i]["box"],
                        'label': track[i]["label"],
                        'frame': j
                    })
            new_tracks.append(track[-1])
            track_history[track_id] = new_tracks

        for track_id, boxes in track_history.items():
            tracked_object = TrackedObject(
                track_id=track_id,
                label=boxes[0]["label"],
                bounding_boxes=[
                    BoundingBox(
                        frame=box["frame"],
                        x1=box["box"][0],
                        y1=box["box"][1],
                        x2=box["box"][2],
                        y2=box["box"][3],
                        occlusion=0
                    ) for box in boxes
                ]
            )
            tracked_objects.append(tracked_object)
        return tracked_objects
