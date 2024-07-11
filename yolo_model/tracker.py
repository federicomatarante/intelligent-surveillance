import cv2
import numpy as np


def multi_object_tracking(frames, model, conf_threshold=0.5, iou_threshold=0.5):
    if not frames or not isinstance(frames[0], np.ndarray):
        print("Input non valido. Assicurati che 'frames' sia una lista di np.ndarray.")
        return {}

    object_ids = {}
    trackers = {}
    next_id = 1

    # Esegui la rilevazione iniziale con YOLOv8 sul primo frame
    results = model(frames[0], conf=conf_threshold, iou=iou_threshold)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            label = model.names[int(cls)]

            tracker = cv2.TrackerKCF_create()
            tracker.init(frames[0], (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

            object_ids[next_id] = {
                "label": label,
                "bboxes": [{
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "frame": 0
                }]
            }
            trackers[next_id] = tracker
            next_id += 1

    # Tracking attraverso i frame rimanenti
    for frame_idx, frame in enumerate(frames[1:], start=1):
        ids_to_remove = []
        for obj_id, tracker in trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                object_ids[obj_id]["bboxes"].append({
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                    "frame": frame_idx
                })
            else:
                ids_to_remove.append(obj_id)

        # Rimuovi i tracker falliti
        for obj_id in ids_to_remove:
            del trackers[obj_id]

        # Esegui nuovamente la rilevazione ogni 30 frame per trovare nuovi oggetti
        if frame_idx % 30 == 0:
            results = model(frame, conf=conf_threshold, iou=iou_threshold)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    label = model.names[int(cls)]

                    # Verifica se l'oggetto è già tracciato
                    new_object = True
                    for existing_id, existing_info in object_ids.items():
                        if existing_id in trackers:
                            last_box = existing_info["bboxes"][-1]
                            if calculate_iou(last_box, {"x1": x1, "y1": y1, "x2": x2, "y2": y2}) > 0.5:
                                new_object = False
                                break

                    if new_object:
                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
                        object_ids[next_id] = {
                            "label": label,
                            "bboxes": [{
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                                "frame": frame_idx
                            }]
                        }
                        trackers[next_id] = tracker
                        next_id += 1

    return object_ids


def calculate_iou(box1, box2):
    # Calcola l'Intersection over Union (IoU) tra due bounding box
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

# Esempio di utilizzo:
# model = YOLO("yolov8n.pt")  # o il percorso al tuo modello personalizzato
# video_frames = ...  # la tua lista di frame numpy
# result = multi_object_tracking(video_frames, model)
