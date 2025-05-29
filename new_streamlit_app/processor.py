import cv2
from collections import defaultdict
import math
from model_loader import load_yolov8_model

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def process_video(input_path, output_path):
    model = load_yolov8_model()
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    color_mapping = {
        'car': (255, 0, 0),
        'bus_l': (0, 255, 0),
        'bus_s': (0, 0, 255),
        'truck_s': (255, 255, 0),
        'truck_m': (255, 0, 255),
        'truck_l': (0, 255, 255),
        'truck_xl': (128, 0, 128)
    }

    next_object_id = 0
    tracked_objects = {}  # ID -> (cx, cy)
    object_boxes = {}     # ID -> (x1, y1, x2, y2, label)
    counted_ids = defaultdict(set)

    distance_threshold = 50

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        current_centroids = []
        current_labels = []
        current_boxes = []

        for box, class_id in zip(boxes, class_ids):
            label = results[0].names[int(class_id)]
            if label not in color_mapping:
                continue
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            current_centroids.append((cx, cy))
            current_labels.append(label)
            current_boxes.append((x1, y1, x2, y2))

        new_tracked_objects = {}
        new_object_boxes = {}
        matched_old_ids = set()

        for i, (cx, cy) in enumerate(current_centroids):
            min_dist = float("inf")
            min_id = None

            for obj_id, (prev_cx, prev_cy) in tracked_objects.items():
                dist = euclidean_distance((cx, cy), (prev_cx, prev_cy))
                if dist < distance_threshold and obj_id not in matched_old_ids:
                    if dist < min_dist:
                        min_dist = dist
                        min_id = obj_id

            if min_id is not None:
                new_tracked_objects[min_id] = (cx, cy)
                new_object_boxes[min_id] = (*current_boxes[i], current_labels[i])
                matched_old_ids.add(min_id)
            else:
                new_tracked_objects[next_object_id] = (cx, cy)
                new_object_boxes[next_object_id] = (*current_boxes[i], current_labels[i])
                counted_ids[current_labels[i]].add(next_object_id)
                next_object_id += 1

        tracked_objects = new_tracked_objects
        object_boxes = new_object_boxes

        # Draw bounding boxes and info
        for obj_id, (x1, y1, x2, y2, label) in object_boxes.items():
            color = color_mapping[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ID:{obj_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cx, cy = tracked_objects[obj_id]
            cv2.circle(frame, (cx, cy), 4, color, -1)

        # Display class counts
        y_offset = 20
        for label, ids in counted_ids.items():
            count = len(ids)
            cv2.putText(frame, f"{label}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_mapping[label], 2)
            y_offset += 25

        out.write(frame)

    cap.release()
    out.release()

    counts_summary = {label: len(ids) for label, ids in counted_ids.items()}
    return counts_summary
