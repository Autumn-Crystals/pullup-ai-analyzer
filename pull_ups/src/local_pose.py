"""Local Pose Estimation Models: YOLOv8-Pose and MediaPipe Pose.

Provides full COCO-17 compatible payloads for local model evaluation and comparison
against cloud ViTPose models.
"""

from __future__ import annotations

import time
from pathlib import Path
import cv2
import numpy as np

from .skeleton import KPT_NAMES

# Mapping MediaPipe 33 landmarks -> COCO-17 indices
# MediaPipe indices:
#  0: nose
#  2: left_eye
#  5: right_eye
#  7: left_ear
#  8: right_ear
# 11: left_shoulder
# 12: right_shoulder
# 13: left_elbow
# 14: right_elbow
# 15: left_wrist
# 16: right_wrist
# 23: left_hip
# 24: right_hip
# 25: left_knee
# 26: right_knee
# 27: left_ankle
# 28: right_ankle
MEDIAPIPE_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def run_yolov8_pose(video_path: Path, model_name: str = "yolov8n-pose.pt", conf_thresh: float = 0.25) -> dict:
    """Run YOLOv8-pose locally on video and return a COCO-17 compliant payload."""
    from ultralytics import YOLO

    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    items = []
    frames = []
    frame_idx = 0
    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append({"frame_id": frame_idx})
        results = model(frame, conf=conf_thresh, verbose=False)

        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            kpts = res.keypoints

            if boxes is not None and kpts is not None and len(boxes) > 0:
                # Find the largest detection box (the athlete)
                best_i = 0
                max_area = 0.0
                for i in range(len(boxes)):
                    xywh = boxes.xywhn[i].cpu().numpy()
                    area = xywh[2] * xywh[3]
                    if area > max_area:
                        max_area = area
                        best_i = i

                xywhn = boxes.xywhn[best_i].cpu().numpy().tolist()
                kpts_data = kpts.data[best_i].cpu().numpy() # [17, 3] or [17, 2]

                kpts_xy = []
                kpts_score = []
                for pt in kpts_data:
                    # pt can be [x, y, conf] in pixel coordinates or normalized
                    x_norm = float(pt[0]) / width if width > 0 else 0.0
                    y_norm = float(pt[1]) / height if height > 0 else 0.0
                    score = float(pt[2]) if len(pt) > 2 else 1.0
                    kpts_xy.append([round(x_norm, 4), round(y_norm, 4)])
                    kpts_score.append(round(score, 4))

                items.append({
                    "frame_id": frame_idx,
                    "track_id": 0,
                    "bbox_xywh": [round(c, 4) for c in xywhn],
                    "kpts_xy": kpts_xy,
                    "kpts_score": kpts_score,
                })

        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - t0

    payload = {
        "content": {
            "object": "vid.pose.kpts",
            "kpts_labels": KPT_NAMES,
            "frames": frames,
            "items": items,
        },
        "meta": {
            "model": model_name,
            "total_frames": frame_idx,
            "fps": fps,
            "elapsed_seconds": round(elapsed, 3),
            "inference_fps": round(frame_idx / max(elapsed, 0.001), 2),
        }
    }
    return payload


def run_mediapipe_pose(video_path: Path, model_asset_path: str = "pose_landmarker_heavy.task") -> dict:
    """Run Google MediaPipe Pose Landmarker locally on video and return a COCO-17 payload."""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    asset_file = Path(model_asset_path)
    if not asset_file.is_file():
        # Fallback to current working directory or ML Project root
        cand = Path(__file__).resolve().parents[2] / model_asset_path
        if cand.is_file():
            asset_file = cand

    base_options = python.BaseOptions(model_asset_path=str(asset_file))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    items = []
    frames = []
    frame_idx = 0
    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append({"frame_id": frame_idx})
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = detector.detect(mp_image)

        if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
            landmarks = detection_result.pose_landmarks[0]

            kpts_xy = []
            kpts_score = []
            xs, ys = [], []

            for coco_idx in MEDIAPIPE_TO_COCO17:
                if coco_idx < len(landmarks):
                    lm = landmarks[coco_idx]
                    x = max(0.0, min(1.0, float(lm.x)))
                    y = max(0.0, min(1.0, float(lm.y)))
                    vis = float(getattr(lm, "visibility", 1.0) or 1.0)
                    kpts_xy.append([round(x, 4), round(y, 4)])
                    kpts_score.append(round(vis, 4))
                    xs.append(x)
                    ys.append(y)
                else:
                    kpts_xy.append([0.0, 0.0])
                    kpts_score.append(0.0)

            if xs and ys:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                bbox = [
                    round(min_x, 4),
                    round(min_y, 4),
                    round(max(0.01, max_x - min_x), 4),
                    round(max(0.01, max_y - min_y), 4),
                ]
            else:
                bbox = [0.0, 0.0, 1.0, 1.0]

            items.append({
                "frame_id": frame_idx,
                "track_id": 0,
                "bbox_xywh": bbox,
                "kpts_xy": kpts_xy,
                "kpts_score": kpts_score,
            })

        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - t0

    payload = {
        "content": {
            "object": "vid.pose.kpts",
            "kpts_labels": KPT_NAMES,
            "frames": frames,
            "items": items,
        },
        "meta": {
            "model": "MediaPipe BlazePose (Heavy)",
            "total_frames": frame_idx,
            "fps": fps,
            "elapsed_seconds": round(elapsed, 3),
            "inference_fps": round(frame_idx / max(elapsed, 0.001), 2),
        }
    }
    return payload
