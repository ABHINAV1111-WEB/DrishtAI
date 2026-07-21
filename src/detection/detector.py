"""
detector.py — DrishtAI pipeline, Stage 2 (YOLOv8 vehicle detection)

Consumes frames (either FrameRecords from frame_extractor.py or raw images)
and produces per-frame vehicle detections. Output is deliberately
schema-adjacent: each detection carries timestamp + centroid position so the
tracking stage (Stage 3) only needs to add persistent object_ids.

Design decisions:

1. VEHICLE CLASSES ONLY. COCO ids: 2=car, 3=motorcycle, 5=bus, 7=truck.
   Everything else (pedestrians, bicycles, traffic lights) is filtered out
   at inference time via the `classes=` argument — cheaper than filtering
   afterward, and enforces the locked scope (road accidents, vehicles).

2. NO object_id AT THIS STAGE. Detection is stateless per-frame; assigning
   persistent IDs across frames is the tracker's job (Stage 3). Faking IDs
   here (e.g. detection order) would produce IDs that shuffle between
   frames and poison the motion math.

3. MODEL: yolov8s (small) by default. We started on yolov8n (nano) but it
   dropped clearly-visible vehicles for several frames at the impact moment
   (motion blur / unusual pose). yolov8s holds detection through impact on
   our footage. Swap models via --model if needed; weights auto-download on
   first run (~22 MB for yolov8s, ~6 MB for yolov8n).

4. CONFIDENCE THRESHOLD 0.25 default. CCTV/dashcam footage is often low
   quality; 0.5 (YOLO's usual default) drops too many real vehicles at a
   distance, and even 0.35 was marginal on hard frames. 0.25 keeps distant
   and partially-degraded vehicles. Tune per-clip with --conf if you see
   misses or ghosts.

Usage (standalone, reads a video end-to-end via the frame extractor):
    python src/detection/detector.py data/clip1.mp4 --interval 2 --out detections.json
    python src/detection/detector.py data/clip1.mp4 --interval 2 --out detections.json --annotate annotated/

Usage (as a module, which is how the pipeline will use it):
    from detection.frame_extractor import extract_frames
    from detection.detector import VehicleDetector
    detector = VehicleDetector()
    detections = detector.detect_video("data/clip1.mp4", frame_interval=2)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
from ultralytics import YOLO

# Allow running this file directly (python src/detection/detector.py) as well
# as importing it as part of the src package.
try:
    from .frame_extractor import frame_index_to_timestamp
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from frame_extractor import frame_index_to_timestamp


# COCO class ids for the vehicle classes in scope. Locked to the pitch:
# road vehicles only.
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    """One detected vehicle in one frame. Input to the tracking stage."""
    frame_index: int          # absolute frame index in source video
    timestamp: str            # HH:MM:SS:FF per schema.md
    time_seconds: float       # float time for motion math
    bbox: list[float]         # [x1, y1, x2, y2] pixel corners
    position: list[float]     # [cx, cy] bbox centroid — matches schema "position"
    confidence: float         # detector confidence 0-1
    vehicle_class: str        # "car" | "motorcycle" | "bus" | "truck"


class VehicleDetector:
    def __init__(self, model_name: str = "yolov8s.pt", conf_threshold: float = 0.25):
        self.model = YOLO(model_name)  # auto-downloads weights on first use
        self.conf_threshold = conf_threshold

    def detect_frame(self, frame, frame_index: int, fps: float) -> list[Detection]:
        """Run detection on a single frame (numpy BGR image from OpenCV)."""
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            classes=list(VEHICLE_CLASSES.keys()),  # filter at inference time
            verbose=False,
        )
        ts = frame_index_to_timestamp(frame_index, fps)
        t = frame_index / fps

        detections: list[Detection] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(
                frame_index=frame_index,
                timestamp=ts,
                time_seconds=t,
                bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                position=[round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
                confidence=round(float(box.conf[0]), 3),
                vehicle_class=VEHICLE_CLASSES[int(box.cls[0])],
            ))
        return detections

    def detect_video(
        self,
        video_path: str | Path,
        frame_interval: int = 1,
        annotate_dir: str | Path | None = None,
    ) -> list[Detection]:
        """
        Run detection across a whole video, every `frame_interval`-th frame.

        Reads frames sequentially (same rationale as frame_extractor: seeking
        is codec-unreliable). If annotate_dir is given, saves each processed
        frame with detection boxes drawn — essential for eyeballing whether
        the detector is actually seeing the vehicles in your footage.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            raise RuntimeError(f"Video reports invalid FPS ({fps}); re-encode it first.")

        if annotate_dir is not None:
            annotate_dir = Path(annotate_dir)
            annotate_dir.mkdir(parents=True, exist_ok=True)

        all_detections: list[Detection] = []
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % frame_interval == 0:
                    dets = self.detect_frame(frame, frame_index, fps)
                    all_detections.extend(dets)

                    if annotate_dir is not None:
                        annotated = frame.copy()
                        for d in dets:
                            x1, y1, x2, y2 = map(int, d.bbox)
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{d.vehicle_class} {d.confidence:.2f}"
                            cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(annotated, d.timestamp if dets else
                                    frame_index_to_timestamp(frame_index, fps),
                                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                        cv2.imwrite(str(annotate_dir / f"frame_{frame_index:06d}.jpg"), annotated)
                frame_index += 1
        finally:
            cap.release()

        return all_detections


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DrishtAI Stage 2: YOLOv8 vehicle detection"
    )
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--interval", type=int, default=1,
                        help="Process every Nth frame (default 1)")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="YOLOv8 weights (default yolov8s.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default 0.25)")
    parser.add_argument("--out", default="detections.json",
                        help="Output JSON path (default detections.json)")
    parser.add_argument("--annotate", default=None,
                        help="Optional dir to save frames with boxes drawn")
    args = parser.parse_args()

    try:
        detector = VehicleDetector(args.model, args.conf)
        detections = detector.detect_video(args.video, args.interval, args.annotate)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[detector] ERROR: {e}", file=sys.stderr)
        return 1

    Path(args.out).write_text(json.dumps([asdict(d) for d in detections], indent=2))

    frames_with = len({d.frame_index for d in detections})
    by_class: dict[str, int] = {}
    for d in detections:
        by_class[d.vehicle_class] = by_class.get(d.vehicle_class, 0) + 1

    print(f"[detector] video: {args.video}")
    print(f"[detector] model: {args.model}  conf>={args.conf}")
    print(f"[detector] total detections: {len(detections)} across {frames_with} frames")
    print(f"[detector] by class: {by_class}")
    print(f"[detector] written to {args.out}")
    if args.annotate:
        print(f"[detector] annotated frames in {args.annotate}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
