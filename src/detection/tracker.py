"""
tracker.py — DrishtAI pipeline, Stage 3 (persistent vehicle tracking)

Takes per-frame, memory-less detections and assigns each vehicle a
persistent `object_id` that survives across frames. This is the field
schema.md reserves; once populated, motion math (Stage 4) can finally
compute per-vehicle velocity and direction over time.

NOTE ON PIPELINE POSITION: model.track() performs detection AND id
assignment in one pass, so this module supersedes detector.py in the live
pipeline (extractor -> tracker -> motion math). detector.py is retained as
an isolated detection-quality debug tool, not a pipeline stage — do not run
both in sequence or YOLO executes twice for no benefit.

Uses Ultralytics' built-in ByteTrack. Reasons over a hand-rolled centroid
tracker:
  - Already inside the `ultralytics` package we use for detection, so the
    integration surface is one method call. No new dependency (except lap,
    see below), no new failure mode.
  - Keeps "lost" tracks alive for `track_buffer` frames and re-attaches the
    SAME id when a vehicle reappears — this bridges short detection flickers.
  - Centroid fallback stays available per the roadmap if this misbehaves.

REQUIRES: `pip install lap` (ByteTrack's assignment solver). Ultralytics
will auto-install it mid-run otherwise and then ask you to re-run, which is
confusing the first time it happens.

Design decisions:

1. `persist=True` IS MANDATORY. It tells the tracker this frame continues
   the previous frame's sequence. Without it every frame is treated as a
   fresh video and ids restart from 1 each time — silently producing JSON
   that looks entirely plausible while being meaningless.

2. CONFIDENCE DEFAULT IS 0.10 HERE, NOT 0.25 LIKE THE DETECTOR. This is
   deliberate and non-obvious, so read before "fixing" it.

   ByteTrack's default config runs TWO association passes:
       track_high_thresh: 0.25   first-stage matching
       track_low_thresh:  0.10   second-stage recovery of weak detections
       new_track_thresh:  0.25   minimum score to START a new track
   The second pass is the whole point of the algorithm (the "BYTE" in the
   name): low-scoring boxes cannot start new tracks, but they CAN keep an
   existing track alive through motion blur, deformation and occlusion.

   The `conf` we pass filters detections BEFORE ByteTrack sees them. Setting
   conf=0.25 therefore discards the entire 0.10-0.25 band and disables the
   recovery pass — which cost us a tracked vehicle for ~6 frames straight
   through the impact on our footage, even though plain detection at 0.25
   found it fine. conf=0.10 feeds ByteTrack the weak boxes it is designed
   to consume. Because new_track_thresh stays at 0.25, weak boxes still
   cannot spawn junk ids.

3. FRAMES MUST BE FED IN ORDER, no skipping mid-sequence. The tracker is
   stateful; it matches this frame's boxes against the previous frame's
   tracks.

4. TRACKING INTERVAL TRADEOFF. At --interval 2 on 60fps footage the tracker
   sees frames 33ms apart; vehicles move further between updates than at
   interval 1, raising id-switch risk (association is IoU-based against
   match_thresh 0.8). If ids switch during a crash, drop to --interval 1 —
   correctness of ids beats runtime, because every downstream stage depends
   on them.

5. ids ARE FORMATTED AS "vehicle_N" to match schema.md. ByteTrack returns
   bare integers; we prefix at the boundary so the rest of the pipeline only
   ever sees schema-shaped strings.

6. velocity / direction / event ARE NOT SET HERE. They are absent from this
   stage's output and filled by motion math (Stage 4) and the collision
   detector (Stage 5). Placeholder zeros would later be indistinguishable
   from a genuinely stationary vehicle — a silent data corruption.

Usage:
    python src/detection/tracker.py data/clip1.mp4 --interval 2 --out tracks.json
    python src/detection/tracker.py data/clip1.mp4 --interval 2 --out tracks.json --annotate tracked/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from .frame_extractor import frame_index_to_timestamp
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from frame_extractor import frame_index_to_timestamp


VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class TrackedDetection:
    """
    One tracked vehicle in one frame.

    Same shape as Stage 2's Detection, plus the object_id that makes it a
    schema-compliant record. velocity/direction/event are deliberately
    absent — Stages 4 and 5 add those.
    """
    object_id: str            # "vehicle_4" — persistent across frames
    frame_index: int
    timestamp: str            # HH:MM:SS:FF per schema.md
    time_seconds: float
    bbox: list[float]
    position: list[float]     # [cx, cy] centroid
    confidence: float
    vehicle_class: str


class VehicleTracker:
    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        conf_threshold: float = 0.10,   # see design note 2 — NOT a typo
        tracker_cfg: str = "bytetrack.yaml",
    ):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.tracker_cfg = tracker_cfg

    def track_video(
        self,
        video_path: str | Path,
        frame_interval: int = 1,
        annotate_dir: str | Path | None = None,
    ) -> list[TrackedDetection]:
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

        records: list[TrackedDetection] = []
        frame_index = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index % frame_interval == 0:
                    results = self.model.track(
                        frame,
                        conf=self.conf_threshold,
                        classes=list(VEHICLE_CLASSES.keys()),
                        tracker=self.tracker_cfg,
                        persist=True,     # see design note 1 — do not remove
                        verbose=False,
                    )
                    ts = frame_index_to_timestamp(frame_index, fps)
                    t = frame_index / fps
                    frame_records: list[TrackedDetection] = []

                    boxes = results[0].boxes
                    # boxes.id is None when the tracker has confirmed no
                    # tracks in this frame. Skip rather than invent an id.
                    if boxes.id is not None:
                        for i in range(len(boxes)):
                            raw_id = int(boxes.id[i].item())
                            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                            frame_records.append(TrackedDetection(
                                object_id=f"vehicle_{raw_id}",
                                frame_index=frame_index,
                                timestamp=ts,
                                time_seconds=t,
                                bbox=[round(x1, 1), round(y1, 1),
                                      round(x2, 1), round(y2, 1)],
                                position=[round((x1 + x2) / 2, 1),
                                          round((y1 + y2) / 2, 1)],
                                confidence=round(float(boxes.conf[i].item()), 3),
                                vehicle_class=VEHICLE_CLASSES[int(boxes.cls[i].item())],
                            ))

                    records.extend(frame_records)

                    if annotate_dir is not None:
                        self._draw(frame, frame_records, ts,
                                   annotate_dir / f"frame_{frame_index:06d}.jpg")

                frame_index += 1
        finally:
            cap.release()

        return records

    @staticmethod
    def _colour_for(object_id: str) -> tuple[int, int, int]:
        """
        Stable BGR colour per id, so the same vehicle keeps the same box
        colour across frames. Makes id switches obvious to the eye: a
        vehicle whose box changes colour mid-clip has been re-identified.

        Uses a fixed arithmetic hash rather than Python's hash(), which is
        randomised per process — otherwise colours would change between
        runs and cross-run visual comparison would be impossible.
        """
        n = int(object_id.rsplit("_", 1)[-1]) if "_" in object_id else 0
        h = (n * 2654435761) & 0xFFFFFFFF   # Knuth multiplicative hash
        return (50 + (h & 0xFF) % 205,
                50 + ((h >> 8) & 0xFF) % 205,
                50 + ((h >> 16) & 0xFF) % 205)

    @classmethod
    def _draw(cls, frame, records: list[TrackedDetection], ts: str, out_path: Path) -> None:
        canvas = frame.copy()
        for r in records:
            x1, y1, x2, y2 = map(int, r.bbox)
            colour = cls._colour_for(r.object_id)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            label = f"{r.object_id} {r.vehicle_class}"
            cv2.putText(canvas, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        cv2.putText(canvas, ts, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 200, 255), 2)
        cv2.imwrite(str(out_path), canvas)


def summarise(records: list[TrackedDetection], frame_interval: int) -> None:
    """
    Print a track-health report. This tells you whether tracking actually
    worked — the raw record count does not.

    Short-lived ids are the warning sign: an id that exists for only a
    couple of frames is usually a false detection or, worse, the result of
    an id switch where a real vehicle was re-identified as new.
    """
    spans: dict[str, list[int]] = defaultdict(list)
    for r in records:
        spans[r.object_id].append(r.frame_index)

    print(f"[tracker] unique object_ids: {len(spans)}")
    print(f"[tracker] total tracked records: {len(records)}")

    ordered = sorted(spans.items(), key=lambda kv: len(kv[1]), reverse=True)
    print("[tracker] longest-lived ids:")
    for oid, frames in ordered[:8]:
        first, last = min(frames), max(frames)
        expected = (last - first) // frame_interval + 1
        gaps = expected - len(frames)
        print(f"    {oid:<12} seen in {len(frames):>4} frames  "
              f"({first}-{last})  internal gaps: {gaps}")

    shortlived = [oid for oid, f in spans.items() if len(f) <= 2]
    if shortlived:
        print(f"[tracker] WARNING: {len(shortlived)} id(s) lived <=2 frames "
              f"(possible false positives or id switches): {shortlived[:10]}")
    else:
        print("[tracker] no suspiciously short-lived ids")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DrishtAI Stage 3: persistent vehicle tracking (ByteTrack)"
    )
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--interval", type=int, default=1,
                        help="Process every Nth frame (default 1)")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="YOLOv8 weights (default yolov8s.pt)")
    parser.add_argument("--conf", type=float, default=0.10,
                        help="Confidence threshold (default 0.10 — low on "
                             "purpose so ByteTrack's low-score recovery pass "
                             "can run; see module docstring)")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                        help="Tracker config: bytetrack.yaml or botsort.yaml")
    parser.add_argument("--out", default="tracks.json",
                        help="Output JSON path (default tracks.json)")
    parser.add_argument("--annotate", default=None,
                        help="Optional dir to save frames with ids drawn")
    args = parser.parse_args()

    try:
        tracker = VehicleTracker(args.model, args.conf, args.tracker)
        records = tracker.track_video(args.video, args.interval, args.annotate)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[tracker] ERROR: {e}", file=sys.stderr)
        return 1

    Path(args.out).write_text(json.dumps([asdict(r) for r in records], indent=2))

    print(f"[tracker] video: {args.video}")
    print(f"[tracker] model: {args.model}  conf>={args.conf}  tracker={args.tracker}")
    summarise(records, args.interval)
    print(f"[tracker] written to {args.out}")
    if args.annotate:
        print(f"[tracker] annotated frames in {args.annotate}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
