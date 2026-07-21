"""
frame_extractor.py — DrishtAI pipeline, Stage 1 (frame extraction + timestamps)

Extracts frames from a video at a fixed frame interval and computes a
per-frame timestamp in HH:MM:SS:FF format (FF = frame number within that
second), matching the locked schema in schema.md.

Design decisions (deliberate — don't "fix" these without reading why):

1. Frames are read SEQUENTIALLY with cap.read(), skipping frames we don't
   want, instead of seeking with CAP_PROP_POS_FRAMES. Seeking is unreliable
   on many codecs (H.264 inter-frame compression means OpenCV can land on
   the wrong frame after a seek). Sequential read is always frame-accurate.

2. Timestamps are computed from the FRAME INDEX + nominal FPS, not from
   CAP_PROP_POS_MSEC. POS_MSEC is inconsistent across backends/codecs and
   sometimes returns 0 or garbage. frame_index / fps is deterministic and
   reproducible — the same frame always gets the same timestamp, which
   matters because every downstream stage (tracking, motion math, timeline)
   keys off these timestamps.

3. The frame index recorded is the ORIGINAL video frame index, not the
   extraction count. If you extract every 5th frame, the timestamps still
   reflect true video time (frame 0, 5, 10, ...), so velocity math later
   uses correct time deltas.

Usage:
    python frame_extractor.py path/to/video.mp4 --interval 5 --out frames/
    python frame_extractor.py path/to/video.mp4 --interval 5          # metadata only, no images written
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2


# --------------------------------------------------------------------------
# Timestamp math
# --------------------------------------------------------------------------

def frame_index_to_timestamp(frame_index: int, fps: float) -> str:
    """
    Convert an absolute frame index into HH:MM:SS:FF.

    FF is the frame number *within* the current second, zero-based.
    For a 30 fps video, FF cycles 00..29 and rolls over as SS increments.

    We use round(fps) as the frames-per-second base. For integer-fps footage
    (24/25/30/60 — which is what our standardized clips will be) this is
    exact. For NTSC-style 29.97 fps it drifts ~1 frame every 16.7 minutes;
    acceptable for short hackathon clips, and avoided entirely if Aditya
    standardizes clips to 25 or 30 fps during editing (already his Day 2-3
    task).
    """
    if fps <= 0:
        raise ValueError(f"Invalid fps: {fps}")
    if frame_index < 0:
        raise ValueError(f"Invalid frame index: {frame_index}")

    fps_base = round(fps)
    total_seconds, ff = divmod(frame_index, fps_base)
    hh, rem = divmod(total_seconds, 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

@dataclass
class FrameRecord:
    """Metadata for one extracted frame. This feeds the detection stage."""
    frame_index: int      # absolute index in the source video (0-based)
    timestamp: str        # HH:MM:SS:FF per schema.md
    time_seconds: float   # exact float time, kept for motion math precision
    image_path: str | None  # where the frame was saved, if saving enabled


def extract_frames(
    video_path: str | Path,
    frame_interval: int = 1,
    output_dir: str | Path | None = None,
    jpeg_quality: int = 90,
) -> list[FrameRecord]:
    """
    Extract every `frame_interval`-th frame from `video_path`.

    Args:
        video_path: source video file.
        frame_interval: keep 1 frame out of every N. 1 = every frame.
        output_dir: if given, frames are written there as JPEGs named
            frame_{index:06d}.jpg. If None, no images are written (useful
            when a later stage consumes frames in-memory instead).
        jpeg_quality: JPEG quality for saved frames (0-100).

    Returns:
        List of FrameRecord, in video order.

    Raises:
        FileNotFoundError: video file missing.
        RuntimeError: video can't be opened or reports invalid FPS.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if frame_interval < 1:
        raise ValueError(f"frame_interval must be >= 1, got {frame_interval}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # Some containers/codecs report 0 FPS. Fail loudly rather than
        # silently producing wrong timestamps — every downstream stage
        # depends on these being right.
        cap.release()
        raise RuntimeError(
            f"Video reports invalid FPS ({fps}). Re-encode the clip "
            f"(e.g. ffmpeg -i in.mp4 -r 30 out.mp4) before running the pipeline."
        )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    records: list[FrameRecord] = []
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of video (or decode error — same handling)

            if frame_index % frame_interval == 0:
                ts = frame_index_to_timestamp(frame_index, fps)
                image_path = None
                if output_dir is not None:
                    image_path = str(output_dir / f"frame_{frame_index:06d}.jpg")
                    cv2.imwrite(
                        image_path, frame,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                    )
                records.append(FrameRecord(
                    frame_index=frame_index,
                    timestamp=ts,
                    time_seconds=frame_index / fps,
                    image_path=image_path,
                ))

            frame_index += 1
    finally:
        cap.release()

    return records


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="DrishtAI Stage 1: frame extraction with HH:MM:SS:FF timestamps"
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument(
        "--interval", type=int, default=1,
        help="Extract every Nth frame (default: 1 = every frame)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Directory to save frames as JPEGs. Omit to skip saving images.",
    )
    parser.add_argument(
        "--metadata", default=None,
        help="Optional path to write frame metadata as JSON.",
    )
    args = parser.parse_args()

    try:
        records = extract_frames(args.video, args.interval, args.out)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"[frame_extractor] ERROR: {e}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"[frame_extractor] video: {args.video}")
    print(f"[frame_extractor] fps reported: {fps:.3f} (timestamp base: {round(fps)})")
    print(f"[frame_extractor] frames extracted: {len(records)} (interval={args.interval})")
    if records:
        print(f"[frame_extractor] first: idx={records[0].frame_index} ts={records[0].timestamp}")
        print(f"[frame_extractor] last:  idx={records[-1].frame_index} ts={records[-1].timestamp}")

    if args.metadata:
        Path(args.metadata).write_text(
            json.dumps([asdict(r) for r in records], indent=2)
        )
        print(f"[frame_extractor] metadata written to {args.metadata}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
