#!/usr/bin/env python
"""
check_vehicle_labels.py — preview the voted vehicle types before wiring
anything into the UI.

Place at: repo root (E:\\DrishtAI\\check_vehicle_labels.py)

Reads a tracks JSON your pipeline has already produced, so it costs no
YOLO inference and no waiting. Point it at whatever your tracker stage
wrote.

Usage:
    python check_vehicle_labels.py tracks.json
    python check_vehicle_labels.py tracks.json --style suffix
    python check_vehicle_labels.py tracks.json --min-detections 5

Then open clip3.mp4 and check the answers against your own eyes,
especially vehicle_2 and vehicle_4. If a label is wrong, raise
--min-detections and look at the vote breakdown printed below it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "reasoning"))

from vehicle_labels import build_class_map, humanise_ids, label_for  # noqa: E402


def load_records(path: Path) -> list[dict]:
    """
    Accept the shapes a pipeline stage might reasonably have written:
    a bare list, or a dict with the records under a common key.
    """
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("tracks", "records", "detections", "objects"):
            if isinstance(data.get(key), list):
                return data[key]

    raise SystemExit(
        f"Could not find a list of detection records in {path}.\n"
        f"Top-level type was {type(data).__name__}"
        + (f" with keys {list(data)[:8]}" if isinstance(data, dict) else "")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", type=Path, help="path to tracks JSON")
    ap.add_argument("--style", default="paren",
                    choices=["paren", "suffix", "plain"])
    ap.add_argument("--min-detections", type=int, default=3)
    args = ap.parse_args()

    if not args.tracks.exists():
        raise SystemExit(f"No such file: {args.tracks}")

    records = load_records(args.tracks)
    print(f"Loaded {len(records)} detection records from {args.tracks}\n")

    # Raw vote breakdown, so a wrong label is diagnosable rather than magic.
    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        oid, klass = r.get("object_id"), r.get("vehicle_class")
        if not oid or not klass:
            continue
        conf = r.get("confidence")
        votes[oid][klass] += float(conf) if isinstance(conf, (int, float)) else 1.0
        counts[oid] += 1

    if not votes:
        raise SystemExit(
            "No records carried both object_id and vehicle_class.\n"
            "Check that this file is the tracker output, not motion.json."
        )

    class_map = build_class_map(records, min_detections=args.min_detections)

    def sort_key(oid: str) -> tuple[int, str]:
        digits = "".join(c for c in oid if c.isdigit())
        return (int(digits) if digits else 0, oid)

    print(f"{'id':<14}{'label':<26}{'frames':>7}   vote breakdown")
    print("-" * 78)
    for oid in sorted(votes, key=sort_key):
        label = label_for(oid, class_map, style=args.style)
        flag = "" if oid in class_map else "   <- too few frames, no type"
        breakdown = ", ".join(
            f"{k}={v:.2f}"
            for k, v in sorted(votes[oid].items(), key=lambda kv: -kv[1])
        )
        print(f"{oid:<14}{label:<26}{counts[oid]:>7}   {breakdown}{flag}")

    print("\n--- sample rendered answer ---")
    sample = ("Sudden velocity change occurred 1.87 seconds in, involving "
              "vehicle_2 and vehicle_4 [00:00:01:26].")
    print("before:", sample)
    print("after :", humanise_ids(sample, class_map, style=args.style))


if __name__ == "__main__":
    main()
