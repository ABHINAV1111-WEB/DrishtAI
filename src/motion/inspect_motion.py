"""
inspect_motion.py — DrishtAI debug tool (not a pipeline stage)

Reads motion.json and prints what is actually happening in a chosen frame
window: per-vehicle velocity/acceleration series, and the distance between
any two vehicles over time.

Why this exists: Stage 5 works by thresholding velocity, acceleration and
inter-vehicle distance. Before writing thresholds you need to SEE those
numbers around the impact, otherwise you are tuning blind. This is also the
tool you will keep using to pick threshold values.

Usage:
    # what merged, and overall sanity
    python src/motion/inspect_motion.py motion.json --summary

    # every vehicle present between frames 40 and 80
    python src/motion/inspect_motion.py motion.json --window 40 80

    # one vehicle's full velocity/acceleration series
    python src/motion/inspect_motion.py motion.json --vehicle vehicle_3

    # distance between two vehicles over the crash window
    python src/motion/inspect_motion.py motion.json --pair vehicle_3 vehicle_7 --window 40 80
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"[inspect] ERROR: {p} not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(p.read_text())


def summary(recs: list[dict]) -> None:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_id[r["object_id"]].append(r)

    merged = {oid: rs[0]["merged_from"] for oid, rs in by_id.items()
              if rs[0].get("merged_from")}
    print(f"[inspect] records: {len(recs)}   vehicles: {len(by_id)}")
    if merged:
        print("[inspect] merged vehicles:")
        for oid, absorbed in merged.items():
            print(f"    {oid}  absorbed {absorbed}")
    else:
        print("[inspect] no merged vehicles in this file")

    vels = sorted(r["velocity"] for r in recs if r["velocity"] > 0)
    if vels:
        print(f"[inspect] velocity px/s   min {vels[0]:.1f}   "
              f"median {vels[len(vels)//2]:.1f}   "
              f"p95 {vels[int(len(vels)*0.95)]:.1f}   max {vels[-1]:.1f}")
    accs = sorted(r["acceleration"] for r in recs)
    if accs:
        print(f"[inspect] accel px/s^2   min {accs[0]:.1f}   "
              f"median {accs[len(accs)//2]:.1f}   max {accs[-1]:.1f}")
        print("[inspect] most negative accelerations (impact candidates):")
        for r in sorted(recs, key=lambda r: r["acceleration"])[:8]:
            print(f"    frame {r['frame_index']:>5}  {r['object_id']:<12} "
                  f"{r['timestamp']}  v={r['velocity']:>8.1f}  "
                  f"a={r['acceleration']:>10.1f}")


def window(recs: list[dict], lo: int, hi: int) -> None:
    sel = [r for r in recs if lo <= r["frame_index"] <= hi]
    if not sel:
        print(f"[inspect] no records between frames {lo} and {hi}")
        return
    print(f"[inspect] frames {lo}-{hi}: {len(sel)} records")
    print(f"{'frame':>6} {'object_id':<12} {'timestamp':<13} "
          f"{'vel px/s':>10} {'accel':>11} {'dir':>7}")
    for r in sorted(sel, key=lambda r: (r["frame_index"], r["object_id"])):
        print(f"{r['frame_index']:>6} {r['object_id']:<12} {r['timestamp']:<13} "
              f"{r['velocity']:>10.1f} {r['acceleration']:>11.1f} "
              f"{r['direction']:>7.1f}")


def vehicle(recs: list[dict], oid: str) -> None:
    sel = sorted([r for r in recs if r["object_id"] == oid],
                 key=lambda r: r["frame_index"])
    if not sel:
        ids = sorted({r["object_id"] for r in recs})
        print(f"[inspect] {oid} not found. Available: {ids[:20]}")
        return
    print(f"[inspect] {oid}  ({sel[0]['vehicle_class']})  "
          f"frames {sel[0]['frame_index']}-{sel[-1]['frame_index']}  "
          f"{len(sel)} records")
    if sel[0].get("merged_from"):
        print(f"[inspect] merged from: {sel[0]['merged_from']}")
    print(f"{'frame':>6} {'timestamp':<13} {'vel px/s':>10} {'accel':>11} {'dir':>7}")
    for r in sel:
        print(f"{r['frame_index']:>6} {r['timestamp']:<13} "
              f"{r['velocity']:>10.1f} {r['acceleration']:>11.1f} "
              f"{r['direction']:>7.1f}")


def pair(recs: list[dict], a: str, b: str, lo: int | None, hi: int | None) -> None:
    """
    Distance between two vehicles frame by frame, plus its rate of change.

    A steadily shrinking distance is the raw signal behind the schema's
    "distance_dropping" event; the frame where it stops shrinking is
    usually the impact.
    """
    fa = {r["frame_index"]: r for r in recs if r["object_id"] == a}
    fb = {r["frame_index"]: r for r in recs if r["object_id"] == b}
    shared = sorted(set(fa) & set(fb))
    if lo is not None:
        shared = [f for f in shared if lo <= f <= hi]
    if not shared:
        print(f"[inspect] {a} and {b} share no frames in that range")
        return

    print(f"[inspect] {a} vs {b}: {len(shared)} shared frames")
    print(f"{'frame':>6} {'timestamp':<13} {'distance':>10} {'d(dist)/dt':>12} "
          f"{'v_a':>9} {'v_b':>9}")
    prev_d = prev_t = None
    for f in shared:
        ra, rb = fa[f], fb[f]
        d = math.dist(ra["position"], rb["position"])
        if prev_d is None:
            rate = 0.0
        else:
            dt = ra["time_seconds"] - prev_t
            rate = (d - prev_d) / dt if dt > 0 else 0.0
        print(f"{f:>6} {ra['timestamp']:<13} {d:>10.1f} {rate:>12.1f} "
              f"{ra['velocity']:>9.1f} {rb['velocity']:>9.1f}")
        prev_d, prev_t = d, ra["time_seconds"]


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect DrishtAI motion.json")
    p.add_argument("motion", help="motion.json from Stage 4")
    p.add_argument("--summary", action="store_true", help="overall stats + merges")
    p.add_argument("--window", nargs=2, type=int, metavar=("LO", "HI"),
                   help="print all records in a frame range")
    p.add_argument("--vehicle", help="print one vehicle's full series")
    p.add_argument("--pair", nargs=2, metavar=("A", "B"),
                   help="distance between two vehicles over time")
    args = p.parse_args()

    recs = load(args.motion)
    did = False
    if args.summary:
        summary(recs); did = True
    if args.window:
        window(recs, args.window[0], args.window[1]); did = True
    if args.vehicle:
        vehicle(recs, args.vehicle); did = True
    if args.pair:
        lo, hi = (args.window if args.window else (None, None))
        pair(recs, args.pair[0], args.pair[1], lo, hi); did = True
    if not did:
        summary(recs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
