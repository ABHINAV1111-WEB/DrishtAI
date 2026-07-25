"""
motion_math.py — DrishtAI pipeline, Stage 4 (motion math + track merging)

Consumes tracks.json from Stage 3 and produces, per tracked vehicle per
frame: velocity, direction and acceleration. These three quantities are the
raw material the collision detector (Stage 5) thresholds to produce the
schema's event vocabulary — "distance_dropping", "sudden_velocity_change",
"trajectory_intersecting" are all just rules over this output.

Also merges fragmented tracks (see MERGE below), because our footage has an
articulated truck that YOLO detects as two separate objects.

=============================================================================
VELOCITY UNITS: PIXELS PER SECOND, matching schema.md v2.

Per-frame was ambiguous once frames are sampled at --interval N: "per source
frame" and "per processed frame" differ by a factor of the interval, and
neither errors visibly — the bug would surface much later as "our collision
thresholds need different values on a 30fps clip than a 60fps clip".
Pixels-per-second is invariant to both fps and interval, so thresholds tuned
on one clip transfer to another. Elapsed time always comes from time_seconds
deltas, never assumed from the sampling interval.
=============================================================================

Design decisions:

1. DIRECTION USES SCREEN-TO-MATH AXIS FLIP. Image y grows DOWNWARD, but
   schema.md specifies "standard unit circle convention". So we compute
   atan2(-dy, dx), negating dy. Without the negation, a vehicle moving up
   the screen would report 270 degrees instead of 90, and every trajectory
   comparison downstream would be mirrored. This is the single easiest
   silent bug in the whole stage.

2. DELTA TIME COMES FROM time_seconds, NEVER FROM A CONSTANT. Two reasons:
   changing --interval would otherwise silently scale every velocity, and a
   tracking gap (vehicle missing for 3 frames then reappearing) has a real
   dt of 3 intervals, not 1. Subtracting actual timestamps handles both.

3. SMOOTHING WINDOW DEFAULT 3, DELIBERATELY SMALL. Bounding boxes jitter a
   few pixels frame to frame even for a perfectly steady vehicle, and raw
   frame-to-frame velocity inherits that as noise. A centred 3-point mean
   removes most of it. It is kept small on purpose: heavy smoothing would
   blur the sudden velocity drop at impact, which is precisely the signal
   Stage 5 must detect. Use --smooth 1 to disable.

4. ACCELERATION is part of the per-object record as of schema.md v2, in
   px/s^2, computed from the smoothed velocity series rather than raw
   position deltas. The collision detector reads a sharp NEGATIVE value as
   the signature of impact or emergency braking.

5. `event` IS NOT SET HERE. Stage 5 owns the event vocabulary. Emitting
   "moving_normally" for everything now would be a claim this stage has not
   earned.

MERGE — fragmented tracks from articulated vehicles
---------------------------------------------------
An articulated truck (cab + trailer) is frequently detected as two objects,
each getting its own object_id. Left alone, the collision detector reports
the same physical impact twice and the Claude explanation names three
vehicles where a viewer sees two.

Two tracks are merged when, across the frames they share:
  a) same vehicle_class, and
  b) their boxes touch or overlap (gap smaller than --merge-gap x the
     smaller box's diagonal), and
  c) their velocities and directions agree within tolerance, and
  d) conditions (b) and (c) hold in at least --merge-stability of shared
     frames, and they share at least --merge-min-frames frames.

Condition (d) is what separates a genuine articulated pair from two
different vehicles that happened to drive close together for a moment. Two
independent cars diverge; a cab and its trailer never do.

Merged output: the surviving object_id is the longer-lived one, the box is
the union of both, position is the union's centroid, and `merged_from`
records the absorbed ids so the merge is auditable rather than magic.

Usage:
    python src/motion/motion_math.py tracks.json --out motion.json
    python src/motion/motion_math.py tracks.json --out motion.json --no-merge
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class MotionRecord:
    """One vehicle in one frame, with motion quantities resolved."""
    object_id: str
    frame_index: int
    timestamp: str
    time_seconds: float
    position: list[float]          # [cx, cy]
    bbox: list[float]
    velocity: float                # px/s, schema.md v2
    direction: float               # degrees, 0-360, standard unit circle
    acceleration: float            # px/s^2, additive field
    vehicle_class: str
    merged_from: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def box_gap(a: list[float], b: list[float]) -> float:
    """
    Smallest distance between two axis-aligned boxes. 0 if they overlap.
    Used by the merge rule: cab and trailer boxes touch or overlap.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return math.hypot(dx, dy)


def box_diagonal(b: list[float]) -> float:
    return math.hypot(b[2] - b[0], b[3] - b[1])


def union_box(a: list[float], b: list[float]) -> list[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def centroid(b: list[float]) -> list[float]:
    return [round((b[0] + b[2]) / 2, 1), round((b[1] + b[3]) / 2, 1)]


def angle_difference(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees (0-180)."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


# ---------------------------------------------------------------------------
# Core motion computation
# ---------------------------------------------------------------------------

def compute_motion_for_track(records: list[dict], smooth_window: int = 3) -> list[dict]:
    """
    Given all records for ONE object_id sorted by time, attach velocity,
    direction and acceleration to each.

    Velocity/direction at record i are computed from the step i-1 -> i.
    The first record of a track has no predecessor, so it gets velocity 0
    and direction 0 — a known, documented sentinel rather than a guess.
    """
    out = []
    for i, r in enumerate(records):
        if i == 0:
            v, ang = 0.0, 0.0
        else:
            prev = records[i - 1]
            dt = r["time_seconds"] - prev["time_seconds"]
            if dt <= 0:
                v, ang = 0.0, 0.0
            else:
                dx = r["position"][0] - prev["position"][0]
                dy = r["position"][1] - prev["position"][1]
                v = math.hypot(dx, dy) / dt
                # Design note 1: negate dy to convert screen axes (y down)
                # into standard unit-circle convention (y up).
                ang = math.degrees(math.atan2(-dy, dx)) % 360.0
        rec = dict(r)
        rec["_v_raw"] = v
        rec["direction"] = round(ang, 1)
        out.append(rec)

    # Design note 3: light centred smoothing of velocity only.
    if smooth_window and smooth_window > 1:
        half = smooth_window // 2
        raw = [r["_v_raw"] for r in out]
        for i, r in enumerate(out):
            lo, hi = max(0, i - half), min(len(raw), i + half + 1)
            r["velocity"] = round(sum(raw[lo:hi]) / (hi - lo), 2)
    else:
        for r in out:
            r["velocity"] = round(r["_v_raw"], 2)

    # Acceleration from the smoothed velocity series (design note 4).
    for i, r in enumerate(out):
        if i == 0:
            r["acceleration"] = 0.0
        else:
            dt = r["time_seconds"] - out[i - 1]["time_seconds"]
            r["acceleration"] = (round((r["velocity"] - out[i - 1]["velocity"]) / dt, 2)
                                 if dt > 0 else 0.0)
        r.pop("_v_raw", None)
    return out


# ---------------------------------------------------------------------------
# Track merging (articulated vehicles)
# ---------------------------------------------------------------------------

def find_merge_pairs(
    by_id: dict[str, list[dict]],
    merge_gap: float = 0.35,
    velocity_tol: float = 0.35,
    direction_tol: float = 25.0,
    min_shared_frames: int = 10,
    stability: float = 0.7,
) -> list[tuple[str, str, float]]:
    """
    Identify (id_a, id_b, agreement_ratio) pairs that are one physical
    vehicle. See MERGE in the module docstring for the rule.
    """
    ids = list(by_id.keys())
    frames_of = {oid: {r["frame_index"]: r for r in recs} for oid, recs in by_id.items()}
    pairs: list[tuple[str, str, float]] = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            fa, fb = frames_of[a], frames_of[b]
            shared = sorted(set(fa) & set(fb))
            if len(shared) < min_shared_frames:
                continue
            if by_id[a][0]["vehicle_class"] != by_id[b][0]["vehicle_class"]:
                continue

            agree = 0
            for f in shared:
                ra, rb = fa[f], fb[f]
                gap = box_gap(ra["bbox"], rb["bbox"])
                ref = min(box_diagonal(ra["bbox"]), box_diagonal(rb["bbox"]))
                if ref <= 0 or gap > merge_gap * ref:
                    continue
                va, vb = ra.get("velocity", 0.0), rb.get("velocity", 0.0)
                fastest = max(va, vb)
                # Both near-stationary: velocity agreement is uninformative,
                # so accept on proximity alone rather than dividing by ~0.
                if fastest > 1.0:
                    if abs(va - vb) / fastest > velocity_tol:
                        continue
                    if angle_difference(ra.get("direction", 0.0),
                                        rb.get("direction", 0.0)) > direction_tol:
                        continue
                agree += 1

            ratio = agree / len(shared)
            if ratio >= stability:
                pairs.append((a, b, round(ratio, 3)))
    return pairs


def apply_merges(
    by_id: dict[str, list[dict]],
    pairs: list[tuple[str, str, float]],
) -> tuple[dict[str, list[dict]], dict[str, list[str]]]:
    """
    Union-find over merge pairs, then fuse each group into one track.
    Survivor id is the longest-lived member (most frames), so the merged
    vehicle keeps the id a human already saw in the annotated frames.
    """
    parent: dict[str, str] = {oid: oid for oid in by_id}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for a, b, _ in pairs:
        union(a, b)

    groups: dict[str, list[str]] = defaultdict(list)
    for oid in by_id:
        groups[find(oid)].append(oid)

    merged: dict[str, list[dict]] = {}
    provenance: dict[str, list[str]] = {}

    for members in groups.values():
        if len(members) == 1:
            oid = members[0]
            merged[oid] = by_id[oid]
            continue

        survivor = max(members, key=lambda o: len(by_id[o]))
        absorbed = sorted(m for m in members if m != survivor)

        per_frame: dict[int, dict] = {}
        for m in members:
            for r in by_id[m]:
                f = r["frame_index"]
                if f not in per_frame:
                    per_frame[f] = dict(r)
                else:
                    cur = per_frame[f]
                    ub = union_box(cur["bbox"], r["bbox"])
                    cur["bbox"] = [round(v, 1) for v in ub]
                    cur["position"] = centroid(ub)
                    cur["confidence"] = max(cur.get("confidence", 0.0),
                                            r.get("confidence", 0.0))

        fused = []
        for f in sorted(per_frame):
            rec = per_frame[f]
            rec["object_id"] = survivor
            fused.append(rec)

        merged[survivor] = fused
        provenance[survivor] = absorbed

    return merged, provenance


# ---------------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------------

def run(
    tracks: list[dict],
    smooth_window: int = 3,
    do_merge: bool = True,
    **merge_kwargs,
) -> tuple[list[MotionRecord], dict[str, list[str]], list[tuple[str, str, float]]]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for r in tracks:
        by_id[r["object_id"]].append(r)
    for oid in by_id:
        by_id[oid].sort(key=lambda r: r["frame_index"])

    # Pass 1: motion on raw tracks — the merge rule needs velocity/direction
    # to decide whether two tracks are co-moving.
    by_id = {oid: compute_motion_for_track(recs, smooth_window)
             for oid, recs in by_id.items()}

    pairs: list[tuple[str, str, float]] = []
    provenance: dict[str, list[str]] = {}
    if do_merge:
        pairs = find_merge_pairs(by_id, **merge_kwargs)
        by_id, provenance = apply_merges(by_id, pairs)
        # Pass 2: recompute on fused geometry — the union box has a
        # different centroid, so the pass-1 numbers are stale for merged
        # tracks. Recomputing everything keeps one code path.
        by_id = {oid: compute_motion_for_track(recs, smooth_window)
                 for oid, recs in by_id.items()}

    out: list[MotionRecord] = []
    for oid, recs in by_id.items():
        for r in recs:
            out.append(MotionRecord(
                object_id=oid,
                frame_index=r["frame_index"],
                timestamp=r["timestamp"],
                time_seconds=r["time_seconds"],
                position=r["position"],
                bbox=r["bbox"],
                velocity=r["velocity"],
                direction=r["direction"],
                acceleration=r["acceleration"],
                vehicle_class=r["vehicle_class"],
                merged_from=provenance.get(oid, []),
            ))
    out.sort(key=lambda r: (r.frame_index, r.object_id))
    return out, provenance, pairs


def main() -> int:
    p = argparse.ArgumentParser(
        description="DrishtAI Stage 4: motion math + articulated-track merging")
    p.add_argument("tracks", help="tracks.json from Stage 3")
    p.add_argument("--out", default="motion.json", help="output JSON path")
    p.add_argument("--smooth", type=int, default=3,
                   help="velocity smoothing window, 1 disables (default 3)")
    p.add_argument("--no-merge", action="store_true",
                   help="skip articulated-vehicle track merging")
    p.add_argument("--merge-gap", type=float, default=0.35,
                   help="max box gap as fraction of smaller box diagonal")
    p.add_argument("--merge-stability", type=float, default=0.7,
                   help="fraction of shared frames that must agree")
    p.add_argument("--merge-min-frames", type=int, default=10,
                   help="minimum shared frames to consider a merge")
    args = p.parse_args()

    src = Path(args.tracks)
    if not src.exists():
        print(f"[motion] ERROR: {src} not found", file=sys.stderr)
        return 1
    tracks = json.loads(src.read_text())
    if not tracks:
        print("[motion] ERROR: tracks file is empty", file=sys.stderr)
        return 1

    records, provenance, pairs = run(
        tracks,
        smooth_window=args.smooth,
        do_merge=not args.no_merge,
        merge_gap=args.merge_gap,
        stability=args.merge_stability,
        min_shared_frames=args.merge_min_frames,
    )

    Path(args.out).write_text(json.dumps([asdict(r) for r in records], indent=2))

    ids_before = len({r["object_id"] for r in tracks})
    ids_after = len({r.object_id for r in records})
    print(f"[motion] input records: {len(tracks)}  ids: {ids_before}")
    if not args.no_merge:
        print(f"[motion] merge pairs found: {len(pairs)}")
        for a, b, ratio in pairs[:10]:
            print(f"    {a} + {b}   agreement {ratio:.0%}")
        for survivor, absorbed in provenance.items():
            print(f"    -> merged into {survivor}: absorbed {absorbed}")
    print(f"[motion] output records: {len(records)}  ids: {ids_after}")

    moving = [r.velocity for r in records if r.velocity > 0]
    if moving:
        moving.sort()
        print(f"[motion] velocity px/s  min {moving[0]:.1f}  "
              f"median {moving[len(moving)//2]:.1f}  max {moving[-1]:.1f}")
    print(f"[motion] written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
