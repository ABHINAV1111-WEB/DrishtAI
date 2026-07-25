"""
collision_detector.py — DrishtAI pipeline, Stage 5 (collision detection)

Reads motion.json (Stage 4) and assigns the schema's `event` vocabulary:
    distance_dropping, trajectory_intersecting, sudden_velocity_change,
    collision
It also RANKS candidate vehicle pairs, so you do not have to identify the
crash vehicles by eye — the detector reports which pair it believes
collided, and you verify that against the annotated frames.

=============================================================================
WHY EVERY RULE HERE IS A SUSTAINED WINDOW, NOT A SINGLE FRAME
=============================================================================
Measured on our real footage, ordinary driving produces single-frame
accelerations of +-14000 px/s^2 — the same magnitude as the actual impact.
A rule like "acceleration < -10000 => collision" fires on vehicles doing
nothing wrong, and one vehicle appeared as a top impact candidate four
separate times in a 4.5 second clip.

The reason is measurement jitter, not motion: bounding boxes wobble, and at
conf=0.10 (needed for track continuity) they wobble more. A centroid
inherits that wobble, and differentiating a noisy signal twice (position ->
velocity -> acceleration) amplifies it.

Noise oscillates: it spikes negative, then positive, then negative again.
A real impact does not. So every rule below compares a SUSTAINED WINDOW
before an instant against a SUSTAINED WINDOW after it. Averaging over a
window suppresses zero-mean jitter while preserving a genuine step change.
This is also the honest answer when a judge asks "how do you avoid false
alarms?".
=============================================================================

Design decisions:

1. PROXIMITY IS MEASURED AS BOX GAP, NOT CENTROID DISTANCE. Two centroids
   200px apart could be touching bumpers (large trucks) or far apart (small
   cars at distance). The gap between bounding boxes is 0 exactly when the
   boxes touch, whatever the vehicle size or camera perspective. Contact is
   therefore a scale-free test rather than a pixel threshold we would have
   to retune per clip.

2. VELOCITY CHANGE IS A RATIO OF WINDOW MEANS, NOT AN ACCELERATION READING.
   mean(v) over W frames after vs mean(v) over W frames before. A 60% drop
   sustained across a window is an impact; a single -14000 spike that
   recovers next frame is noise.

3. CANDIDATE PAIRS ARE RANKED, NOT THRESHOLDED INTO A BOOLEAN. Real footage
   rarely gives one unambiguous answer. The detector scores every pair that
   ever comes close, prints the ranking, and marks the top pair's events.
   A ranking is inspectable; a silent boolean is not.

4. `event` IS ASSIGNED ONLY WHERE EARNED. Records with no detected event
   have no `event` key, per schema.md — "not yet evaluated" must stay
   distinguishable from "evaluated as normal".

5. NO EARLIEST-WARNING FLAG HERE. Stage 6 (timeline builder) walks the
   chain backwards to set is_earliest_warning. This stage only reports what
   happened, not which moment was the first risk signal.

Usage:
    python src/reasoning/collision_detector.py motion.json --out events.json
    python src/reasoning/collision_detector.py motion.json --out events.json --top 5
    python src/reasoning/collision_detector.py motion.json --out events.json --window 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def box_gap(a: list[float], b: list[float]) -> float:
    """Smallest distance between two axis-aligned boxes; 0 if they overlap."""
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return math.hypot(dx, dy)


def box_scale(b: list[float]) -> float:
    """Characteristic size of a box, used to normalise distances."""
    return math.hypot(b[2] - b[0], b[3] - b[1])


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# Event records
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """One detected event. Feeds the Stage 6 timeline builder."""
    timestamp: str
    frame_index: int
    time_seconds: float
    event: str
    objects_involved: list[str]
    detail: str


@dataclass
class PairScore:
    a: str
    b: str
    shared_frames: int
    min_gap: float                 # closest approach, px (0 = boxes touched)
    min_gap_frame: int
    approach_rate: float           # px/s, negative = closing
    approach_frames: int           # consecutive frames of sustained closing
    velocity_drop: float           # 0-1, fraction of speed lost at contact
    contact: bool
    score: float


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def analyse_pair(
    ra: list[dict],
    rb: list[dict],
    window: int,
    contact_ratio: float,
    min_shared: int,
) -> PairScore | None:
    """
    Score one vehicle pair. Returns None if they never interact.

    ra, rb are the motion records for two object_ids, each sorted by frame.
    """
    fa = {r["frame_index"]: r for r in ra}
    fb = {r["frame_index"]: r for r in rb}
    shared = sorted(set(fa) & set(fb))
    if len(shared) < min_shared:
        return None

    gaps: list[float] = []
    for f in shared:
        gap = box_gap(fa[f]["bbox"], fb[f]["bbox"])
        scale = min(box_scale(fa[f]["bbox"]), box_scale(fb[f]["bbox"]))
        gaps.append(gap / scale if scale > 0 else float("inf"))

    i_min = min(range(len(gaps)), key=lambda i: gaps[i])
    min_gap_norm = gaps[i_min]
    min_gap_frame = shared[i_min]

    # Design note 1: contact means boxes essentially touching, scaled by
    # vehicle size rather than an absolute pixel threshold.
    contact = min_gap_norm <= contact_ratio

    # Sustained approach: count consecutive frames before closest approach
    # where the normalised gap was shrinking.
    approach_frames = 0
    for i in range(i_min, 0, -1):
        if gaps[i] < gaps[i - 1]:
            approach_frames += 1
        else:
            break

    # Approach rate over the sustained run, in px/s of real gap.
    if approach_frames >= 1:
        i0 = i_min - approach_frames
        f0, f1 = shared[i0], shared[i_min]
        g0 = box_gap(fa[f0]["bbox"], fb[f0]["bbox"])
        g1 = box_gap(fa[f1]["bbox"], fb[f1]["bbox"])
        dt = fa[f1]["time_seconds"] - fa[f0]["time_seconds"]
        approach_rate = (g1 - g0) / dt if dt > 0 else 0.0
    else:
        approach_rate = 0.0

    # Design note 2: window-mean velocity ratio, not a single acceleration.
    def sustained_drop(recs: dict[int, dict]) -> float:
        before = [recs[f]["velocity"] for f in shared[max(0, i_min - window):i_min]]
        after = [recs[f]["velocity"] for f in shared[i_min + 1:i_min + 1 + window]]
        # A truncated window is not evidence. Without this guard, a pair
        # whose closest approach lands at the START or END of the clip has
        # an empty "after" list, mean() returns 0, and the ratio reports a
        # 100% speed loss — a phantom collision for vehicles that merely
        # drove off the edge of the footage. Caught by the bystander pair
        # in the test suite.
        if len(before) < window or len(after) < window:
            return 0.0
        vb, va = mean(before), mean(after)
        return (vb - va) / vb if vb > 1.0 else 0.0

    velocity_drop = max(sustained_drop(fa), sustained_drop(fb))

    # Score. Every convergence term is GATED BY PROXIMITY, because two
    # vehicles converging for 60 frames on opposite sides of the road are
    # not a collision candidate — without this gate, bystanders crossing
    # the frame outranked an actual rear-end impact in the test suite.
    # proximity is 1.0 when boxes touch and decays to 0 once the gap
    # exceeds one vehicle length.
    proximity = max(0.0, 1.0 - min(min_gap_norm, 1.0))

    score = 0.0
    if contact:
        # Contact in a 2D PROJECTION is not proof of physical contact:
        # vehicles at different depths on the road overlap in image space
        # while being far apart in reality. On real footage this saturated
        # — five unrelated pairs all reported minGap 0.00 in one clip.
        # So the contact bonus is earned only by pairs that actually
        # CONVERGED into contact; pairs merely co-occupying screen space
        # have approach_frames near 0 and collect almost nothing.
        score += 50.0 * min(approach_frames / 3.0, 1.0)
    score += min(approach_frames, 20) * 2.0 * proximity
    score += (max(0.0, -approach_rate) / 50.0) * proximity
    score += max(0.0, velocity_drop) * 40.0 * proximity
    score += proximity * 20.0

    return PairScore(
        a=ra[0]["object_id"], b=rb[0]["object_id"],
        shared_frames=len(shared),
        min_gap=round(min_gap_norm, 3),
        min_gap_frame=min_gap_frame,
        approach_rate=round(approach_rate, 1),
        approach_frames=approach_frames,
        velocity_drop=round(velocity_drop, 3),
        contact=contact,
        score=round(score, 2),
    )


def build_events(
    by_id: dict[str, list[dict]],
    pair: PairScore,
    window: int,
    drop_threshold: float,
    min_approach: int,
) -> list[Event]:
    """Turn the winning pair's behaviour into schema event records."""
    ra = {r["frame_index"]: r for r in by_id[pair.a]}
    rb = {r["frame_index"]: r for r in by_id[pair.b]}
    shared = sorted(set(ra) & set(rb))
    events: list[Event] = []

    def at(f: int) -> dict:
        return ra[f]

    # distance_dropping — start of the sustained approach run
    idx = shared.index(pair.min_gap_frame)
    if pair.approach_frames >= 2:
        f_start = shared[max(0, idx - pair.approach_frames)]
        r = at(f_start)
        events.append(Event(
            timestamp=r["timestamp"], frame_index=f_start,
            time_seconds=r["time_seconds"], event="distance_dropping",
            objects_involved=[pair.a, pair.b],
            detail=(f"gap closing for {pair.approach_frames} consecutive frames "
                    f"at {pair.approach_rate:.0f} px/s"),
        ))

    # trajectory_intersecting — directions converge while closing
    for f in shared[max(0, idx - pair.approach_frames):idx + 1]:
        da, db = ra[f]["direction"], rb[f]["direction"]
        diff = abs(da - db) % 360.0
        diff = diff if diff <= 180 else 360 - diff
        if 20.0 < diff < 160.0 and ra[f]["velocity"] > 20 and rb[f]["velocity"] > 20:
            r = at(f)
            events.append(Event(
                timestamp=r["timestamp"], frame_index=f,
                time_seconds=r["time_seconds"], event="trajectory_intersecting",
                objects_involved=[pair.a, pair.b],
                detail=f"headings converge ({da:.0f} deg vs {db:.0f} deg)",
            ))
            break

    # sudden_velocity_change — sustained window drop at closest approach
    if pair.velocity_drop >= drop_threshold:
        r = at(pair.min_gap_frame)
        events.append(Event(
            timestamp=r["timestamp"], frame_index=pair.min_gap_frame,
            time_seconds=r["time_seconds"], event="sudden_velocity_change",
            objects_involved=[pair.a, pair.b],
            detail=(f"mean speed fell {pair.velocity_drop:.0%} across a "
                    f"{window}-frame window"),
        ))

    # collision — contact, reached by sustained approach, plus speed loss.
    #
    # The approach requirement is what makes a LOW drop threshold safe.
    # A real low-speed collision may only shed 10-15% of speed (measured
    # on our footage: a genuine impact showed 13%), so thresholding on
    # speed loss alone would either miss it or, if lowered, fire on every
    # projection overlap. Requiring that the pair converged for several
    # consecutive frames first removes those false pairs, which in turn
    # lets the drop threshold be low enough to catch gentle impacts.
    if (pair.contact
            and pair.approach_frames >= min_approach
            and pair.velocity_drop >= drop_threshold):
        r = at(pair.min_gap_frame)
        events.append(Event(
            timestamp=r["timestamp"], frame_index=pair.min_gap_frame,
            time_seconds=r["time_seconds"], event="collision",
            objects_involved=[pair.a, pair.b],
            detail=(f"boxes met (normalised gap {pair.min_gap:.2f}) with "
                    f"{pair.velocity_drop:.0%} sustained speed loss"),
        ))

    events.sort(key=lambda e: e.frame_index)
    return events


def run(
    records: list[dict],
    window: int = 3,
    contact_ratio: float = 0.05,
    drop_threshold: float = 0.10,
    min_shared: int = 6,
    top: int = 5,
    min_approach: int = 4,
) -> tuple[list[PairScore], list[Event]]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_id[r["object_id"]].append(r)
    for oid in by_id:
        by_id[oid].sort(key=lambda r: r["frame_index"])

    ids = sorted(by_id)
    scored: list[PairScore] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ps = analyse_pair(by_id[ids[i]], by_id[ids[j]],
                              window, contact_ratio, min_shared)
            if ps is not None and ps.score > 0:
                scored.append(ps)

    scored.sort(key=lambda p: p.score, reverse=True)
    events = (build_events(by_id, scored[0], window, drop_threshold, min_approach)
              if scored else [])
    return scored[:top], events


def main() -> int:
    p = argparse.ArgumentParser(
        description="DrishtAI Stage 5: collision detection (sustained-window rules)")
    p.add_argument("motion", help="motion.json from Stage 4")
    p.add_argument("--out", default="events.json", help="output events JSON")
    p.add_argument("--window", type=int, default=3,
                   help="frames averaged either side of impact (default 3)")
    p.add_argument("--contact-ratio", type=float, default=0.05,
                   help="box gap / vehicle size counting as contact (default 0.05)")
    p.add_argument("--drop", type=float, default=0.10,
                   help="sustained speed-loss fraction for impact (default 0.10; "
                        "low on purpose because the approach requirement, not "
                        "this threshold, rejects false pairs)")
    p.add_argument("--min-approach", type=int, default=4,
                   help="consecutive closing frames required before contact "
                        "counts as a collision (default 4)")
    p.add_argument("--min-shared", type=int, default=6,
                   help="minimum shared frames to consider a pair")
    p.add_argument("--top", type=int, default=5, help="how many candidates to print")
    args = p.parse_args()

    src = Path(args.motion)
    if not src.exists():
        print(f"[collision] ERROR: {src} not found", file=sys.stderr)
        return 1
    records = json.loads(src.read_text())
    if not records:
        print("[collision] ERROR: motion file is empty", file=sys.stderr)
        return 1

    ranked, events = run(records, args.window, args.contact_ratio,
                         args.drop, args.min_shared, args.top,
                         args.min_approach)

    print(f"[collision] vehicles: {len({r['object_id'] for r in records})}")
    if not ranked:
        print("[collision] no interacting pairs found — no vehicles ever "
              "approached each other in this clip")
        Path(args.out).write_text("[]")
        return 0

    print(f"[collision] top {len(ranked)} candidate pairs:")
    print(f"{'rank':>4} {'pair':<26} {'score':>7} {'contact':>8} "
          f"{'minGap':>7} {'@frame':>7} {'closing':>8} {'vDrop':>7}")
    for i, ps in enumerate(ranked, 1):
        print(f"{i:>4} {ps.a + ' + ' + ps.b:<26} {ps.score:>7.1f} "
              f"{str(ps.contact):>8} {ps.min_gap:>7.2f} {ps.min_gap_frame:>7} "
              f"{ps.approach_frames:>8} {ps.velocity_drop:>7.0%}")

    best = ranked[0]
    print(f"\n[collision] most likely collision: {best.a} + {best.b} "
          f"at frame {best.min_gap_frame}")
    print(f"[collision] events assigned: {len(events)}")
    for e in events:
        print(f"    {e.timestamp}  frame {e.frame_index:>5}  {e.event:<24} {e.detail}")

    Path(args.out).write_text(json.dumps([asdict(e) for e in events], indent=2))
    print(f"[collision] written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
