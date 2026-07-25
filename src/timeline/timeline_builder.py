"""
timeline_builder.py — DrishtAI pipeline, Stage 6 (timeline + earliest warning)

Takes events.json (Stage 5) and produces the time-sorted event chain in the
schema.md Timeline Record format, with exactly one event flagged
`is_earliest_warning`.

This is the stage the pitch is actually about. Stages 1-5 answer "what
happened and when". This one answers "what was the FIRST observable moment
that made the outcome likely" — the difference between a system that
reports accidents and one that could have warned about them.

Design decisions:

1. EARLIEST WARNING IS FOUND BY WALKING THE CHAIN BACKWARDS, not by taking
   the first event in the file. Start at the collision, collect the vehicles
   involved, then walk back through time keeping only events that (a) share
   at least one vehicle with the collision and (b) are risk-elevating. The
   earliest surviving event is the warning. Taking the chronologically first
   event instead would happily flag an unrelated vehicle's manoeuvre at the
   start of the clip.

2. CAUSAL LINKAGE IS BY SHARED object_id. An event involving neither crash
   vehicle cannot be the warning for that crash, however early it occurred.
   This is why persistent ids from Stage 3 matter: without stable ids there
   is no way to say "this earlier event involved the same car".

3. `moving_normally` IS NOT RISK-ELEVATING and can never be the warning. It
   is emitted into the timeline as baseline context (schema.md's example
   timeline opens with two such records) so the Claude explanation layer can
   describe what normal looked like before things changed, but it is
   excluded from warning candidacy.

4. EXACTLY ONE EVENT CARRIES is_earliest_warning=true, per schema.md. Every
   other event gets false rather than omitting the key, so a consumer never
   has to distinguish "false" from "missing".

5. LEAD TIME IS REPORTED. The gap between the earliest warning and the
   collision is the single number that quantifies the project's claim. It
   is computed from time_seconds and printed; it is not written into the
   schema record, which stays exactly as schema.md defines it.

Usage:
    python src/timeline/timeline_builder.py events.json --out timeline.json
    python src/timeline/timeline_builder.py events.json --out timeline.json --no-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


# Ordered from least to most severe. Order matters for choosing the anchor
# when no explicit collision event exists.
RISK_EVENTS = [
    "distance_dropping",
    "trajectory_intersecting",
    "sudden_velocity_change",
    "collision",
]

NON_RISK_EVENTS = ["moving_normally"]


@dataclass
class TimelineRecord:
    """
    The shape schema.md specifies for a timeline record, plus the auxiliary
    field `time_seconds`.

    time_seconds and frame_index are carried through deliberately.
    schema.md lists frame_index under Auxiliary Fields as the way to "locate
    a record in extracted frames" — the UI needs it to show the impact
    still, and the causal analysis needs it to find the motion records
    around the impact.

    On time_seconds specifically: schema.md lists it under
    Auxiliary Fields as "the numeric source of truth for all timing math;
    the HH:MM:SS:FF string is for display and schema compliance". Dropping
    it here forced Stage 7 to reconstruct seconds by parsing HH:MM:SS:FF,
    which is impossible without knowing the source fps — FF is frames
    within a second, so the same string means different times at 30 and 60
    fps. That reconstruction silently reported a 1.22 s lead time for a
    clip whose true lead time was 1.43 s. Carrying the float removes the
    conversion entirely rather than making the caller supply fps.
    """
    timestamp: str
    event: str
    objects_involved: list[str]
    is_earliest_warning: bool
    time_seconds: float
    frame_index: int


def find_anchor(events: list[dict]) -> dict | None:
    """
    The anchor is the event the warning is a warning ABOUT.

    Prefer an explicit collision. If the detector found none (a near-miss,
    or a clip where contact was never confirmed), fall back to the most
    severe risk event present, so the chain still has something to reason
    backwards from rather than silently producing no warning.
    """
    collisions = [e for e in events if e["event"] == "collision"]
    if collisions:
        return max(collisions, key=lambda e: e["time_seconds"])

    for level in reversed(RISK_EVENTS[:-1]):
        matches = [e for e in events if e["event"] == level]
        if matches:
            return max(matches, key=lambda e: e["time_seconds"])
    return None


def find_earliest_warning(events: list[dict], anchor: dict) -> dict | None:
    """
    Walk backwards from the anchor and return the first risk-elevating
    event causally linked to it (design notes 1 and 2).
    """
    involved = set(anchor["objects_involved"])

    candidates = [
        e for e in events
        if e["time_seconds"] <= anchor["time_seconds"]
        and e["event"] in RISK_EVENTS
        and involved & set(e["objects_involved"])
    ]
    if not candidates:
        return None

    # Walking backwards and keeping the last survivor is equivalent to
    # taking the earliest candidate, but expressed as the backward walk the
    # design calls for — and it makes the tie-break explicit: on equal
    # timestamps, prefer the LESS severe event, because the milder signal
    # is the earlier warning in causal terms.
    earliest = None
    for e in sorted(candidates, key=lambda e: e["time_seconds"], reverse=True):
        if earliest is None or e["time_seconds"] <= earliest["time_seconds"]:
            if (earliest is None
                    or e["time_seconds"] < earliest["time_seconds"]
                    or RISK_EVENTS.index(e["event"]) < RISK_EVENTS.index(earliest["event"])):
                earliest = e
    return earliest


def build_timeline(
    events: list[dict],
    include_baseline: bool = True,
) -> tuple[list[TimelineRecord], dict | None, dict | None, float | None]:
    if not events:
        return [], None, None, None

    # Sort by time, then by SEVERITY rather than alphabetically. Two events
    # can share a timestamp (an impact registers as both a velocity change
    # and a collision in the same frame), and alphabetical order would put
    # "collision" before "sudden_velocity_change" — narrating the effect
    # before its cause. The explanation layer reads this order directly, so
    # a wrong order becomes a wrong sentence.
    def order_key(e: dict) -> tuple[float, int]:
        severity = (RISK_EVENTS.index(e["event"])
                    if e["event"] in RISK_EVENTS else -1)
        return (e["time_seconds"], severity)

    events = sorted(events, key=order_key)
    anchor = find_anchor(events)
    warning = find_earliest_warning(events, anchor) if anchor else None

    records: list[TimelineRecord] = []

    # Design note 3: open with baseline context for the involved vehicles,
    # matching the shape of schema.md's example timeline.
    if include_baseline and warning:
        for oid in sorted(warning["objects_involved"]):
            records.append(TimelineRecord(
                timestamp=warning["timestamp"],
                event="moving_normally",
                objects_involved=[oid],
                is_earliest_warning=False,
                time_seconds=warning["time_seconds"],
                frame_index=warning["frame_index"],
            ))

    for e in events:
        records.append(TimelineRecord(
            timestamp=e["timestamp"],
            event=e["event"],
            objects_involved=e["objects_involved"],
            is_earliest_warning=bool(
                warning is not None
                and e["time_seconds"] == warning["time_seconds"]
                and e["event"] == warning["event"]
                and e["objects_involved"] == warning["objects_involved"]
            ),
            time_seconds=e["time_seconds"],
            frame_index=e["frame_index"],
        ))

    lead = None
    if warning and anchor:
        lead = round(anchor["time_seconds"] - warning["time_seconds"], 3)

    return records, anchor, warning, lead


def main() -> int:
    p = argparse.ArgumentParser(
        description="DrishtAI Stage 6: timeline builder + earliest-warning logic")
    p.add_argument("events", help="events.json from Stage 5")
    p.add_argument("--out", default="timeline.json", help="output timeline JSON")
    p.add_argument("--no-baseline", action="store_true",
                   help="omit the leading moving_normally context records")
    args = p.parse_args()

    src = Path(args.events)
    if not src.exists():
        print(f"[timeline] ERROR: {src} not found", file=sys.stderr)
        return 1
    events = json.loads(src.read_text())
    if not events:
        print("[timeline] no events to build a timeline from — Stage 5 found "
              "nothing in this clip", file=sys.stderr)
        Path(args.out).write_text("[]")
        return 0

    records, anchor, warning, lead = build_timeline(
        events, include_baseline=not args.no_baseline)

    print(f"[timeline] events in: {len(events)}   timeline records: {len(records)}")
    if anchor:
        print(f"[timeline] anchor event: {anchor['event']} at {anchor['timestamp']} "
              f"involving {anchor['objects_involved']}")
    else:
        print("[timeline] no risk anchor found")

    if warning:
        print(f"[timeline] EARLIEST WARNING: {warning['event']} at "
              f"{warning['timestamp']} involving {warning['objects_involved']}")
        print(f"[timeline] detail: {warning.get('detail', '(none)')}")
        if lead is not None:
            print(f"[timeline] LEAD TIME: {lead:.2f} s before the anchor event")
    else:
        print("[timeline] no earliest warning identified")

    print("\n[timeline] chain:")
    for r in records:
        mark = "  <== EARLIEST WARNING" if r.is_earliest_warning else ""
        print(f"    {r.timestamp}  {r.event:<24} {r.objects_involved}{mark}")

    Path(args.out).write_text(json.dumps([asdict(r) for r in records], indent=2))
    print(f"\n[timeline] written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
