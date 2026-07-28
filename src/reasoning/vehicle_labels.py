"""
vehicle_labels.py — human-readable names for tracked vehicles.

Turns "vehicle_2 and vehicle_4 collided" into
"vehicle_2 (car) and vehicle_4 (truck) collided".

Place at: src/reasoning/vehicle_labels.py

Design decisions:

1. THE ID NEVER CHANGES. `object_id` is a locked schema field that the
   collision detector, timeline builder and merge provenance all key on.
   The vehicle type is added when text is rendered, not baked into the
   id. Every downstream comparison keeps working, and the id stays
   greppable in the JSON when debugging a demo failure.

2. VOTE ACROSS THE WHOLE TRACK, NOT ONE FRAME. YOLO flickers between
   car and truck on the same physical vehicle — a van reads as "car" in
   one frame and "truck" in the next. Whatever class happened to land on
   the collision frame is not more trustworthy than the other 200
   frames, so we take the winner over every detection of that id.

3. VOTES ARE WEIGHTED BY DETECTOR CONFIDENCE. Tracking runs at
   conf=0.10 so ByteTrack's second association pass has boxes to work
   with. That deliberately admits weak detections, and weak detections
   are where class flicker lives. A 0.91 box should outweigh three 0.12
   boxes rather than being outvoted by them.

4. TIES BREAK ALPHABETICALLY, NOT BY DICT ORDER. Same reason the ID
   colours use a fixed hash: a label that changes between runs on
   identical input is a bug you will only notice on stage.

5. SILENCE BEATS A GUESS. A vehicle seen in fewer than `min_detections`
   frames gets no type at all — the bare id is returned. One or two
   frames is not enough evidence to put a noun in an incident report.

6. FIRST MENTION ONLY. "vehicle_2 (car) braked, then vehicle_2 (car)
   stopped" is noise. Full detail once per id per block of text, bare id
   thereafter.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Mapping, MutableMapping

__all__ = [
    "build_class_map",
    "label_for",
    "humanise_ids",
    "annotate_records",
    "VALID_CLASSES",
]

# The four classes tracker.py filters to (COCO ids 2, 3, 5, 7).
VALID_CLASSES = ("car", "motorcycle", "bus", "truck")

# Matches a whole vehicle id and nothing less. The word boundaries matter:
# without the trailing \b, "vehicle_2" would also match inside "vehicle_20"
# and corrupt every two-digit id in the text.
_VEH_RE = re.compile(r"\bvehicle_\d+\b")


def build_class_map(
    records: Iterable[Mapping[str, Any]],
    min_detections: int = 3,
    min_weight: float = 2.0,
    min_margin: float = 0.60,
) -> dict[str, str]:
    """
    Decide one vehicle class per object_id, voted over all its detections.

    Three independent gates, because they catch different failures:

    min_detections
        Frame count. A vehicle seen twice is not evidence.
    min_weight
        Total confidence summed across all its detections. Ten frames at
        0.11 confidence look like plenty of frames but are almost no
        evidence — this is the gate that catches them.
    min_margin
        The winning class's share of total weight. A 53/47 split is a
        coin flip; printing the winner as fact is the one kind of error
        a viewer can disprove by looking at the video. Below this, the
        vehicle gets no type at all.

    Returns
    -------
    dict mapping object_id -> class name, e.g. {"vehicle_2": "car"}.
    Ids that did not meet the evidence bar are absent, not None, so
    `.get()` returning None means "no claim" with no ambiguity.
    """
    weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)

    for rec in records:
        oid = rec.get("object_id")
        klass = rec.get("vehicle_class")
        if not oid or not klass:
            continue

        conf = rec.get("confidence")
        # Fall back to an unweighted vote when confidence is absent, rather
        # than dropping the detection — some debug records omit it.
        if isinstance(conf, (int, float)) and conf > 0:
            weight = float(conf)
        else:
            weight = 1.0

        weights[oid][klass] += weight
        counts[oid] += 1

    class_map: dict[str, str] = {}
    for oid, votes in weights.items():
        if counts[oid] < min_detections:
            continue

        total = sum(votes.values())
        if total < min_weight:
            continue

        # Sort by descending weight, then by name. The second key makes the
        # result identical across runs when two classes tie exactly.
        winner, top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]

        if total <= 0 or (top / total) < min_margin:
            continue

        class_map[oid] = winner

    return class_map


def label_for(
    object_id: str,
    class_map: Mapping[str, str],
    style: str = "paren",
) -> str:
    """
    Render one id for human eyes.

    style="paren"  -> "vehicle_2 (car)"     [default, recommended]
    style="suffix" -> "vehicle_2_car"
    style="plain"  -> "vehicle_2"

    An unknown id returns unchanged under every style.
    """
    klass = class_map.get(object_id)
    if not klass or style == "plain":
        return object_id
    if style == "suffix":
        return f"{object_id}_{klass}"
    return f"{object_id} ({klass})"


def humanise_ids(
    text: str,
    class_map: Mapping[str, str],
    style: str = "paren",
    first_only: bool = True,
) -> str:
    """
    Expand every vehicle id in a block of text.

    Works on any string the system produces — the API explanation, the
    offline template, the query answer, the causal analysis — because it
    operates on the rendered output rather than on any one code path.
    That is deliberate: one function to maintain, and no way for the
    four paths to drift apart in how they name a vehicle.

    With first_only=True (default) each id is expanded on its first
    appearance and left bare afterwards.
    """
    if not text:
        return text

    seen: set[str] = set()

    def _replace(match: re.Match) -> str:
        oid = match.group(0)
        if first_only and oid in seen:
            return oid
        seen.add(oid)
        return label_for(oid, class_map, style=style)

    return _VEH_RE.sub(_replace, text)


def annotate_records(
    records: Iterable[MutableMapping[str, Any]],
    class_map: Mapping[str, str],
    field: str = "vehicle_class_voted",
) -> None:
    """
    Write the voted class back onto every record, in place.

    Optional. The per-frame `vehicle_class` flickers by design — it is
    whatever YOLO said in that frame. This adds a second, stable
    auxiliary field so the JSON shows both what was seen frame-by-frame
    and what the track as a whole was judged to be.

    Writes to a NEW field name rather than overwriting `vehicle_class`,
    so the raw detector output stays auditable. Auxiliary only — nothing
    downstream is permitted to require it.
    """
    for rec in records:
        oid = rec.get("object_id")
        if oid and oid in class_map:
            rec[field] = class_map[oid]
