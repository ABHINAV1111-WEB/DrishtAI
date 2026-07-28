"""
test_vehicle_labels.py — tests for the vehicle type labelling layer.

Place at: tests/test_vehicle_labels.py

Run:  pytest tests/ -v

No video, no model weights, no GPU — these are pure functions over
dicts, so they run in under a second and are safe to leave in CI.

Every test here corresponds to a specific way this module could break
quietly. Read the docstrings as a list of the failure modes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Import the module under test from src/reasoning/ without requiring an
# installed package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "reasoning"))

from vehicle_labels import (  # noqa: E402
    annotate_records,
    build_class_map,
    humanise_ids,
    label_for,
)


def rec(oid: str, klass: str | None = "car", conf: float | None = 0.8) -> dict:
    """Minimal detection record — only the fields this module reads."""
    out: dict = {"object_id": oid}
    if klass is not None:
        out["vehicle_class"] = klass
    if conf is not None:
        out["confidence"] = conf
    return out


# ---------------------------------------------------------------------------
# build_class_map
# ---------------------------------------------------------------------------

def test_unanimous_track_gets_its_class():
    """The simple case: every frame agrees."""
    records = [rec("vehicle_2", "car") for _ in range(10)]
    assert build_class_map(records) == {"vehicle_2": "car"}


def test_majority_wins_over_flicker():
    """
    YOLO misreading a car as a truck in a couple of frames must not
    change the label. This is the whole reason the module votes rather
    than reading the collision frame.
    """
    records = [rec("vehicle_2", "car") for _ in range(8)]
    records += [rec("vehicle_2", "truck") for _ in range(2)]
    assert build_class_map(records)["vehicle_2"] == "car"


def test_confidence_outweighs_raw_count():
    """
    Tracking runs at conf=0.10, so weak boxes are admitted on purpose.
    Three 0.11 detections must not outvote two 0.95 ones.
    Count: truck 3, car 2.  Weight: truck 0.33, car 1.90.  Car wins.
    """
    records = [rec("vehicle_4", "truck", 0.11) for _ in range(3)]
    records += [rec("vehicle_4", "car", 0.95) for _ in range(2)]
    assert build_class_map(records)["vehicle_4"] == "car"


def test_below_min_detections_gets_no_label():
    """Two sightings is not evidence. Absent from the map, not None."""
    records = [rec("vehicle_9", "bus"), rec("vehicle_9", "bus")]
    assert "vehicle_9" not in build_class_map(records, min_detections=3)


def test_min_detections_boundary_is_inclusive():
    """Exactly min_detections must pass, not fall one short."""
    records = [rec("vehicle_9", "bus") for _ in range(3)]
    assert build_class_map(records, min_detections=3)["vehicle_9"] == "bus"


def test_coin_flip_vote_gets_no_label():
    """
    An even split is not an answer. Observed on real footage:
    vehicle_11 came back truck=1.52 vs car=1.35 — 53/47 — and was
    labelled "truck" as though it were a fact. A wrong vehicle type in
    an incident report is disprovable by looking at the video, which
    makes it worse than saying nothing.
    """
    records = [rec("vehicle_5", "car", 0.5) for _ in range(4)]
    records += [rec("vehicle_5", "truck", 0.5) for _ in range(4)]
    assert "vehicle_5" not in build_class_map(records)


def test_low_total_confidence_gets_no_label():
    """
    Many frames at very low confidence is not the same as evidence.
    Five 0.15 boxes clear min_detections but total only 0.75.
    """
    records = [rec("vehicle_6", "car", 0.15) for _ in range(5)]
    assert "vehicle_6" not in build_class_map(records)


def test_margin_boundary():
    """Exactly at min_margin passes; just under does not."""
    at = [rec("v", "car", 0.6)] * 3 + [rec("v", "truck", 0.4)] * 3
    assert build_class_map(at, min_margin=0.60)["v"] == "car"
    assert "v" not in build_class_map(at, min_margin=0.61)


def test_clear_winner_is_deterministic():
    """
    Identical input must give an identical label on every run. The
    randomised-hash bug in the ID colours was this same failure mode:
    correct output that silently changes between runs.
    """
    records = [rec("vehicle_5", "car", 0.9) for _ in range(8)]
    records += [rec("vehicle_5", "truck", 0.9) for _ in range(2)]
    results = {build_class_map(records)["vehicle_5"] for _ in range(50)}
    assert results == {"car"}


def test_records_missing_class_are_skipped_not_fatal():
    """
    Motion-math records carry no vehicle_class. Handing this function
    the wrong list should degrade to no labels, not raise mid-demo.
    """
    records = [rec("vehicle_2", klass=None) for _ in range(5)]
    assert build_class_map(records) == {}


def test_missing_confidence_falls_back_to_unweighted_vote():
    """Debug records omit confidence. They must still count."""
    records = [rec("vehicle_2", "bus", conf=None) for _ in range(5)]
    assert build_class_map(records)["vehicle_2"] == "bus"


def test_empty_input():
    assert build_class_map([]) == {}


def test_multiple_vehicles_are_independent():
    records = [rec("vehicle_2", "car") for _ in range(5)]
    records += [rec("vehicle_4", "truck") for _ in range(5)]
    assert build_class_map(records) == {"vehicle_2": "car", "vehicle_4": "truck"}


# ---------------------------------------------------------------------------
# label_for
# ---------------------------------------------------------------------------

def test_label_styles():
    cm = {"vehicle_2": "car"}
    assert label_for("vehicle_2", cm) == "vehicle_2 (car)"
    assert label_for("vehicle_2", cm, style="suffix") == "vehicle_2_car"
    assert label_for("vehicle_2", cm, style="plain") == "vehicle_2"


def test_unknown_id_returned_unchanged():
    """No entry means no claim — never invent a type."""
    assert label_for("vehicle_99", {"vehicle_2": "car"}) == "vehicle_99"
    assert label_for("vehicle_99", {}, style="suffix") == "vehicle_99"


# ---------------------------------------------------------------------------
# humanise_ids
# ---------------------------------------------------------------------------

def test_expands_ids_in_a_sentence():
    cm = {"vehicle_2": "car", "vehicle_4": "truck"}
    text = "Collision involving vehicle_2 and vehicle_4."
    assert humanise_ids(text, cm) == (
        "Collision involving vehicle_2 (car) and vehicle_4 (truck)."
    )


def test_first_mention_only():
    """Repeating the type on every mention makes the prose unreadable."""
    cm = {"vehicle_2": "car"}
    text = "vehicle_2 braked, then vehicle_2 stopped, then vehicle_2 was hit."
    assert humanise_ids(text, cm) == (
        "vehicle_2 (car) braked, then vehicle_2 stopped, then vehicle_2 was hit."
    )


def test_first_only_false_expands_every_mention():
    cm = {"vehicle_2": "car"}
    out = humanise_ids("vehicle_2 hit vehicle_2", cm, first_only=False)
    assert out == "vehicle_2 (car) hit vehicle_2 (car)"


def test_two_digit_ids_are_not_corrupted():
    """
    THE IMPORTANT ONE. A naive str.replace("vehicle_2", ...) turns
    "vehicle_20" into "vehicle_2 (car)0". The regex word boundary is
    what prevents it, and this test is what keeps the boundary there.
    """
    cm = {"vehicle_2": "car", "vehicle_20": "bus"}
    out = humanise_ids("vehicle_20 and vehicle_2 collided", cm)
    assert out == "vehicle_20 (bus) and vehicle_2 (car) collided"
    assert "(car)0" not in out


def test_unknown_ids_left_bare_in_text():
    cm = {"vehicle_2": "car"}
    out = humanise_ids("vehicle_2 and vehicle_7 collided", cm)
    assert out == "vehicle_2 (car) and vehicle_7 collided"


def test_suffix_style_in_text():
    cm = {"vehicle_2": "car", "vehicle_4": "truck"}
    out = humanise_ids("vehicle_2 hit vehicle_4", cm, style="suffix")
    assert out == "vehicle_2_car hit vehicle_4_truck"


def test_text_without_ids_is_untouched():
    assert humanise_ids("No collision was detected.", {"vehicle_2": "car"}) == (
        "No collision was detected."
    )


@pytest.mark.parametrize("empty", ["", None])
def test_empty_text_is_safe(empty):
    """The offline template can return an empty string on no-collision."""
    assert humanise_ids(empty, {"vehicle_2": "car"}) == empty


def test_realistic_answer_string():
    """The actual output the UI produces, end to end."""
    cm = {"vehicle_2": "car", "vehicle_4": "truck"}
    text = ("Sudden velocity change occurred 1.87 seconds in, involving "
            "vehicle_2 and vehicle_4 [00:00:01:26].")
    assert humanise_ids(text, cm) == (
        "Sudden velocity change occurred 1.87 seconds in, involving "
        "vehicle_2 (car) and vehicle_4 (truck) [00:00:01:26]."
    )


# ---------------------------------------------------------------------------
# annotate_records
# ---------------------------------------------------------------------------

def test_annotate_adds_stable_field_without_touching_original():
    """
    The per-frame class must stay auditable — we add a field, never
    overwrite what the detector actually said in that frame.
    """
    records = [rec("vehicle_2", "car"), rec("vehicle_2", "truck")]
    annotate_records(records, {"vehicle_2": "car"})
    assert records[1]["vehicle_class"] == "truck"      # untouched
    assert records[1]["vehicle_class_voted"] == "car"  # added


def test_annotate_skips_unknown_ids():
    records = [rec("vehicle_9", "bus")]
    annotate_records(records, {})
    assert "vehicle_class_voted" not in records[0]
