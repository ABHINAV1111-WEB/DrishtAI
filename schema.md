# DrishtAI — Event JSON Schema

This is the locked data structure passed between pipeline stages.
Do not change field names without updating all dependent modules and
flagging the change explicitly.

## Per-Object Detection Record

Produced by: YOLO detection + tracking + motion math stages.

```json
{
  "object_id": "vehicle_4",
  "timestamp": "00:14:32:08",
  "position": [x, y],
  "velocity": 0.0,
  "direction": 0.0,
  "acceleration": 0.0,
  "event": "moving_normally"
}
```

### Field Definitions

- **object_id**: unique tracked ID for a vehicle, assigned by the tracker
  (e.g., "vehicle_4"). Must stay consistent across frames for the same
  physical vehicle.
- **timestamp**: frame timestamp in `HH:MM:SS:FF` format (FF = frame number
  within that second, based on video FPS).
- **position**: `[x, y]` pixel coordinates of the object's bounding-box
  centroid in that frame.
- **velocity**: pixels-per-second speed, calculated as position change
  divided by the actual elapsed time between consecutive records for that
  `object_id`.

  *Changed from pixels-per-frame (see Revision History).* Per-frame is
  ambiguous once frames are sampled at an interval — per source frame or
  per processed frame differ by a factor of the interval, and neither
  errors visibly. Pixels-per-second is invariant to both FPS and sampling
  interval, so thresholds tuned on one clip transfer to another.

  Elapsed time must always be derived from `time_seconds` deltas, never
  assumed from the sampling interval: a vehicle missing for several frames
  and reappearing has a real gap larger than one interval.
- **direction**: angle in degrees representing movement direction
  (0-360, standard unit circle convention: 0 = right, 90 = up).

  Note that image coordinates place y=0 at the top with y growing downward,
  so producing this convention requires negating the y-delta
  (`atan2(-dy, dx)`). Without that flip, upward motion reports 270 instead
  of 90 and all trajectory comparisons are mirrored.
- **acceleration**: change in velocity per second, in px/s², calculated
  from the velocity series of a single `object_id`. Positive means
  speeding up, negative means slowing down.

  Consumed by the collision detector to identify `sudden_velocity_change`
  — a sharp negative acceleration is the signature of impact or emergency
  braking.
- **event**: one of:
  - `"moving_normally"`
  - `"distance_dropping"`
  - `"trajectory_intersecting"`
  - `"sudden_velocity_change"`
  - `"collision"`

  Assigned by the collision detector stage. Motion math does not populate
  this field — an unassigned record has no `event` key rather than a
  default value, so "not yet evaluated" is never confused with
  "evaluated as normal".

## Event Timeline Record (after Timeline Builder)

Produced by: Event Timeline Builder stage, consumed by Earliest-Warning
Logic and the Claude API explanation layer.

```json
{
  "timestamp": "00:14:32:14",
  "event": "distance_dropping",
  "objects_involved": ["vehicle_4", "vehicle_7"],
  "is_earliest_warning": true,
  "time_seconds": 872.4667,
  "frame_index": 26174
}
```

### Field Definitions

- **timestamp**: same `HH:MM:SS:FF` format as above.
- **event**: same event vocabulary as above, describing what changed at
  this point in the timeline.
- **objects_involved**: list of `object_id`s relevant to this event.
- **is_earliest_warning**: `true` only on the single event in the chain
  identified as the first meaningfully risk-elevating moment. `false` or
  omitted on all other events in the chain.
- **time_seconds** and **frame_index**: auxiliary fields (defined below)
  carried into the timeline rather than dropped.

  Consumers of the timeline must not reconstruct seconds by parsing the
  `HH:MM:SS:FF` string. `FF` is a frame count within the second, so the
  same string means different times at different frame rates — 00:00:01:25
  is 1.42 s at 60 fps and 1.83 s at 30 fps. Reconstructing it once produced
  a lead time of 1.22 s for a clip whose true lead time was 1.43 s, with no
  error raised. Carrying the float forward removes the conversion entirely;
  `frame_index` likewise lets the UI locate the exact frame without
  recomputation.

## Full Timeline Example (Road Accident)

```json
[
  {"timestamp": "00:14:32:08", "event": "moving_normally", "objects_involved": ["vehicle_4"], "is_earliest_warning": false},
  {"timestamp": "00:14:32:11", "event": "moving_normally", "objects_involved": ["vehicle_7"], "is_earliest_warning": false},
  {"timestamp": "00:14:32:14", "event": "distance_dropping", "objects_involved": ["vehicle_4", "vehicle_7"], "is_earliest_warning": true},
  {"timestamp": "00:14:32:16", "event": "trajectory_intersecting", "objects_involved": ["vehicle_4", "vehicle_7"], "is_earliest_warning": false},
  {"timestamp": "00:14:32:18", "event": "sudden_velocity_change", "objects_involved": ["vehicle_4", "vehicle_7"], "is_earliest_warning": false},
  {"timestamp": "00:14:32:19", "event": "collision", "objects_involved": ["vehicle_4", "vehicle_7"], "is_earliest_warning": false}
]
```

This is the exact shape of data the Claude API explanation layer receives
and turns into plain-language output for the Streamlit UI.

## Auxiliary Fields (not part of the locked record)

These are emitted by pipeline stages for debugging and provenance. They are
not required by downstream consumers and may be dropped without breaking
anything.

- **frame_index**: absolute frame index in the source video. Used to locate
  a record in extracted frames and annotated output.
- **time_seconds**: exact float time of the frame. The numeric source of
  truth for all timing math; the `HH:MM:SS:FF` string is for display and
  schema compliance.
- **bbox**: `[x1, y1, x2, y2]` pixel corners of the detection box.
  `position` is this box's centroid.
- **confidence**: detector confidence 0-1 for that box.
- **vehicle_class**: one of `"car"`, `"motorcycle"`, `"bus"`, `"truck"`.
- **merged_from**: list of `object_id`s absorbed into this record by the
  articulated-vehicle merge in motion math. Empty for unmerged vehicles.
  Present so a merge is auditable rather than silent.

## Revision History

- **v3** — `time_seconds` and `frame_index` carried into the Event Timeline
  Record so downstream stages never re-derive time from the display
  timestamp.
- **v2** — `velocity` redefined from pixels-per-frame to pixels-per-second;
  `acceleration` added to the per-object record; explicit note added that
  `event` is unset until the collision detector assigns it.
- **v1** — initial locked schema (Day 1).
