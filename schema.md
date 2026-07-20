# DrishtAI — Event JSON Schema

This is the locked data structure passed between pipeline stages.
Do not change field names without updating all dependent modules.

## Per-Object Detection Record

```json
{
  "object_id": "vehicle_4",
  "timestamp": "00:14:32:08",
  "position": [x, y],
  "velocity": 0.0,
  "direction": 0.0,
  "event": "moving_normally"
}
```

## Field Definitions

- **object_id**: unique tracked ID for a vehicle, assigned by the tracker (e.g., "vehicle_4")
- **timestamp**: frame timestamp in HH:MM:SS:FF format (FF = frame number within the second)
- **position**: [x, y] pixel coordinates of the object's centroid in that frame
- **velocity**: pixels-per-frame speed, calculated from position change across frames
- **direction**: angle in degrees representing movement direction
- **event**: one of: "moving_normally", "distance_dropping", "trajectory_intersecting", "sudden_velocity_change", "collision"

## Event Timeline Format (after Timeline Builder)

```json
{
  "timestamp": "00:14:32:14",
  "event": "distance_dropping",
  "objects_involved": ["vehicle_4", "vehicle_7"],
  "is_earliest_warning": true
}
```