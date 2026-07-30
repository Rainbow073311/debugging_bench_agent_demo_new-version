# MG400 XYZ-only eye-in-hand calibration

## Scope

The camera is treated as an eye-in-hand sensor whose position follows MG400
`X`, `Y`, and `Z`. Robot `R` is explicitly ignored and must not change the
computed camera pose.

For every captured frame:

```text
T_base_to_camera =
T_base_to_end(robot X/Y/Z, identity rotation)
× T_end_to_camera(fixed calibration)
```

This branch does not replace the existing probe Z decision. Probe descent still
starts from the configured fixed Z, moves in the configured small steps, stops
on the first voltage-threshold latch, and never passes the configured minimum Z.
The camera transform only consumes the Z value associated with an image.

## Safety boundary

`calibrate_extrinsics.py` is offline and does not open the camera, connect to
the robot, or send motion commands. `verify_transform.py` opens the camera and
reads robot status, but is read-only and does not move the robot.

## Required calibration data

The calibration marker must remain stationary. Its complete pose in the MG400
base frame must be measured independently. Capture at least five stable samples
covering X, Y, and Z; each sample must bind one image observation to the robot
pose read for that image.

Example JSON structure:

```json
{
  "marker_pose_in_base": {
    "translation_mm": [300.0, 0.0, -228.0],
    "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  },
  "samples": [
    {
      "robot_pose": {"x": 250.0, "y": 0.0, "z": 120.0, "r": 90.0},
      "marker_pose_in_camera": {
        "rvec": [3.14159265, 0.0, 0.0],
        "t_mm": [0.0, 0.0, 348.0]
      }
    }
  ]
}
```

The sample values above only demonstrate the schema; they are not real
calibration measurements.

## Dry-run and write

Review quality without changing the config:

```powershell
python calibration/calibrate_extrinsics.py --dataset path\to\samples.json
```

Write only after reviewing the mean and maximum marker residuals:

```powershell
python calibration/calibrate_extrinsics.py `
  --dataset path\to\samples.json `
  --write
```

The written schema uses:

```yaml
extrinsics:
  mount_mode: eye_in_hand_xyz
  status: calibrated
  robot_axes_used: [x, y, z]
  robot_axes_ignored: [r]
  camera_orientation_source: fixed_calibration
  T_end_to_camera:
    R: ...
    t_mm: ...
```

The former fixed `T_base_to_cam` schema is rejected. Historical values are
retained only under `legacy_extrinsics_do_not_use`.

## Runtime requirements

- Bind the MG400 XYZ pose read for the selected frame.
- Reject missing/stale pose data.
- Keep camera resolution, lens and focus consistent with intrinsic calibration.
- Use the configured table/PCB plane for ray-plane intersection.
- Validate projected known points before allowing hover-only motion.
- Do not descend toward the PCB from a camera-only coordinate result.
