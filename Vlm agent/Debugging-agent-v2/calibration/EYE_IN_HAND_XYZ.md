# MG400 XYZ-only eye-in-hand calibration

## Scope

The camera is treated as an eye-in-hand sensor whose position follows MG400
`X`, `Y`, and `Z`. Robot `R` is not an input feature of the camera transform,
but the physical arm must remain at the calibrated fixed value `7.686619`
degrees. Runtime projection is rejected when R differs by more than 0.1 degree.

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
  "target_plane_z_mm": -228.0,
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
The target plane Z is the PCB surface height expressed in the MG400 base frame;
it is required when writing the calibration and must be measured for the real
fixture.

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
table_homography:
  table_z_mm: -228.0
  source: eye_in_hand_target_plane_in_robot_base
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


## Integrated automatic capture workflow

Both the CLI `BenchAgent` and the web service run the same guarded pre-VLM
workflow:

1. read MG400 status and require `ENABLED_IDLE`;
2. move to the configured high overview pose using a safe lift/traverse/descend
   trajectory;
3. wait for settling and bind the fresh MG400 XYZ pose to the overview frame;
4. find exactly one PCB candidate with deterministic contour geometry;
5. project its center through the calibrated eye-in-hand model and compensate
   the camera-to-end offset at the configured close-capture Z;
6. move only to that close hover pose, wait, and capture exactly three frames;
7. select the frame with the largest Laplacian sharpness score;
8. store all image timestamps, sharpness values and robot poses, then inject the
   selected image as `front_board_photo` before the VLM case is created.

The workflow is off by default. Real use requires both gates and explicit
installation-specific poses:

```powershell
$env:ENABLE_EYE_IN_HAND_CAPTURE = "true"
$env:ENABLE_EYE_IN_HAND_MOTION = "true"
$env:EYE_IN_HAND_GLOBAL_POSE_JSON = '{"x":300,"y":0,"z":120}'
$env:EYE_IN_HAND_CLOSE_Z_MM = "-150"
```

Those numeric examples are schema examples, not validated poses for the real
installation. Set them only after reachability and clearance review. Optional
settings include `EYE_IN_HAND_CAMERA_INDEX`, `EYE_IN_HAND_SETTLE_MS`,
`EYE_IN_HAND_MIN_SHARPNESS`, `EYE_IN_HAND_CAPTURE_DIR`, and
`EYE_IN_HAND_CALIBRATION_FILE`.

Motion is blocked before the first command when calibration is not
`calibrated`, and close motion is blocked for zero or multiple PCB candidates.
Robot R is commanded to the calibrated fixed value for the physical move but
never participates as a fitted geometry feature. The later probe routine remains separate: its fixed start,
0.1 mm steps, first 3 V latch, and minimum-Z stop are unchanged.
