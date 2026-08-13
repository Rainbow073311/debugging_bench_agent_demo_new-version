# MG400 J1 eye-in-hand calibration

## Scope

The camera is treated as an eye-in-hand sensor whose pose follows MG400
`X`, `Y`, `Z` and base joint `J1`:

```text
T_base_to_camera =
Trans(robot X/Y/Z)
× Rz(J1)
× T_end_to_camera(fixed calibration in the J1 / arm-head frame)
```

`J1` is taken from `pose.j1_deg` / `pose.j1` when present, otherwise
`atan2(TCP_y, TCP_x)`. Flange `R` (J4) is not an input of the camera mount
transform. When the optional `xy_pose_correction` is enabled, runtime still
rejects flange R outside the calibrated guard band.

This branch does not replace the existing probe Z decision. Probe descent still
starts from the configured fixed Z, moves in the configured small steps, stops
on the first voltage-threshold latch, and never passes the configured minimum Z.
The camera transform only consumes the Z value associated with an image.

## Safety boundary

`calibrate_extrinsics.py` is offline and does not open the camera, connect to
the robot, or send motion commands. `verify_transform.py` opens the camera and
reads robot status, but is read-only and does not move the robot. The two
`run_basler_extrinsics_*_sequence.py` tools do move the robot and may only be
run after explicit authorization for the fixed-R calibration trajectory.

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

Review fit quality and independent validation without changing the config:

```powershell
python calibration/calibrate_extrinsics.py `
  --dataset path\to\samples.json `
  --validation-dataset path\to\validation_samples.json
```

Write only after reviewing the mean and maximum marker residuals:

```powershell
python calibration/calibrate_extrinsics.py `
  --dataset path\to\samples.json `
  --validation-dataset path\to\validation_samples.json `
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

## Basler fixed-R acquisition

For the current Basler installation, `capture_basler_extrinsics_sample.py`
requires `ENABLED_IDLE`, verifies that XYZ/R remain stable during exposure,
rejects incomplete 9x6 detections and rejects PnP reprojection RMSE above
1.5 px. `run_basler_extrinsics_sequence.py` fixes R at `7.686619` degrees,
resumes from the existing sample count, uses a Z=50 mm travel height, and
returns to the Z=50 mm safe pose on exit. Independent frames are collected by
`run_basler_extrinsics_validation_sequence.py` into `validation_samples.json`;
they are never included in the fitted transform.

The XYZ-only solver averages full-board PnP orientations, then fits the fixed
camera offset from synchronized camera-frame marker translations and robot XYZ
positions. A Kabsch translation-only orientation is retained as a cross-check;
the full board is the primary orientation baseline for pixel projection. The
solver rejects insufficient axis span and weak translation geometry before
writing a transform.

## Runtime requirements

- Bind the MG400 XYZ pose read for the selected frame.
- Hold robot R at `7.686619` degrees for every overview and close frame.
- Reject missing/stale pose data.
- Reject projection outside robot X `305.49996..380.499967`, Y
  `-75.800005..-5.8`, or Z `50..140` mm.
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
9. project the VLM-selected pixel from that close image into Base XY using the
   close image's synchronized robot XYZ; pause automatic contact motion until
   the calibrated XY has passed an independent hover validation.

The overview image therefore chooses the close-camera area. The selected close
image determines the final PCB test-point XY. The visual path never supplies
the contact Z; contact Z remains the responsibility of the voltage-latched
probe descent.

The workflow is off by default. Real use requires both gates and explicit
installation-specific poses:

```powershell
$env:ENABLE_EYE_IN_HAND_CAPTURE = "true"
$env:ENABLE_EYE_IN_HAND_MOTION = "true"
$env:EYE_IN_HAND_GLOBAL_POSE_JSON = '{"x":350,"y":-20.9156}'
$env:EYE_IN_HAND_GLOBAL_Z_MM = "50"
$env:EYE_IN_HAND_CLOSE_Z_MM = "-43.59"
$env:EYE_IN_HAND_SAFE_TRAVEL_Z_MM = "50"
```

The global pose above was included in the 5 mm-board calibration set. Recheck
clearance whenever the fixture or camera mount changes. Optional
settings include `EYE_IN_HAND_CAMERA_INDEX`, `EYE_IN_HAND_SETTLE_MS`,
`EYE_IN_HAND_MIN_SHARPNESS`, `EYE_IN_HAND_CAPTURE_DIR`, and
`EYE_IN_HAND_CALIBRATION_FILE`. `EYE_IN_HAND_STAGING_RADIUS_MM` defaults to
`300` for the validated inner-radius height transition.

The active 5 mm-board session covers close capture at `Z=-43.59 mm`, an
intermediate layer at `Z=25 mm`, and global capture at `Z=50 mm`. Horizontal
travel uses `Z=50 mm`; the narrow workspace near `Z=10 mm` is crossed only at
an inner-radius staging pose by the calibration collection scripts.

Motion is blocked before the first command when calibration is not
`calibrated`, and close motion is blocked for zero or multiple PCB candidates.
Robot R is commanded to the calibrated fixed value for the physical move but
never participates as a fitted geometry feature. After ray-plane projection,
a pose-aware XY residual correction uses the image's robot XYZ. Its independent
three-frame holdout error is 0.63 mm mean and 1.71 mm maximum. The later probe routine remains separate: its fixed start,
0.1 mm steps, first 3 V latch, and minimum-Z stop are unchanged.
