const DEFAULTS = Object.freeze({
  settleMs: 1200,
  burstCount: 3,
  burstIntervalMs: 350,
  maxPoseAgeMs: 1500,
  minSharpness: 35,
  safeTravelZ: 60,
  travelSpeed: 20,
  descentSpeed: 8
});

function enabled(value) {
  return String(value || "").trim().toLowerCase() === "true";
}

function finiteNumber(value, fallback, label) {
  const result = value === undefined || value === "" ? fallback : Number(value);
  if (!Number.isFinite(result)) throw new Error(`${label} must be a finite number.`);
  return result;
}

function parsePose(value) {
  if (!value) return null;
  let pose;
  try {
    pose = typeof value === "string" ? JSON.parse(value) : value;
  } catch {
    throw new Error("EYE_IN_HAND_GLOBAL_POSE_JSON must be valid JSON.");
  }
  const clean = {};
  for (const key of ["x", "y", "z"]) {
    clean[key] = finiteNumber(pose?.[key], undefined, `Global pose ${key.toUpperCase()}`);
  }
  // Camera geometry deliberately uses only robot X/Y/Z. R is held at the live value.
  return clean;
}

export function readEyeInHandCaptureConfig(env = process.env) {
  const captureEnabled = enabled(env.ENABLE_EYE_IN_HAND_CAPTURE);
  const motionEnabled = enabled(env.ENABLE_EYE_IN_HAND_MOTION);
  const globalPose = parsePose(env.EYE_IN_HAND_GLOBAL_POSE_JSON);
  const closeZ = env.EYE_IN_HAND_CLOSE_Z_MM === undefined
    ? null
    : finiteNumber(env.EYE_IN_HAND_CLOSE_Z_MM, undefined, "Close capture Z");

  return {
    captureEnabled,
    motionEnabled,
    globalPose,
    closeZ,
    cameraIndex: Math.round(finiteNumber(env.EYE_IN_HAND_CAMERA_INDEX, 0, "Camera index")),
    outputDir: env.EYE_IN_HAND_CAPTURE_DIR || null,
    calibrationFile: env.EYE_IN_HAND_CALIBRATION_FILE || null,
    settleMs: Math.max(0, finiteNumber(env.EYE_IN_HAND_SETTLE_MS, DEFAULTS.settleMs, "Settle time")),
    burstCount: 3,
    burstIntervalMs: Math.max(0, finiteNumber(env.EYE_IN_HAND_BURST_INTERVAL_MS, DEFAULTS.burstIntervalMs, "Burst interval")),
    maxPoseAgeMs: Math.max(1, finiteNumber(env.EYE_IN_HAND_MAX_POSE_AGE_MS, DEFAULTS.maxPoseAgeMs, "Maximum pose age")),
    minSharpness: Math.max(0, finiteNumber(env.EYE_IN_HAND_MIN_SHARPNESS, DEFAULTS.minSharpness, "Minimum sharpness")),
    trajectory: {
      mode: "safe-lift-traverse-descend",
      safeTravelZ: finiteNumber(env.EYE_IN_HAND_SAFE_TRAVEL_Z_MM, DEFAULTS.safeTravelZ, "Safe travel Z"),
      travelSpeed: finiteNumber(env.EYE_IN_HAND_TRAVEL_SPEED, DEFAULTS.travelSpeed, "Travel speed"),
      descentSpeed: finiteNumber(env.EYE_IN_HAND_DESCENT_SPEED, DEFAULTS.descentSpeed, "Descent speed")
    }
  };
}
