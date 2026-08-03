const DEFAULTS = Object.freeze({
  settleMs: 1200,
  burstCount: 3,
  burstIntervalMs: 350,
  maxPoseAgeMs: 1500,
  minSharpness: 35,
  globalZ: 50,
  closeZ: -43.59,
  safeTravelZ: 50,
  stagingRadius: 300,
  fixedR: 7.686619,
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

function parsePose(value, globalZ) {
  if (!value) return null;
  let pose;
  try {
    pose = typeof value === "string" ? JSON.parse(value) : value;
  } catch {
    throw new Error("EYE_IN_HAND_GLOBAL_POSE_JSON must be valid JSON.");
  }
  const clean = {};
  for (const key of ["x", "y"]) {
    clean[key] = finiteNumber(pose?.[key], undefined, `Global pose ${key.toUpperCase()}`);
  }
  clean.z = globalZ;
  // Camera geometry deliberately uses only robot X/Y/Z. R is held at the live value.
  return clean;
}

export function readEyeInHandCaptureConfig(env = process.env) {
  const captureEnabled = enabled(env.ENABLE_EYE_IN_HAND_CAPTURE);
  const motionEnabled = enabled(env.ENABLE_EYE_IN_HAND_MOTION);
  const globalZ = finiteNumber(
    env.EYE_IN_HAND_GLOBAL_Z_MM,
    DEFAULTS.globalZ,
    "Global capture Z"
  );
  const globalPose = parsePose(env.EYE_IN_HAND_GLOBAL_POSE_JSON, globalZ);
  const closeZ = finiteNumber(
    env.EYE_IN_HAND_CLOSE_Z_MM,
    DEFAULTS.closeZ,
    "Close capture Z"
  );

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
    fixedR: finiteNumber(env.EYE_IN_HAND_FIXED_R_DEG, DEFAULTS.fixedR, "Fixed camera R"),
    trajectory: {
      mode: "safe-lift-traverse-descend",
      safeTravelZ: finiteNumber(env.EYE_IN_HAND_SAFE_TRAVEL_Z_MM, DEFAULTS.safeTravelZ, "Safe travel Z"),
      stagingRadius: finiteNumber(env.EYE_IN_HAND_STAGING_RADIUS_MM, DEFAULTS.stagingRadius, "Staging radius"),
      travelSpeed: finiteNumber(env.EYE_IN_HAND_TRAVEL_SPEED, DEFAULTS.travelSpeed, "Travel speed"),
      descentSpeed: finiteNumber(env.EYE_IN_HAND_DESCENT_SPEED, DEFAULTS.descentSpeed, "Descent speed")
    }
  };
}
