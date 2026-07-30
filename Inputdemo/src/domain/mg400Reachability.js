const DEG = Math.PI / 180;

const JOINT_LIMITS = {
  j1Deg: [-160, 160],
  j2Deg: [-25, 85],
  j3Deg: [-25, 105],
  j4Deg: [-180, 180]
};

const WORKSPACE_BOUNDARY = [
  [-250.0, 200.0, 360.0],   // 桌面以下 (Z≈-228), 实测 r≈330 可到达
  [-100.0, 200.0, 380.0],   // 桌面以上过渡
  [0.0, 205.0, 400.0],      // 基座平面
  [10.391, 229.49, 328.019],
  [18.604, 228.507, 351.188],
  [26.816, 227.117, 365.915],
  [78.2, 209.433, 423.98],
  [100.0, 197.584, 438.7],
  [150.0, 133.358, 455.014],
  [317.2, 198.192, 423.032],
  [343.2, 197.277, 409.038],
  [396.371, 181.223, 348.0],
  [404.583, 188.86, 332.291],
  [412.796, 204.588, 315.923]
];

const WORKSPACE_MARGIN_MM = 5;
const LOW_Z_STALL_GUARD_MM = -240;  // 桌面 Z≈-228, 保护阈值设在桌面下方
const SAFE_HOVER_Z_MM = -200;       // 桌面上方 ~28mm 安全悬停高度

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function formatPose(pose) {
  return `x=${pose.x}, y=${pose.y}, z=${pose.z}, r=${pose.r}`;
}

function normalizePose(pose) {
  if (!pose || typeof pose !== "object") return { pose: null, errors: ["No MG400 pose was provided."] };
  const clean = {};
  const errors = [];
  for (const key of ["x", "y", "z", "r"]) {
    const value = Number(pose[key]);
    if (!Number.isFinite(value)) errors.push(`${key.toUpperCase()} must be a finite number.`);
    clean[key] = value;
  }
  return { pose: errors.length ? null : clean, errors };
}

function interpolateBoundary(z) {
  const rows = WORKSPACE_BOUNDARY;
  if (z <= rows[0][0]) return { z: rows[0][0], rMin: rows[0][1], rMax: rows[0][2] };
  if (z >= rows[rows.length - 1][0]) {
    const last = rows[rows.length - 1];
    return { z: last[0], rMin: last[1], rMax: last[2] };
  }

  for (let i = 1; i < rows.length; i += 1) {
    const prev = rows[i - 1];
    const next = rows[i];
    if (z <= next[0]) {
      const t = (z - prev[0]) / (next[0] - prev[0]);
      return {
        z,
        rMin: prev[1] + (next[1] - prev[1]) * t,
        rMax: prev[2] + (next[2] - prev[2]) * t
      };
    }
  }

  const last = rows[rows.length - 1];
  return { z: last[0], rMin: last[1], rMax: last[2] };
}

function poseFromPolar(radius, thetaDeg, z, r) {
  const theta = thetaDeg * DEG;
  return {
    x: Number((radius * Math.cos(theta)).toFixed(3)),
    y: Number((radius * Math.sin(theta)).toFixed(3)),
    z: Number(z.toFixed(3)),
    r
  };
}

export function evaluateMg400PoseReachability(poseInput, options = {}) {
  const { allowAdjustment = true } = options;
  const normalized = normalizePose(poseInput);
  if (!normalized.pose) {
    return {
      reachable: false,
      adjusted: false,
      pose: null,
      requestedPose: poseInput || null,
      violations: normalized.errors,
      message: normalized.errors.join(" ")
    };
  }

  const requestedPose = normalized.pose;
  const radius = Math.hypot(requestedPose.x, requestedPose.y);
  const thetaDeg = Math.atan2(requestedPose.y, requestedPose.x) / DEG;
  const violations = [];
  let adjustedThetaDeg = thetaDeg;
  let adjustedRadius = radius;
  let adjustedZ = requestedPose.z;
  let canAdjust = allowAdjustment;

  if (Math.abs(thetaDeg) > JOINT_LIMITS.j1Deg[1]) {
    violations.push(`J1 base angle ${thetaDeg.toFixed(1)} deg exceeds +/-160 deg.`);
    adjustedThetaDeg = clamp(thetaDeg, JOINT_LIMITS.j1Deg[0], JOINT_LIMITS.j1Deg[1]);
  }

  const zLow = WORKSPACE_BOUNDARY[0][0] + WORKSPACE_MARGIN_MM;
  const zHigh = WORKSPACE_BOUNDARY[WORKSPACE_BOUNDARY.length - 1][0] - WORKSPACE_MARGIN_MM;
  if (requestedPose.z < zLow || requestedPose.z > zHigh) {
    violations.push(`Z ${requestedPose.z} mm is outside the sampled MG400 workspace ${zLow.toFixed(1)}-${zHigh.toFixed(1)} mm.`);
    adjustedZ = clamp(requestedPose.z, zLow, zHigh);
  }

  const boundary = interpolateBoundary(adjustedZ);
  const rLow = boundary.rMin + WORKSPACE_MARGIN_MM;
  const rHigh = boundary.rMax - WORKSPACE_MARGIN_MM;
  if (radius < rLow || radius > rHigh) {
    violations.push(`Radial reach ${radius.toFixed(1)} mm is outside ${rLow.toFixed(1)}-${rHigh.toFixed(1)} mm at Z ${adjustedZ.toFixed(1)} mm.`);
    adjustedRadius = clamp(radius, rLow, rHigh);
  }

  const enforceLowZStallGuard = options.enforceLowZStallGuard !== false;
  const lowZEvidenceApplies = enforceLowZStallGuard
    && requestedPose.z < 100
    && radius >= 190
    && radius <= 300;
  if (lowZEvidenceApplies) {
    violations.push(
      "Recent MG400 simulation evidence shows this low-Z measurement band stalls before the target: " +
      "pose x=245.6, y=-32.4, z=78.2 stopped near z=81.9 with 6-7 mm residual error after replanning."
    );
    canAdjust = false;
  }

  const adjustedPose = poseFromPolar(
    adjustedRadius,
    adjustedThetaDeg,
    lowZEvidenceApplies ? Math.max(requestedPose.z, SAFE_HOVER_Z_MM) : adjustedZ,
    requestedPose.r
  );

  if (violations.length === 0) {
    return {
      reachable: true,
      adjusted: false,
      pose: requestedPose,
      requestedPose,
      metrics: { radiusMm: radius, thetaDeg, workspaceBoundary: boundary },
      violations: []
    };
  }

  if (canAdjust) {
    return {
      reachable: true,
      adjusted: true,
      pose: adjustedPose,
      requestedPose,
      metrics: { radiusMm: radius, thetaDeg, workspaceBoundary: boundary },
      violations,
      message: `Adjusted MG400 pose from ${formatPose(requestedPose)} to ${formatPose(adjustedPose)} to stay inside workspace/joint limits.`
    };
  }

  return {
    reachable: false,
    adjusted: false,
    pose: null,
    requestedPose,
    fallbackPose: adjustedPose,
    fallbackAction: "Move to the safe hover pose for visual confirmation, then request a recalibrated measurement point or reposition the fixture.",
    metrics: { radiusMm: radius, thetaDeg, workspaceBoundary: boundary },
    violations,
    jointLimits: JOINT_LIMITS,
    message: `Requested MG400 measurement pose is not reachable before motion: ${violations.join(" ")}`
  };
}
