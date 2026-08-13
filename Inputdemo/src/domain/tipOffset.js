/**
 * Locked MG400 tip↔TCP XY offset.
 *
 * tip_xy = TCP_xy + 27 * (sin(J1+δ), -cos(J1+δ))
 * J1 ≈ atan2(TCP_y, TCP_x) [deg]
 *
 * Source of truth: ../../calibration/tip_offset.json
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_JSON = path.resolve(HERE, "../../../calibration/tip_offset.json");

const FALLBACK = {
  status: "calibrated",
  radius_xy_mm: 27.0,
  delta_deg: 6.15,
  z_mm: 0.0,
  locked: true
};

export function loadTipOffsetParams(jsonPath = DEFAULT_JSON) {
  try {
    const raw = JSON.parse(readFileSync(jsonPath, "utf8"));
    if (raw?.status !== "calibrated" && raw?.status !== "approx_calibrated") {
      throw new Error(`tip_offset status is ${raw?.status}`);
    }
    return {
      status: raw.status,
      radius_xy_mm: Number(raw.radius_xy_mm),
      delta_deg: Number(raw.delta_deg),
      z_mm: Number(raw.z_mm ?? 0),
      locked: Boolean(raw.locked),
      source: raw.source || null
    };
  } catch {
    return { ...FALLBACK };
  }
}

export function j1FromTcpXy(xMm, yMm) {
  return (Math.atan2(Number(yMm), Number(xMm)) * 180) / Math.PI;
}

export function tipOffsetXy(j1Deg, params = loadTipOffsetParams()) {
  const ang = ((Number(j1Deg) + Number(params.delta_deg)) * Math.PI) / 180;
  const r = Number(params.radius_xy_mm);
  return { dx: r * Math.sin(ang), dy: -r * Math.cos(ang) };
}

export function tipXyFromTcp(tcpXy, j1Deg = null, params = loadTipOffsetParams()) {
  const x = Number(tcpXy.x ?? tcpXy[0]);
  const y = Number(tcpXy.y ?? tcpXy[1]);
  const j1 = j1Deg == null ? j1FromTcpXy(x, y) : Number(j1Deg);
  const { dx, dy } = tipOffsetXy(j1, params);
  return { x: x + dx, y: y + dy, j1_deg: j1, offset: { dx, dy } };
}

/**
 * Solve TCP so tip lands on tipXy (J1-consistent iteration).
 */
export function tcpXyFromTip(tipXy, options = {}) {
  const params = options.params || loadTipOffsetParams();
  const tx = Number(tipXy.x ?? tipXy[0]);
  const ty = Number(tipXy.y ?? tipXy[1]);
  const iterations = options.iterations ?? 8;
  let j1 =
    options.j1HintDeg == null ? j1FromTcpXy(tx, ty) : Number(options.j1HintDeg);
  let x = 0;
  let y = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < iterations; i += 1) {
    ({ dx, dy } = tipOffsetXy(j1, params));
    x = tx - dx;
    y = ty - dy;
    j1 = j1FromTcpXy(x, y);
  }
  const tipCheckX = x + dx;
  const tipCheckY = y + dy;
  const errMm = Math.hypot(tipCheckX - tx, tipCheckY - ty);
  return {
    tcpXy: { x, y },
    j1_deg: j1,
    offset: { dx, dy },
    tipCheckXy: { x: tipCheckX, y: tipCheckY },
    errMm,
    params: {
      radius_xy_mm: params.radius_xy_mm,
      delta_deg: params.delta_deg
    }
  };
}

/** Convert camera/base tip target pose into a TCP command pose (XY only). */
export function tipTargetToTcpPose(tipPose, options = {}) {
  const solved = tcpXyFromTip(tipPose, options);
  return {
    ...solved,
    pose: {
      x: solved.tcpXy.x,
      y: solved.tcpXy.y,
      z: tipPose.z,
      r: tipPose.r
    }
  };
}
