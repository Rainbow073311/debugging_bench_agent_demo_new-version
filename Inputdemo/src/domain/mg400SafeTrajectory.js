import { evaluateMg400PoseReachability } from "./mg400Reachability.js";

const SEGMENT_SAMPLE_COUNT = 24;
const POSITION_EPSILON_MM = 0.001;

function finitePose(pose, label) {
  const clean = {};
  for (const key of ["x", "y", "z", "r"]) {
    const value = Number(pose?.[key]);
    if (!Number.isFinite(value)) {
      throw new Error(`${label} ${key.toUpperCase()} must be a finite number.`);
    }
    clean[key] = value;
  }
  return clean;
}

function boundedSpeed(value, fallback, label) {
  const speed = Number(value ?? fallback);
  if (!Number.isFinite(speed) || speed < 1 || speed > 100) {
    throw new Error(`${label} must be between 1 and 100.`);
  }
  return Math.round(speed);
}

function interpolatePose(start, end, ratio) {
  return Object.fromEntries(
    ["x", "y", "z", "r"].map((key) => [
      key,
      start[key] + ((end[key] - start[key]) * ratio)
    ])
  );
}

function validateLinearSegment(start, end, name) {
  for (let index = 0; index <= SEGMENT_SAMPLE_COUNT; index += 1) {
    const pose = interpolatePose(start, end, index / SEGMENT_SAMPLE_COUNT);
    const reachability = evaluateMg400PoseReachability(pose, {
      allowAdjustment: false,
      enforceLowZStallGuard: false
    });
    if (!reachability.reachable) {
      throw new Error(
        `MG400 safe trajectory ${name} is outside the sampled workspace at sample ${index}/${SEGMENT_SAMPLE_COUNT}: `
        + reachability.message
      );
    }
  }
}

function posesMatch(left, right) {
  return ["x", "y", "z", "r"].every(
    (key) => Math.abs(left[key] - right[key]) <= POSITION_EPSILON_MM
  );
}

export function createMg400SafeTrajectory(currentPose, targetPose, options = {}) {
  const start = finitePose(currentPose, "Current pose");
  const target = finitePose(targetPose, "Target pose");
  const safeTravelZ = Number(options.safeTravelZ ?? 100);
  if (!Number.isFinite(safeTravelZ)) {
    throw new Error("Safe travel Z must be a finite number.");
  }
  const travelSpeed = boundedSpeed(options.travelSpeed, 30, "Travel speed");
  const descentSpeed = boundedSpeed(options.descentSpeed, 10, "Descent speed");
  const travelZ = Math.max(start.z, target.z, safeTravelZ);
  const stagingRadius = options.stagingRadius === undefined
    ? null
    : Number(options.stagingRadius);
  if (stagingRadius !== null && (!Number.isFinite(stagingRadius) || stagingRadius <= 0)) {
    throw new Error("Staging radius must be a positive finite number.");
  }

  const radialPose = (source, z, r) => {
    const theta = Math.atan2(source.y, source.x);
    return {
      x: stagingRadius * Math.cos(theta),
      y: stagingRadius * Math.sin(theta),
      z,
      r
    };
  };
  const candidates = stagingRadius === null
    ? [
        { name: "lift", targetPose: { ...start, z: travelZ }, speed: travelSpeed },
        { name: "traverse", targetPose: { x: target.x, y: target.y, z: travelZ, r: target.r }, speed: travelSpeed },
        { name: "descend", targetPose: target, speed: descentSpeed }
      ]
    : (() => {
        const stagingStart = radialPose(start, start.z, start.r);
        const stagingStartHigh = { ...stagingStart, z: travelZ };
        const stagingTargetHigh = radialPose(target, travelZ, target.r);
        const stagingTarget = { ...stagingTargetHigh, z: target.z };
        return [
          { name: "retract_to_staging", targetPose: stagingStart, speed: travelSpeed },
          { name: "lift_staging", targetPose: stagingStartHigh, speed: travelSpeed },
          { name: "traverse_staging", targetPose: stagingTargetHigh, speed: travelSpeed },
          { name: "descend_staging", targetPose: stagingTarget, speed: descentSpeed },
          { name: "extend_from_staging", targetPose: target, speed: descentSpeed }
        ];
      })();

  const stages = [];
  let segmentStart = start;
  for (const candidate of candidates) {
    if (posesMatch(segmentStart, candidate.targetPose)) continue;
    validateLinearSegment(segmentStart, candidate.targetPose, candidate.name);
    stages.push(candidate);
    segmentStart = candidate.targetPose;
  }

  return {
    mode: "safe-lift-traverse-descend",
    safeTravelZ,
    travelZ,
    travelSpeed,
    descentSpeed,
    stagingRadius,
    startPose: start,
    targetPose: target,
    stages
  };
}
