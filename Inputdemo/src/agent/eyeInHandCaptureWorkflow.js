import { StepKind } from "../domain/states.js";
import { evaluateMg400PoseReachability } from "../domain/mg400Reachability.js";
import { readEyeInHandCaptureConfig } from "../domain/eyeInHandCaptureConfig.js";

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function modeLabel(status) {
  const mode = status?.robot?.mode;
  if (typeof mode === "string") return mode;
  return mode?.label || (mode?.code === 5 ? "ENABLED_IDLE" : null);
}

function robotPose(status) {
  const pose = status?.robot?.pose;
  if (!pose || !["x", "y", "z"].every((key) => Number.isFinite(Number(pose[key])))) {
    throw new Error("Fresh MG400 XYZ pose is unavailable.");
  }
  return {
    x: Number(pose.x), y: Number(pose.y), z: Number(pose.z),
    r: Number.isFinite(Number(pose.r)) ? Number(pose.r) : 0
  };
}

function motionStep(id, command, targetPose, trajectory) {
  const reachabilityPrecheck = evaluateMg400PoseReachability(targetPose, { allowAdjustment: false });
  if (!reachabilityPrecheck.reachable) {
    throw new Error(`Camera motion blocked before execution: ${reachabilityPrecheck.message}`);
  }
  return {
    id, kind: StepKind.ARM_MOTION, command, targetLocationId: id,
    targetPose, trajectory, reachabilityPrecheck
  };
}

function requireCompleted(result, label) {
  if (result?.status !== "COMPLETED") {
    throw new Error(`${label} failed: ${result?.message || result?.error || result?.status || "unknown result"}`);
  }
}

function requireStableCapturePose(before, after, label, tolerance = 0.05) {
  const deltas = Object.fromEntries(
    ["x", "y", "z", "r"].map((key) => [key, Math.abs(Number(after[key]) - Number(before[key]))])
  );
  if (Math.max(deltas.x, deltas.y, deltas.z, deltas.r) > tolerance) {
    throw new Error(`${label} moved during camera exposure: ${JSON.stringify(deltas)}`);
  }
}

function requireCalibratedPoseRange(pose, health, label) {
  const min = health.validRobotXYZMin;
  const max = health.validRobotXYZMax;
  if (!Array.isArray(min) || !Array.isArray(max) || min.length !== 3 || max.length !== 3) return;
  const values = [pose.x, pose.y, pose.z].map(Number);
  if (values.some((value, index) => !Number.isFinite(value) || value < Number(min[index]) || value > Number(max[index]))) {
    throw new Error(`${label} is outside the calibrated eye-in-hand XYZ range.`);
  }
}

export async function executeEyeInHandCaptureWorkflow({
  run,
  input = run.input,
  armController,
  cameraController,
  config = readEyeInHandCaptureConfig(),
  onEvent = () => {},
  sleep = delay,
  now = () => Date.now()
}) {
  run.execution.camera = {
    status: config.captureEnabled ? "STARTING" : "DISABLED",
    captures: [],
    selectedImage: null
  };
  if (!config.captureEnabled) return run.execution.camera;
  try {
    if (!config.motionEnabled) {
      throw new Error("Eye-in-hand capture is enabled, but ENABLE_EYE_IN_HAND_MOTION is not true.");
    }
    if (!config.globalPose || !Number.isFinite(config.closeZ)) {
      throw new Error("Eye-in-hand motion requires EYE_IN_HAND_GLOBAL_POSE_JSON and EYE_IN_HAND_CLOSE_Z_MM.");
    }
    if (!cameraController) throw new Error("Eye-in-hand camera controller is unavailable.");

    const health = await cameraController.health(config);
    run.execution.camera.calibrationFile = health.calibrationFile || config.calibrationFile || null;
    if (health.calibrationStatus !== "calibrated") {
      throw new Error("Eye-in-hand extrinsics are not calibrated; no robot motion was issued.");
    }

    if (health.tablePlaneConfigured !== true) {
      throw new Error("The calibrated table/PCB plane is missing; no robot motion was issued.");
    }
    if (health.cameraReady !== true) {
      throw new Error("Eye-in-hand camera preflight did not return a frame; no robot motion was issued.");
    }
    const initialStatus = await armController.runCommand("status");
    if (modeLabel(initialStatus) !== "ENABLED_IDLE") {
      throw new Error(`MG400 must be ENABLED_IDLE before camera motion; current mode is ${modeLabel(initialStatus) || "unknown"}.`);
    }
    robotPose(initialStatus);
    const fixedR = Number(health.fixedR ?? config.fixedR);
    if (!Number.isFinite(fixedR)) {
      throw new Error("Eye-in-hand motion requires a calibrated fixed camera R.");
    }
    const globalPose = { ...config.globalPose, r: fixedR };
    requireCalibratedPoseRange(globalPose, health, "Global camera pose");
    requireCalibratedPoseRange({ ...globalPose, z: config.closeZ }, health, "Close camera height");
    const globalStep = motionStep("camera-global-pose", "MOVE_TO_CAMERA_GLOBAL_POSE", globalPose, config.trajectory);
    const globalMove = await armController.execute(globalStep);
    run.execution.arm.push(globalMove);
    requireCompleted(globalMove, "Global camera move");
    onEvent("camera.global_pose_reached", { pose: globalMove.executedPose || globalPose });

    await sleep(config.settleMs);
    const globalStatus = await armController.runCommand("status");
    if (modeLabel(globalStatus) !== "ENABLED_IDLE") throw new Error("MG400 was not idle after reaching the global camera pose.");
    const globalRobotPose = robotPose(globalStatus);
    const globalCapture = await cameraController.capture({ config, robotPose: globalRobotPose, label: "global" });
    const globalStatusAfterCapture = await armController.runCommand("status");
    if (modeLabel(globalStatusAfterCapture) !== "ENABLED_IDLE") throw new Error("MG400 was not idle after the global capture.");
    requireStableCapturePose(globalRobotPose, robotPose(globalStatusAfterCapture), "MG400");
    run.execution.camera.captures.push(globalCapture);
    onEvent("camera.global_capture_finished", { capture: globalCapture });

    const localization = await cameraController.coarseLocalize({ config, capture: globalCapture });
    if (localization.status !== "UNIQUE" || localization.candidateCount !== 1) {
      throw new Error(`PCB coarse localization is not unique (${localization.candidateCount ?? 0} candidates).`);
    }
    const target = localization.recommendedEndXY;
    if (!target || !Number.isFinite(Number(target.x)) || !Number.isFinite(Number(target.y))) {
      throw new Error("PCB coarse localization did not return a calibrated close-pose end X/Y.");
    }
    run.execution.camera.localization = localization;
    onEvent("camera.pcb_coarse_localized", { localization });

    const closePose = { x: Number(target.x), y: Number(target.y), z: config.closeZ, r: fixedR };
    requireCalibratedPoseRange(closePose, health, "Close camera pose");
    const closeStep = motionStep("camera-close-pose", "MOVE_TO_CAMERA_CLOSE_HOVER", closePose, config.trajectory);
    const closeMove = await armController.execute(closeStep);
    run.execution.arm.push(closeMove);
    requireCompleted(closeMove, "Close camera move");
    onEvent("camera.close_pose_reached", { pose: closeMove.executedPose || closePose });

    await sleep(config.settleMs);
    const closeStatus = await armController.runCommand("status");
    if (modeLabel(closeStatus) !== "ENABLED_IDLE") throw new Error("MG400 was not idle before the close burst.");
    const closeRobotPose = robotPose(closeStatus);
    const burst = await cameraController.captureBurst({ config, robotPose: closeRobotPose });
    const closeStatusAfterCapture = await armController.runCommand("status");
    if (modeLabel(closeStatusAfterCapture) !== "ENABLED_IDLE") throw new Error("MG400 was not idle after the close burst.");
    requireStableCapturePose(closeRobotPose, robotPose(closeStatusAfterCapture), "MG400");
    if (!Array.isArray(burst.captures) || burst.captures.length !== 3) {
      throw new Error("Close capture must contain exactly three images.");
    }
    if (!burst.selected || Number(burst.selected.sharpness) < config.minSharpness) {
      throw new Error("None of the three close images passed the sharpness threshold.");
    }
    run.execution.camera.captures.push(...burst.captures);
    run.execution.camera.selectedImage = burst.selected;
    run.execution.camera.status = "COMPLETED";
    input.cameraImage = burst.selectedFile;
    input.visualCapture = {
      imageRef: burst.selectedFile.name, image: burst.selectedFile,
      target: "vlm", source: "eye-in-hand"
    };
    onEvent("camera.burst_capture_finished", { captures: burst.captures, selected: burst.selected });
    onEvent("camera.selected_image_ready", { name: burst.selectedFile.name, robotPose: closeRobotPose });
    return run.execution.camera;
  } catch (error) {
    run.execution.camera.status = "BLOCKED";
    run.execution.camera.error = error.message;
    onEvent("camera.workflow_blocked", { error: error.message, details: error.details || null });
    throw error;
  }
}
