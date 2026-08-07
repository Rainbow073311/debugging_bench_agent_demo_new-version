// Quick test: use case-310 step08 results to trigger arm movement
import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspace = "C:/Users/32825/Desktop/new_version_demo/new_vlm_agent/workspace/vlm-agent-runs/case-310-run_mse6k2vu_7sv9wazj";

// Read step08 results
const tps = ["TP11", "TP9"];
const points = [];
for (const tp of tps) {
  const p = `${workspace}/${tp}/debug/step08_result.json`;
  if (existsSync(p)) {
    const data = JSON.parse(readFileSync(p, "utf8"));
    points.push({ id: tp, pixel: data.pixel, camera_view: data.camera_view });
    console.log(`${tp}: pixel=${data.pixel}, camera_view=${data.camera_view}`);
  }
}
if (points.length === 0) { console.error("No step08 found"); process.exit(1); }

// Load modules
const { RobotGatewayClient } = await import("../src/adapters/robotGatewayClient.js");
const { EyeInHandCameraController } = await import("../src/adapters/eyeInHandCameraController.js");
const { projectEyeInHandVlmPixel } = await import("../src/agent/calibratedVisionTarget.js");
const { executeProbeSignalSearch } = await import("../src/agent/probeSignalSearch.js");
const { PROBE_SIGNAL_SEARCH_CONFIG } = await import("../src/domain/probeSignalSearchConfig.js");
const { evaluateMg400PoseReachability } = await import("../src/domain/mg400Reachability.js");
const { createEquipmentController } = await import("../src/adapters/equipmentControllerFactory.js");

const arm = new RobotGatewayClient();
const cam = new EyeInHandCameraController();
const eq = createEquipmentController();

// Build mock cameraExecution from captures
const captureDir = workspace;
const captureFiles = ["close_20260804_125119_298749_1.jpg", "close_20260804_125120_939397_2.jpg", "close_20260804_125122_581254_3.jpg"]
  .map(f => `${captureDir}/${f}`)
  .filter(f => existsSync(f));

if (captureFiles.length === 0) { console.error("No captures found"); process.exit(1); }

// Read close robot pose from first capture if available, else use known pose
const closePose = { x: 360.737183, y: -14.016606, z: -27.421679, r: 7.686619 };
const cameraExecution = {
  status: "COMPLETED",
  calibrationFile: "C:/Users/32825/Desktop/new_version_demo/calibration/camera_config.yaml",
  selectedImage: {
    path: captureFiles[0],
    robotPose: closePose
  },
  fixedR: 7.686619,
  validRobotXYZMin: [339.90, -31.02, -43.69],
  validRobotXYZMax: [383.39, 19.18, 50.10]
};

// Project each point
for (const pt of points) {
  console.log(`\n--- ${pt.id} ---`);
  if (!pt.pixel) { console.log("no pixel, skipping"); continue; }
  const projection = await projectEyeInHandVlmPixel({
    cameraExecution,
    cameraController: cam,
    pixel: pt.pixel
  });
  pt.basePoint = projection?.basePoint || null;
  console.log(`pixel=${pt.pixel} -> basePoint=`, pt.basePoint);
}

const valid = points.filter(p => p.basePoint);
if (valid.length === 0) { console.error("No valid basePoint"); process.exit(1); }

const pt = valid.find(p => p.id === "TP9") || valid[0];
const bp = pt.basePoint;
const fixedR = 7.686619;
const hoverOffsetMm = 5;
const descentMarginMm = 2;

const safeHoverZ = bp.z + hoverOffsetMm;
const minimumZ = bp.z - descentMarginMm;
const hoverPose = { x: bp.x, y: bp.y, z: safeHoverZ, r: fixedR };

const reachability = evaluateMg400PoseReachability(hoverPose, { allowAdjustment: true });
if (!reachability.reachable) {
  console.error("Unreachable:", reachability.message);
  process.exit(1);
}
const actualPose = reachability.pose;
console.log(`Hover: requested=`, hoverPose);
console.log(`Hover: actual=`, actualPose);

// Move to hover
console.log("\nMoving to hover...");
const hoverResult = await arm.execute({
  id: "test-hover",
  kind: "ARM_MOTION",
  command: "MOVE_TO_CALIBRATED_PROBE_HOVER",
  targetLocationId: pt.id,
  targetPose: actualPose,
  trajectory: { mode: "direct" }
});
console.log("Hover result:", hoverResult.status, hoverResult.message || "");

if (hoverResult.status !== "COMPLETED") {
  console.error("Hover move failed");
  process.exit(1);
}

// Probe descent - use ACTUAL pose XY, not basePoint XY
console.log("\nProbe descent...");
const probeConfig = {
  ...PROBE_SIGNAL_SEARCH_CONFIG,
  x: actualPose.x, y: actualPose.y, r: actualPose.r,
  startZ: actualPose.z, minimumZ: bp.z - descentMarginMm
};
const searchResult = await executeProbeSignalSearch({
  armController: arm,
  equipmentController: eq,
  config: probeConfig,
  onEvent: (e) => console.log(" ", e.type, JSON.stringify(e.payload).slice(0, 100))
});
console.log("Result:", searchResult.status, "signalDetected:", searchResult.signalDetected);
if (searchResult.finalPose) console.log("Final pose:", searchResult.finalPose);
