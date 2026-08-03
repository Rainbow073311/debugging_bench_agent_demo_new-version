import test from "node:test";
import assert from "node:assert/strict";
import { createBenchRun } from "../src/domain/run.js";
import { executeEyeInHandCaptureWorkflow } from "../src/agent/eyeInHandCaptureWorkflow.js";
import { readEyeInHandCaptureConfig } from "../src/domain/eyeInHandCaptureConfig.js";
import { BenchAgent } from "../src/agent/benchAgent.js";
import { MockVlmClient } from "../src/adapters/mockVlmClient.js";
import { MockLargeModelClient } from "../src/adapters/mockLargeModelClient.js";

function config(overrides = {}) {
  return {
    captureEnabled: true,
    motionEnabled: true,
    globalPose: { x: 300, y: 0, z: 120, r: 999 },
    fixedCloseXY: { x: 360.737183, y: -14.016606 },
    closeZ: -150,
    settleMs: 0,
    burstCount: 3,
    burstIntervalMs: 0,
    maxPoseAgeMs: 1000,
    minSharpness: 20,
    fixedR: 7.686619,
    trajectory: {
      mode: "safe-lift-traverse-descend",
      safeTravelZ: 150,
      travelSpeed: 20,
      descentSpeed: 8
    },
    ...overrides
  };
}

function status(pose = { x: 300, y: 0, z: 120, r: 27 }, mode = "ENABLED_IDLE") {
  return { robot: { mode: { code: mode === "ENABLED_IDLE" ? 5 : 4, label: mode }, pose } };
}

function armMock(statuses) {
  const calls = [];
  return {
    calls,
    async runCommand(action) {
      calls.push({ type: "command", action });
      return statuses.shift();
    },
    async execute(step) {
      calls.push({ type: "execute", step });
      return {
        status: "COMPLETED",
        stepId: step.id,
        executedPose: step.targetPose,
        trajectory: step.trajectory
      };
    }
  };
}

test("eye-in-hand height defaults use the requested Z=50 safe-travel profile", () => {
  const result = readEyeInHandCaptureConfig({
    EYE_IN_HAND_GLOBAL_POSE_JSON: '{"x":345.5,"y":-40.8,"z":999}'
  });
  assert.deepEqual(result.globalPose, { x: 345.5, y: -40.8, z: 50 });
  assert.deepEqual(result.fixedCloseXY, { x: 360.737183, y: -14.016606 });
  assert.equal(result.closeZ, -43.59);
  assert.equal(result.trajectory.safeTravelZ, 50);
  assert.equal(result.trajectory.stagingRadius, 300);
});

test("eye-in-hand workflow is inert when capture is disabled", async () => {
  const run = createBenchRun({ command: "test" });
  const result = await executeEyeInHandCaptureWorkflow({
    run,
    armController: null,
    cameraController: null,
    config: config({ captureEnabled: false })
  });
  assert.equal(result.status, "DISABLED");
  assert.deepEqual(run.execution.arm, []);
});

test("uncalibrated eye-in-hand workflow blocks before any robot command", async () => {
  const run = createBenchRun({ command: "test" });
  const arm = armMock([]);
  await assert.rejects(
    executeEyeInHandCaptureWorkflow({
      run,
      armController: arm,
      cameraController: { async health() { return { calibrationStatus: "uncalibrated", tablePlaneConfigured: false, cameraReady: false }; } },
      config: config()
    }),
    /not calibrated/
  );
  assert.deepEqual(arm.calls, []);
  assert.equal(run.execution.camera.status, "BLOCKED");
});

test("eye-in-hand workflow does not move when MG400 is not ENABLED_IDLE", async () => {
  const run = createBenchRun({ command: "test" });
  const arm = armMock([
    status({ x: 290, y: 0, z: 80, r: 27 }, "RUNNING")
  ]);
  await assert.rejects(
    executeEyeInHandCaptureWorkflow({
      run,
      armController: arm,
      cameraController: {
        async health() {
          return { calibrationStatus: "calibrated", tablePlaneConfigured: true, cameraReady: true };
        }
      },
      config: config()
    }),
    /must be ENABLED_IDLE/
  );
  assert.equal(arm.calls.filter((call) => call.type === "execute").length, 0);
  assert.equal(run.execution.camera.status, "BLOCKED");
});

test("eye-in-hand workflow blocks an overview pose outside the calibrated XYZ range", async () => {
  const run = createBenchRun({ command: "inspect PCB" });
  const arm = armMock([status({ x: 345.5, y: -40.8, z: 50, r: 7.686619 })]);
  const camera = {
    async health() {
      return {
        calibrationStatus: "calibrated",
        tablePlaneConfigured: true,
        cameraReady: true,
        fixedR: 7.686619,
        validRobotXYZMin: [305.49996, -75.800005, 50],
        validRobotXYZMax: [380.499967, -5.8, 140]
      };
    }
  };

  await assert.rejects(
    executeEyeInHandCaptureWorkflow({
      run,
      armController: arm,
      cameraController: camera,
      config: config({ globalPose: { x: 400, y: -40.8, z: 120 }, closeZ: 50 })
    }),
    /outside the calibrated eye-in-hand XYZ range/
  );
  assert.equal(arm.calls.filter((call) => call.type === "execute").length, 0);
});

test("eye-in-hand workflow moves high, localizes, hovers close, and injects the sharpest of three images", async () => {
  const run = createBenchRun({ command: "inspect PCB", cameraImage: null, visualCapture: null });
  const arm = armMock([
    status({ x: 290, y: 0, z: 80, r: 27 }),
    status({ x: 350, y: -20, z: 120, r: 7.686619 }),
    status({ x: 350, y: -20, z: 120, r: 7.686619 }),
    status({ x: 360.737183, y: -14.016606, z: -150, r: 7.686619 }),
    status({ x: 360.737183, y: -14.016606, z: -150, r: 7.686619 })
  ]);
  const selectedFile = {
    kind: "camera_image",
    name: "close_best.jpg",
    type: "image/jpeg",
    size: 123,
    dataUrl: "data:image/jpeg;base64,YQ=="
  };
  const camera = {
    async health() {
      return {
        calibrationStatus: "calibrated",
        tablePlaneConfigured: true,
        cameraReady: true,
        validRobotXYZMin: [339, -31, -151],
        validRobotXYZMax: [383, 19, 130]
      };
    },
    async capture({ robotPose }) {
      return { path: "global.jpg", robotPose, sharpness: 40 };
    },
    async coarseLocalize() {
      return {
        status: "UNIQUE",
        candidateCount: 1,
        basePoint: { x: 311, y: 3, z: -228 },
        recommendedEndXY: { x: 312, y: 4 }
      };
    },
    async captureBurst({ robotPose }) {
      const captures = [30, 55, 41].map((value, index) => ({
        path: `close_${index + 1}.jpg`, sharpness: value, robotPose
      }));
      return { captures, selected: captures[1], selectedFile };
    }
  };
  const events = [];
  await executeEyeInHandCaptureWorkflow({
    run,
    armController: arm,
    cameraController: camera,
    config: config({ globalPose: { x: 350, y: -20, z: 120 } }),
    sleep: async () => {},
    now: () => 100,
    onEvent: (type) => events.push(type)
  });

  const moves = arm.calls.filter((call) => call.type === "execute");
  assert.equal(moves.length, 2);
  assert.deepEqual(moves.map((call) => call.step.command), [
    "MOVE_TO_CAMERA_GLOBAL_POSE",
    "MOVE_TO_CAMERA_CLOSE_HOVER"
  ]);
  assert.equal(moves[0].step.targetPose.r, 7.686619, "calibrated fixed R is enforced");
  assert.deepEqual(moves[1].step.targetPose, {
    x: 360.737183, y: -14.016606, z: -150, r: 7.686619
  });
  assert.equal(moves[1].step.trajectory.mode, "safe-lift-traverse-descend");
  assert.equal(run.execution.camera.captures.length, 4);
  assert.equal(run.execution.camera.selectedImage.path, "close_2.jpg");
  assert.equal(run.input.cameraImage.name, "close_best.jpg");
  assert.equal(run.input.visualCapture.source, "eye-in-hand");
  assert.ok(events.includes("camera.selected_image_ready"));
});

test("global capture checks actual pose drift instead of rejecting a slow Basler exposure", async () => {
  const run = createBenchRun({ command: "inspect PCB" });
  const arm = armMock([
    status({ x: 290, y: 0, z: 80, r: 27 }),
    status({ x: 300, y: 0, z: 120, r: 7.686619 }),
    status({ x: 300.06, y: 0, z: 120, r: 7.686619 })
  ]);
  const camera = {
    async health() {
      return { calibrationStatus: "calibrated", tablePlaneConfigured: true, cameraReady: true };
    },
    async capture({ robotPose }) {
      return { path: "slow_global.jpg", robotPose, sharpness: 40 };
    }
  };

  await assert.rejects(
    executeEyeInHandCaptureWorkflow({
      run,
      armController: arm,
      cameraController: camera,
      config: config({ maxPoseAgeMs: 1 }),
      sleep: async () => {}
    }),
    /moved during camera exposure/
  );
  assert.equal(run.execution.camera.status, "BLOCKED");
});

test("ambiguous PCB localization blocks the close move", async () => {
  const run = createBenchRun({ command: "inspect PCB" });
  const arm = armMock([
    status({ x: 290, y: 0, z: 80, r: 27 }),
    status({ x: 300, y: 0, z: 120, r: 27 }),
    status({ x: 300, y: 0, z: 120, r: 27 })
  ]);
  const camera = {
    async health() { return { calibrationStatus: "calibrated", tablePlaneConfigured: true, cameraReady: true }; },
    async capture({ robotPose }) { return { path: "global.jpg", robotPose, sharpness: 40 }; },
    async coarseLocalize() { return { status: "AMBIGUOUS", candidateCount: 2 }; }
  };

  await assert.rejects(
    executeEyeInHandCaptureWorkflow({
      run,
      armController: arm,
      cameraController: camera,
      config: config(),
      sleep: async () => {},
      now: () => 100
    }),
    /not unique/
  );
  assert.equal(arm.calls.filter((call) => call.type === "execute").length, 1);
  assert.equal(run.execution.camera.status, "BLOCKED");
});

test("BenchAgent injects the selected close image before creating the VLM case", async () => {
  const input = { command: "inspect PCB", modelAttachments: [], cameraImage: null, visualCapture: null };
  const poses = [
    status({ x: 290, y: 0, z: 80, r: 27 }),
    status({ x: 300, y: 0, z: 120, r: 27 }),
    status({ x: 300, y: 0, z: 120, r: 27 }),
    status({ x: 312, y: 4, z: -150, r: 27 }),
    status({ x: 312, y: 4, z: -150, r: 27 })
  ];
  let executeCount = 0;
  const arm = {
    async runCommand() { return poses.shift(); },
    async execute(step) {
      executeCount += 1;
      if (executeCount === 3) return { status: "FAILED", error: "stop after VLM for test" };
      return { status: "COMPLETED", stepId: step.id, executedPose: step.targetPose };
    }
  };
  const selectedFile = {
    kind: "camera_image", name: "auto_close_best.jpg", type: "image/jpeg",
    size: 10, dataUrl: "data:image/jpeg;base64,YQ=="
  };
  const camera = {
    async health() {
      return { calibrationStatus: "calibrated", tablePlaneConfigured: true, cameraReady: true };
    },
    async capture({ robotPose }) {
      return { path: "global.jpg", robotPose, sharpness: 40 };
    },
    async coarseLocalize() {
      return { status: "UNIQUE", candidateCount: 1, recommendedEndXY: { x: 312, y: 4 } };
    },
    async captureBurst({ robotPose }) {
      const captures = [30, 55, 41].map((sharpness, index) => ({
        path: `close_${index + 1}.jpg`, sharpness, robotPose
      }));
      return { captures, selected: captures[1], selectedFile };
    }
  };
  let imageSeenByAdapter = null;
  const agent = new BenchAgent({
    vlmClient: new MockVlmClient(),
    largeModelClient: new MockLargeModelClient(),
    ragRepository: {},
    armController: arm,
    equipmentController: {},
    reportGenerator: { create() { return {}; } },
    vlmAgentCaseAdapter: {
      async adapt({ input: adaptedInput }) {
        imageSeenByAdapter = adaptedInput.cameraImage;
        return { taskFile: "mock-task.yaml" };
      }
    },
    eyeInHandCameraController: camera,
    eyeInHandCaptureConfig: config({ settleMs: 0 })
  });

  const run = await agent.run(input);
  assert.equal(imageSeenByAdapter.name, "auto_close_best.jpg");
  assert.equal(run.input.visualCapture.source, "eye-in-hand");
  assert.equal(run.execution.camera.status, "COMPLETED");
  assert.equal(executeCount, 2, "missing close-image pixel must not trigger probe motion");
  assert.match(run.error, /did not contain a pixel/);
});

test("BenchAgent projects the VLM pixel from the selected close image and pauses before probe motion", async () => {
  const input = { command: "inspect PCB", modelAttachments: [], cameraImage: null, visualCapture: null };
  const poses = [
    status({ x: 345.5, y: -40.8, z: 50, r: 7.686619 }),
    status({ x: 345.5, y: -40.8, z: 120, r: 7.686619 }),
    status({ x: 345.5, y: -40.8, z: 120, r: 7.686619 }),
    status({ x: 340, y: -35, z: 50, r: 7.686619 }),
    status({ x: 340, y: -35, z: 50, r: 7.686619 })
  ];
  let executeCount = 0;
  const arm = {
    async runCommand() { return poses.shift(); },
    async execute(step) {
      executeCount += 1;
      return { status: "COMPLETED", stepId: step.id, executedPose: step.targetPose };
    }
  };
  const selectedFile = {
    kind: "camera_image", name: "close_best.jpg", type: "image/jpeg",
    size: 10, dataUrl: "data:image/jpeg;base64,YQ=="
  };
  const camera = {
    async health() {
      return { calibrationStatus: "calibrated", tablePlaneConfigured: true, cameraReady: true, calibrationFile: "camera.yaml" };
    },
    async capture({ robotPose }) { return { path: "global.jpg", robotPose, sharpness: 40 }; },
    async coarseLocalize() {
      return { status: "UNIQUE", candidateCount: 1, recommendedEndXY: { x: 340, y: -35 } };
    },
    async captureBurst({ robotPose }) {
      const captures = [30, 55, 41].map((sharpness, index) => ({
        path: `close_${index + 1}.jpg`, sharpness, robotPose
      }));
      return { captures, selected: captures[1], selectedFile };
    },
    async pixelToBase({ pixel, robotPose }) {
      assert.deepEqual(pixel, { x: 1200, y: 900 });
      assert.equal(robotPose.z, 50);
      return { basePoint: { x: 321.25, y: -22.5, z: -149.48003 } };
    }
  };
  const agent = new BenchAgent({
    vlmClient: {
      async analyzeBench() {
        return {
          pixel: { x: 1200, y: 900 },
          locations: [{ id: "TP1", pixel: { x: 1200, y: 900 } }],
          recommendedMeasurements: [{ locationId: "TP1", instrument: "oscilloscope" }]
        };
      }
    },
    largeModelClient: {
      async generateMg400Pose() {
        return { pixel: { x: 1200, y: 900 }, mg400Pose: { x: 1200, y: 900, z: 0, r: 0 } };
      }
    },
    ragRepository: {}, armController: arm, equipmentController: {},
    reportGenerator: { create() { return {}; } },
    eyeInHandCameraController: camera,
    eyeInHandCaptureConfig: config({ globalPose: { x: 345.5, y: -40.8, z: 120 }, closeZ: 50 })
  });

  const run = await agent.run(input);
  assert.equal(executeCount, 2, "only overview and close-camera moves are allowed");
  assert.equal(run.modelOutput.mg400Pose, null);
  assert.deepEqual(run.modelOutput.calibratedBasePoint, { x: 321.25, y: -22.5, z: -149.48003 });
  assert.match(run.timeline.at(-1).message, /hover validation/);
});
