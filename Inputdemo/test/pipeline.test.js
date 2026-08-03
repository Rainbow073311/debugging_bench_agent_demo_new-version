import test from "node:test";
import assert from "node:assert/strict";
import net from "node:net";
import http from "node:http";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { BenchAgent } from "../src/agent/benchAgent.js";
import { parseUserCommand } from "../src/agent/inputCore.js";
import { AgentState } from "../src/domain/states.js";
import { SimulationArmController } from "../src/adapters/simulationArmController.js";
import { MockEquipmentController } from "../src/adapters/mockEquipmentController.js";
import { MockLargeModelClient } from "../src/adapters/mockLargeModelClient.js";
import { MockRagRepository } from "../src/adapters/mockRagRepository.js";
import { MockVlmClient } from "../src/adapters/mockVlmClient.js";
import { ReportGenerator } from "../src/adapters/reportGenerator.js";
import { VlmAgentCaseAdapter } from "../src/adapters/vlmAgentCaseAdapter.js";
import { RealVlmAgentRunner } from "../src/adapters/realVlmAgentRunner.js";
import { RealVlmAgentModelClient } from "../src/adapters/realVlmAgentClient.js";
import { VlmAgentServiceRunner } from "../src/adapters/vlmAgentServiceRunner.js";
import { RobotGatewayClient } from "../src/adapters/robotGatewayClient.js";
import { Mg400ArmController } from "../src/adapters/mg400ArmController.js";
import { Rto6EquipmentController } from "../src/adapters/rto6EquipmentController.js";
import { Dsox1204gEquipmentController } from "../src/adapters/dsox1204gEquipmentController.js";
import { createEquipmentController } from "../src/adapters/equipmentControllerFactory.js";
import { evaluateMg400PoseReachability } from "../src/domain/mg400Reachability.js";
import {
  VLM_COMPLETION_READY_POSE,
  VLM_COMPLETION_SAFE_TRAJECTORY,
  createVlmCompletionReadyStep,
  postReadyTargetExecutionEnabled
} from "../src/agent/vlmCompletionReadyMove.js";
import { createMg400SafeTrajectory } from "../src/domain/mg400SafeTrajectory.js";
import {
  executeProbeSignalSearch,
  retractProbeToStart
} from "../src/agent/probeSignalSearch.js";
import { PROBE_SIGNAL_SEARCH_CONFIG } from "../src/domain/probeSignalSearchConfig.js";

test("bench agent runs the mocked VLM-to-report flow with image and PDF model inputs", async () => {
  const armCalls = [];
  const mockArmController = {
    async execute(step) {
      return {
        stepId: step.id,
        status: "COMPLETED",
        motionCommand: step.command,
        targetLocationId: step.targetLocationId,
        targetPose: step.targetPose,
        executedPose: step.targetPose || { x: 304.667, y: -26.621, z: -128.252, r: 169.928 },
        tcpCommand: "MovJ(···)",
        controller: "mg400",
        robot: { mode: { code: 5, label: "ENABLED_IDLE" } },
        responses: ["0,{1}"],
        trajectory: null,
        durationMs: 10
      };
    },
    async runCommand(action, payload = {}) {
      armCalls.push({ action, payload });
      if (action === "status") {
        return {
          robot: {
            mode: { code: 5, label: "ENABLED_IDLE" },
            pose: {
              x: 307.660927,
              y: -12.981141,
              z: -128.251923,
              r: 169.927811
            }
          }
        };
      }
      if (action === "command") {
        const cmd = payload.command || {};
        if (cmd.name === "probeStep") {
          return {
            robot: {
              mode: { code: 5, label: "ENABLED_IDLE" },
              pose: { ...cmd.pose }
            }
          };
        }
      }
      return { ok: true };
    }
  };

  const agent = new BenchAgent({
    vlmClient: new MockVlmClient(),
    largeModelClient: new MockLargeModelClient(),
    ragRepository: new MockRagRepository(),
    armController: mockArmController,
    equipmentController: new MockEquipmentController(),
    reportGenerator: new ReportGenerator(),
    vlmAgentCaseAdapter: new VlmAgentCaseAdapter({
      outputDir: path.join(tmpdir(), "inputdemo-vlm-agent-cases-test")
    })
  });
  try {
    const tinyPng = "data:image/png;base64,iVBORw0KGgo=";
    const tinyPdf = "data:application/pdf;base64,JVBERi0xLjQK";
    const input = parseUserCommand({
      Instruction: "Measure 5V rail ripple",
      Case_ID: "case-001",
      Operator: "demo-user",
      Target_board_side: "front",
      Camera_image: {
        name: "PCBA_IMG.jpg",
        type: "image/jpeg",
        size: 2048,
        dataUrl: tinyPng
      },
      Camera_image_back: {
        name: "PCBA_BACK.jpg",
        type: "image/jpeg",
        size: 2048,
        dataUrl: tinyPng
      },
      Bit_image: [
        { name: "bit-map.png", type: "image/png", size: 1024, dataUrl: tinyPng },
        { name: "bit-map.pdf", type: "application/pdf", size: 4096, dataUrl: tinyPdf }
      ],
      Schematic_Diagram: [
        { name: "schematic.png", type: "image/png", size: 1024, dataUrl: tinyPng },
        { name: "schematic.pdf", type: "application/pdf", size: 8192, dataUrl: tinyPdf }
      ]
    });

    const run = await agent.run(input);

    assert.equal(run.input.instruction, "Measure 5V rail ripple");
    assert.equal(run.input.caseId, "case-001");
    assert.equal(run.input.operator, "demo-user");
    assert.equal(run.input.targetBoardSide, "front");
    assert.equal(run.input.cameraImage.name, "PCBA_IMG.jpg");
    assert.equal(run.input.cameraImageBack.name, "PCBA_BACK.jpg");
    assert.equal(run.state, AgentState.REPORTING);
    assert.equal(run.vlmObservation.model, "mock-vlm-v0");
    assert.equal("modelInputYaml" in run, false);
    assert.equal("modelInputYamlFile" in run, false);
    assert.match(run.vlmAgentCase.taskFile, /task\.yaml$/);
    assert.match(run.vlmAgentCase.caseDir, /case-001$/);
    assert.match(run.vlmAgentCase.taskYaml, /user_measurement_question:/);
    assert.match(run.vlmAgentCase.taskYaml, /front_board_photo: "PCBA_IMG.jpg"/);
    assert.match(run.vlmAgentCase.taskYaml, /back_board_photo: "PCBA_BACK.jpg"/);
    assert.match(run.vlmAgentCase.taskYaml, /target_board_side: "front"/);
    assert.doesNotMatch(run.vlmAgentCase.taskYaml, /global temporary override/);
    assert.match(run.vlmAgentCase.taskYaml, /PCB outline plus mounting\/tooling-hole registration/);
    assert.match(run.vlmAgentCase.taskYaml, /IC detection\/IC-anchor code is retained/);
    assert.match(run.vlmAgentCase.taskYaml, /assembly_drawing_pdf: "bit-map.pdf"/);
    assert.match(run.vlmAgentCase.taskYaml, /assembly_drawing: "bit-map.png"/);
    assert.match(run.vlmAgentCase.taskYaml, /schematic_pdf: "schematic.pdf"/);
    assert.match(run.vlmAgentCase.taskYaml, /schematic_image: "schematic.png"/);
    assert.equal(await readFile(run.vlmAgentCase.taskFile, "utf8"), run.vlmAgentCase.taskYaml);
    assert.equal(await readFile(path.join(run.vlmAgentCase.caseDir, "bit-map.pdf"), "utf8"), "%PDF-1.4\n");
    assert.equal(run.modelOutput.model, "mock-large-model-v0");
    assert.deepEqual(run.modelOutput.mg400Pose, {
      x: 245.6,
      y: -32.4,
      z: 78.2,
      r: 91.5
    });
    assert.deepEqual(run.plan.steps[0].targetPose, run.modelOutput.mg400Pose);
    assert.equal(run.plan.steps[0].reachabilityPrecheck.reachable, false);
    assert.equal(run.execution.arm[0].status, "COMPLETED");
    assert.deepEqual(run.execution.arm[0].executedPose, VLM_COMPLETION_READY_POSE);
    assert.equal(run.execution.arm.length, 1);
    // Probe signal search verified via runCommand calls
    assert.ok(armCalls.some((c) => c.action === "status"), "probe search checked arm status");
    assert.equal(run.execution.probeSearch.status, "SIGNAL_FOUND");
    assert.equal(run.execution.probeSearch.signalDetected, true);
    assert.equal(run.execution.probeSearch.stopReason, "threshold-reached-at-start");
    assert.ok(run.execution.probeSearch.samples.length > 0, "voltage samples collected");
    assert.equal(run.execution.probeRetract.status, "COMPLETED");
    assert.equal(run.ragEvidence.length, 0);
    assert.equal(run.vlmObservation.modelInputSummary.attachmentCount, 4);
    assert.equal(run.vlmObservation.modelInputSummary.bitImageCount, 2);
    assert.equal(run.vlmObservation.modelInputSummary.schematicCount, 2);
    assert.equal(run.execution.equipment.length, 1);
    assert.equal(run.execution.equipment[0].signal, "Current display: AMPL");
    assert.equal(run.execution.equipment[0].value, 3.3);
    assert.equal(run.report.measurements.length, 1);
    assert.match(run.timeline.at(-1).message, /probe descent finished.*display saved to Excel/);
  } finally {
    // mock arm controller – nothing to clean up
  }
});

test("VLM completion ready move preserves the operator-recorded pose exactly", () => {
  const step = createVlmCompletionReadyStep();

  assert.equal(step.command, "MOVE_TO_VLM_COMPLETION_READY_POSE");
  assert.deepEqual(step.targetPose, {
    x: 307.660927,
    y: -12.981141,
    z: -128.251923,
    r: 169.927811
  });
  assert.equal(step.reachabilityPrecheck.reachable, true);
  assert.equal(step.reachabilityPrecheck.adjusted, false);
  assert.deepEqual(step.reachabilityPrecheck.pose, step.targetPose);
  assert.equal(step.trajectory, null);
});

test("MG400 safe trajectory keeps a start that is already above the configured travel height", () => {
  const trajectory = createMg400SafeTrajectory(
    { x: 245.6, y: -32.4, z: 78.2, r: 91.5 },
    VLM_COMPLETION_READY_POSE,
    VLM_COMPLETION_SAFE_TRAJECTORY
  );

  assert.equal(trajectory.safeTravelZ, 60);
  assert.equal(trajectory.travelZ, 78.2);
  assert.deepEqual(
    trajectory.stages.map((stage) => ({
      name: stage.name,
      z: stage.targetPose.z,
      speed: stage.speed
    })),
    [
      { name: "traverse", z: 78.2, speed: 30 },
      { name: "descend", z: -128.251923, speed: 10 }
    ]
  );
});

test("MG400 safe trajectory lifts a lower start to the configured travel height", () => {
  const trajectory = createMg400SafeTrajectory(
    { x: 230.549521, y: 104.660345, z: 16.656279, r: 195.039337 },
    VLM_COMPLETION_READY_POSE,
    VLM_COMPLETION_SAFE_TRAJECTORY
  );

  assert.equal(trajectory.safeTravelZ, 60);
  assert.equal(trajectory.travelZ, 60);
  assert.deepEqual(
    trajectory.stages.map((stage) => ({
      name: stage.name,
      z: stage.targetPose.z,
      speed: stage.speed
    })),
    [
      { name: "lift", z: 60, speed: 30 },
      { name: "traverse", z: 60, speed: 30 },
      { name: "descend", z: -128.251923, speed: 10 }
    ]
  );
});

test("MG400 safe trajectory uses an inner-radius staging path across the Z=10 workspace neck", () => {
  const trajectory = createMg400SafeTrajectory(
    { x: 350, y: -20.9156, z: 50, r: 7.686619 },
    { x: 373.285553, y: -20.9156, z: -43.59, r: 7.686619 },
    {
      mode: "safe-lift-traverse-descend",
      safeTravelZ: 50,
      stagingRadius: 300,
      travelSpeed: 20,
      descentSpeed: 8
    }
  );

  assert.equal(trajectory.stagingRadius, 300);
  assert.deepEqual(
    trajectory.stages.map((stage) => stage.name),
    [
      "retract_to_staging",
      "traverse_staging",
      "descend_staging",
      "extend_from_staging"
    ]
  );
});

test("MG400 safe trajectory blocks a straight traverse through the inner workspace", () => {
  assert.throws(
    () => createMg400SafeTrajectory(
      { x: 300, y: 0, z: 100, r: 0 },
      { x: -259.808, y: 150, z: -100, r: 0 },
      VLM_COMPLETION_SAFE_TRAJECTORY
    ),
    /traverse is outside the sampled workspace/
  );
});

test("post-ready target execution stays paused unless explicitly enabled", () => {
  assert.equal(postReadyTargetExecutionEnabled({}), false);
  assert.equal(postReadyTargetExecutionEnabled({
    ENABLE_POST_READY_TARGET_EXECUTION: "false"
  }), false);
  assert.equal(postReadyTargetExecutionEnabled({
    ENABLE_POST_READY_TARGET_EXECUTION: "true"
  }), true);
});

function createProbeHarness({
  readings = [],
  mode = { code: 5, label: "ENABLED_IDLE" },
  startPose = {
    x: PROBE_SIGNAL_SEARCH_CONFIG.x,
    y: PROBE_SIGNAL_SEARCH_CONFIG.y,
    z: PROBE_SIGNAL_SEARCH_CONFIG.startZ,
    r: PROBE_SIGNAL_SEARCH_CONFIG.r
  }
} = {}) {
  let pose = { ...startPose };
  const commands = [];
  return {
    commands,
    armController: {
      async runCommand(action, payload = {}) {
        if (action === "status") {
          return { robot: { mode, pose: { ...pose } } };
        }
        commands.push(payload.command);
        pose = { ...payload.command.pose };
        return { robot: { mode, pose: { ...pose } } };
      }
    },
    equipmentController: {
      async readMeanVoltage() {
        const next = readings.shift();
        if (next instanceof Error) throw next;
        return { value: next, unit: "V", source: "CHANNEL2" };
      }
    }
  };
}

test("probe search does not move when the start pose is already at 3 V", async () => {
  const harness = createProbeHarness({ readings: [3.1, 3.2, 3.3] });
  const result = await executeProbeSignalSearch({
    ...harness,
    config: { ...PROBE_SIGNAL_SEARCH_CONFIG, settleMs: 0 },
    sleep: async () => {}
  });
  assert.equal(result.status, "SIGNAL_FOUND");
  assert.equal(harness.commands.length, 0);
});

test("probe search stops after the exact step that first reaches 3 V", async () => {
  const harness = createProbeHarness({ readings: [1.0, 2.0, 3.1, 3.2, 3.3] });
  const result = await executeProbeSignalSearch({
    ...harness,
    config: {
      ...PROBE_SIGNAL_SEARCH_CONFIG,
      minimumZ: PROBE_SIGNAL_SEARCH_CONFIG.startZ - 0.5,
      settleMs: 0
    },
    sleep: async () => {}
  });
  assert.equal(result.status, "SIGNAL_FOUND");
  assert.equal(harness.commands.length, 2);
  assert.equal(harness.commands.at(-1).pose.z, -128.451923);
});

test("probe search never resumes descent after an unstable threshold latch", async () => {
  const harness = createProbeHarness({ readings: [1.0, 3.1, 2.9, 3.2] });
  const result = await executeProbeSignalSearch({
    ...harness,
    config: {
      ...PROBE_SIGNAL_SEARCH_CONFIG,
      minimumZ: PROBE_SIGNAL_SEARCH_CONFIG.startZ - 0.5,
      settleMs: 0
    },
    sleep: async () => {}
  });
  assert.equal(result.status, "SIGNAL_LATCHED_UNSTABLE");
  assert.equal(result.signalDetected, true);
  assert.equal(harness.commands.length, 1);
});

test("probe search stops at its fixed Z lower limit", async () => {
  const harness = createProbeHarness({ readings: [1.0, 1.0, 1.0] });
  const minimumZ = Number((PROBE_SIGNAL_SEARCH_CONFIG.startZ - 0.2).toFixed(6));
  const result = await executeProbeSignalSearch({
    ...harness,
    config: {
      ...PROBE_SIGNAL_SEARCH_CONFIG,
      minimumZ,
      settleMs: 0
    },
    sleep: async () => {}
  });
  assert.equal(result.status, "LIMIT_REACHED");
  assert.equal(harness.commands.length, 2);
  assert.equal(harness.commands.at(-1).pose.z, minimumZ);
});

test("probe search does not move on DSOX communication failure or non-idle MG400", async () => {
  const communicationFailure = createProbeHarness({
    readings: [new Error("scope offline")]
  });
  await assert.rejects(
    executeProbeSignalSearch({
      ...communicationFailure,
      sleep: async () => {}
    }),
    /scope offline/
  );
  assert.equal(communicationFailure.commands.length, 0);

  const nonIdle = createProbeHarness({
    mode: { code: 7, label: "RUNNING" },
    readings: [3.5]
  });
  await assert.rejects(
    executeProbeSignalSearch({ ...nonIdle, sleep: async () => {} }),
    /not ENABLED_IDLE/
  );
  assert.equal(nonIdle.commands.length, 0);
});

test("probe retraction uses one synchronized probeStep target at the fixed start pose", async () => {
  const harness = createProbeHarness();
  const result = await retractProbeToStart({
    armController: harness.armController
  });
  assert.equal(result.status, "COMPLETED");
  assert.equal(harness.commands.length, 1);
  assert.equal(harness.commands[0].name, "probeStep");
  assert.deepEqual(harness.commands[0].pose, {
    x: PROBE_SIGNAL_SEARCH_CONFIG.x,
    y: PROBE_SIGNAL_SEARCH_CONFIG.y,
    z: PROBE_SIGNAL_SEARCH_CONFIG.startZ,
    r: PROBE_SIGNAL_SEARCH_CONFIG.r
  });
});

test("RTO6 controller records the MEAN display without model-generated data", async () => {
  const testDir = path.join(tmpdir(), `rto6-controller-${Date.now()}`);
  const configPath = path.join(testDir, "rto6.json");
  await mkdir(testDir, { recursive: true });
  await writeFile(configPath, JSON.stringify({
    host: "192.168.2.3",
    port: 5025,
    measurementSlot: 1,
    captureMode: "set_mean_then_capture",
    desiredMeasurement: "MEAN",
    outputDir: testDir
  }));
  let workbookInput = null;
  const controller = new Rto6EquipmentController({
    configPath,
    async bridgeRunner({ config, screenshotPath }) {
      assert.equal(config.measurementSlot, 1);
      assert.equal(config.captureMode, "set_mean_then_capture");
      assert.equal(config.desiredMeasurement, "MEAN");
      return {
        instrument: "Rohde&Schwarz,RTO6,TEST,5.30.1.0",
        host: config.host,
        port: config.port,
        measurementSlot: 1,
        measurement: "MEAN",
        source: "C1W1",
        value: 3.287654,
        unit: "V",
        screenshotPath,
        capturedAt: "2026-07-27T18:00:00+0800",
        durationMs: 1200
      };
    },
    async excelRunner(input) {
      workbookInput = input;
      return {
        outputPath: input.outputPath,
        previewPath: input.outputPath.replace(/\.xlsx$/i, ".preview.png")
      };
    }
  });

  const result = await controller.captureCurrentDisplayReport({
    runId: "run-test",
    caseId: "case-test",
    targetPoints: ["TP9"],
    robotPose: VLM_COMPLETION_READY_POSE
  });

  assert.equal(result.value, 3.287654);
  assert.equal(result.unit, "V");
  assert.equal(result.measurement, "MEAN");
  assert.equal(result.source, "C1W1");
  assert.equal(result.pass, null);
  assert.equal(workbookInput.value, 3.287654);
  assert.equal(workbookInput.measurement, "MEAN");
  assert.deepEqual(workbookInput.robotPose, VLM_COMPLETION_READY_POSE);
  assert.match(result.workbookPath, /\.xlsx$/);
});

test("DSOX1204G controller records the MEAN display through its parallel adapter", async () => {
  const testDir = path.join(tmpdir(), `dsox1204g-controller-${Date.now()}`);
  const configPath = path.join(testDir, "dsox1204g.json");
  await mkdir(testDir, { recursive: true });
  await writeFile(configPath, JSON.stringify({
    host: "192.168.2.3",
    transport: "vxi11",
    measurementSlot: 1,
    captureMode: "set_mean_then_capture",
    desiredMeasurement: "MEAN",
    meanInterval: "DISPLAY",
    outputDir: testDir
  }));
  let workbookInput = null;
  const controller = new Dsox1204gEquipmentController({
    configPath,
    async bridgeRunner({ config, screenshotPath }) {
      assert.equal(config.transport, "vxi11");
      assert.equal(config.desiredMeasurement, "MEAN");
      return {
        instrument: "KEYSIGHT TECHNOLOGIES,DSOX1204G,CNTEST,02.12",
        host: config.host,
        visaAddress: `TCPIP::${config.host}::inst0::INSTR`,
        measurementSlot: 1,
        measurement: "MEAN",
        source: "CHANNEL2",
        value: 3.287654,
        unit: "V",
        screenshotPath,
        capturedAt: "2026-07-29T10:30:00+0800",
        durationMs: 900
      };
    },
    async excelRunner(input) {
      workbookInput = input;
      return {
        outputPath: input.outputPath,
        previewPath: input.outputPath.replace(/\.xlsx$/i, ".preview.png")
      };
    }
  });

  const result = await controller.captureCurrentDisplayReport({
    runId: "run-dsox-test",
    caseId: "case-dsox-test",
    targetPoints: ["TP9"],
    robotPose: VLM_COMPLETION_READY_POSE
  });

  assert.equal(result.instrument, "DSOX1204G");
  assert.equal(result.stepId, "dsox1204g-current-display-capture");
  assert.equal(result.measurement, "MEAN");
  assert.equal(result.source, "CHANNEL2");
  assert.equal(result.value, 3.287654);
  assert.match(result.screenshotPath, /DSOX1204G_MEAN_.*\.png$/);
  assert.match(workbookInput.scpiAddress, /TCPIP::192\.168\.2\.3::inst0::INSTR/);
});

test("DSOX1204G controller exposes a lightweight CHANNEL2 MEAN reader", async () => {
  const testDir = path.join(tmpdir(), `dsox1204g-fast-reader-${Date.now()}`);
  const configPath = path.join(testDir, "dsox1204g.json");
  await mkdir(testDir, { recursive: true });
  await writeFile(configPath, JSON.stringify({
    host: "192.168.2.3",
    transport: "vxi11",
    probeMeasurementSource: "CHANNEL2",
    meanInterval: "DISPLAY",
    outputDir: testDir
  }));
  const controller = new Dsox1204gEquipmentController({
    configPath,
    async measurementRunner({ config }) {
      assert.equal(config.probeMeasurementSource, "CHANNEL2");
      return {
        source: "CHANNEL2",
        value: "3.125",
        unit: "V",
        durationMs: 75
      };
    }
  });

  assert.deepEqual(await controller.readMeanVoltage(), {
    instrument: "DSOX1204G",
    measurement: "MEAN",
    source: "CHANNEL2",
    value: 3.125,
    unit: "V",
    durationMs: 75
  });
});

test("oscilloscope selector preserves RTO6 default and supports DSOX1204G", () => {
  assert.ok(createEquipmentController({ env: {} }) instanceof Rto6EquipmentController);
  assert.ok(
    createEquipmentController({
      env: { OSCILLOSCOPE_DRIVER: "dsox1204g" }
    }) instanceof Dsox1204gEquipmentController
  );
  assert.throws(
    () => createEquipmentController({ env: { OSCILLOSCOPE_DRIVER: "unknown" } }),
    /Unsupported OSCILLOSCOPE_DRIVER/
  );
});

test("bench agent stops later hardware actions when the VLM completion ready move fails", async () => {
  const executedSteps = [];
  const agent = new BenchAgent({
    vlmClient: {
      async analyzeBench() {
        return {
          recommendedMeasurements: [{
            locationId: "TP9",
            instrument: "oscilloscope",
            signal: "3V3"
          }]
        };
      }
    },
    largeModelClient: {
      async generateMg400Pose() {
        return { mg400Pose: { x: 300, y: 0, z: 0, r: 0 } };
      }
    },
    ragRepository: null,
    armController: {
      async execute(step) {
        executedSteps.push(step);
        return {
          stepId: step.id,
          status: "BLOCKED",
          message: "ready position unavailable"
        };
      }
    },
    equipmentController: {
      async measure() {
        throw new Error("measurement must not run");
      }
    },
    reportGenerator: {
      create({ measurements }) {
        return { measurements };
      }
    }
  });

  const run = await agent.run({ command: "measure TP9" });

  assert.equal(executedSteps.length, 1);
  assert.equal(executedSteps[0].id, "step-vlm-complete-ready-position");
  assert.equal(run.execution.arm.length, 1);
  assert.equal(run.execution.equipment.length, 0);
  assert.equal(run.state, AgentState.REPORTING);
  assert.match(run.timeline.at(-1).message, /later hardware steps were stopped/);
});

test("board-side input preserves front, back, and auto routing", () => {
  const base = { Instruction: "measure TP9" };

  assert.equal(parseUserCommand({ ...base, Target_board_side: "front" }).targetBoardSide, "front");
  assert.equal(parseUserCommand({ ...base, target_board_side: "back" }).targetBoardSide, "back");
  assert.equal(parseUserCommand({ ...base, targetBoardSide: "auto" }).targetBoardSide, "auto");
  assert.equal(parseUserCommand({ ...base, Target_board_side: "top" }).targetBoardSide, "front");
  assert.equal(parseUserCommand({ ...base, Target_board_side: "bottom" }).targetBoardSide, "back");
  assert.equal(parseUserCommand({ ...base, Target_board_side: "invalid" }).targetBoardSide, "auto");
});

test("MG400 reachability guard adjusts workspace envelope violations", () => {
  const result = evaluateMg400PoseReachability({
    x: 500,
    y: 0,
    z: 100,
    r: 0
  });

  assert.equal(result.reachable, true);
  assert.equal(result.adjusted, true);
  assert.ok(result.pose.x < 500);
  assert.match(result.message, /Adjusted MG400 pose/);
});

test("MG400 enable command only requires the dashboard TCP port", async () => {
  const dashboardRequests = [];
  const dashboardServer = net.createServer((socket) => {
    let buffer = "";
    socket.setEncoding("utf8");
    socket.on("data", (chunk) => {
      buffer += chunk;
      while (buffer.includes("\n")) {
        const newline = buffer.indexOf("\n");
        const command = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        dashboardRequests.push(command);
        if (command === "RobotMode()") {
          socket.write("0,{5}\n");
        } else if (command === "GetPose()") {
          socket.write("0,{100,0,80,0}\n");
        } else {
          socket.write("0,{}\n");
        }
      }
    });
  });
  await new Promise((resolve) => dashboardServer.listen(0, "127.0.0.1", resolve));

  const closedMotionServer = net.createServer();
  await new Promise((resolve) => closedMotionServer.listen(0, "127.0.0.1", resolve));
  const closedMotionPort = closedMotionServer.address().port;
  await new Promise((resolve) => closedMotionServer.close(resolve));

  try {
    const result = await Mg400ArmController.runCommand("command", {
      config: {
        mode: "mg400",
        ip: "127.0.0.1",
        dashboardPort: dashboardServer.address().port,
        motionPort: closedMotionPort,
        timeoutMs: 500,
        autoEnable: true,
        load: 0.5
      },
      command: { name: "enable" }
    });

    assert.equal(result.ok, true);
    assert.equal(result.action, "enable");
    assert.equal(result.result.command, "EnableRobot()");
    assert.deepEqual(dashboardRequests, ["EnableRobot()", "RobotMode()", "GetPose()"]);
  } finally {
    await new Promise((resolve) => dashboardServer.close(resolve));
  }
});

test("real VLM runner prechecks Debugging-agent-v2 .env before launching", async () => {
  const root = path.join(tmpdir(), "inputdemo-vlm-precheck-missing-env");
  const agentDir = path.join(root, "Debugging-agent-v2");
  await mkdir(path.join(agentDir, "agent"), { recursive: true });
  await writeFile(path.join(agentDir, "agent", "cli.py"), "");
  await writeFile(path.join(agentDir, ".env"), "VLM_BASE_URL=https://example.test/v1\nVLM_MODEL=test-model\n");

  const runner = new RealVlmAgentRunner({
    agentDir,
    runsRoot: path.join(root, "runs"),
    connectivityPrecheck: false
  });

  await assert.rejects(
    () => runner.runOnce({
      input: { caseId: "case-001", command: "measure" },
      vlmAgentCase: { taskFile: path.join(root, "task.yaml"), runId: "run-001" }
    }),
    (error) => {
      assert.match(error.message, /Missing VLM_API_KEY/);
      assert.equal(error.details.category, "vlm-env");
      assert.equal(error.details.envFile, path.join(agentDir, ".env"));
      return true;
    }
  );
});

test("real VLM runner reports incomplete case-001 run directory when summary is missing", async () => {
  const root = path.join(tmpdir(), "inputdemo-vlm-summary-missing");
  const agentDir = path.join(root, "Debugging-agent-v2");
  const runsRoot = path.join(root, "vlm-agent-runs");
  const latestRunDir = "20260519-132412-run_mpc6u7gu_yya8q992";
  await mkdir(path.join(agentDir, "agent"), { recursive: true });
  await writeFile(path.join(agentDir, "agent", "cli.py"), "");
  await writeFile(
    path.join(agentDir, ".env"),
    "VLM_BASE_URL=https://example.test/v1\nVLM_API_KEY=test-key\nVLM_MODEL=test-model\n"
  );
  await mkdir(path.join(runsRoot, "case-001", "runs", latestRunDir), { recursive: true });
  await writeFile(path.join(runsRoot, "case-001", "runs", latestRunDir, "messages.init.jsonl"), "{}\n");

  class MissingSummaryRunner extends RealVlmAgentRunner {
    spawnAgent() {
      return Promise.resolve({
        latestRunDir,
        stdout: `[bold]run_dir[/bold] = ${path.join(runsRoot, "case-001", "runs", latestRunDir)}\n[bold]model[/bold] = test-model\n`,
        stderr: ""
      });
    }
  }

  const runner = new MissingSummaryRunner({
    agentDir,
    runsRoot,
    connectivityPrecheck: false
  });

  await assert.rejects(
    () => runner.runOnce({
      input: { caseId: "case-001", command: "measure" },
      vlmAgentCase: { taskFile: path.join(root, "task.yaml"), runId: "run-001" }
    }),
    (error) => {
      assert.match(error.message, /summary\.json was not readable/);
      assert.equal(error.details.category, "vlm-summary-missing");
      assert.match(error.details.runDir, /20260519-132412-run_mpc6u7gu_yya8q992$/);
      assert.match(error.details.action, /Inspect the real VLM run directory/);
      return true;
    }
  );
});

test("real VLM runner rebuilds the agent process after retryable network failures", async () => {
  class FlakyNetworkRunner extends RealVlmAgentRunner {
    constructor() {
      super({
        agentDir: "unused",
        connectivityPrecheck: false,
        networkRetries: 2,
        networkRetryBaseDelayMs: 0
      });
      this.calls = 0;
    }

    spawnAgent() {
      this.calls += 1;
      if (this.calls === 1) {
        const error = new Error("Server disconnected without sending a response.");
        error.details = {
          category: "vlm-api",
          stderr: "httpx.RemoteProtocolError: Server disconnected without sending a response."
        };
        return Promise.reject(error);
      }
      return Promise.resolve({
        latestRunDir: "retry-succeeded",
        stdout: "[bold]run_dir[/bold] = retry-succeeded\n",
        stderr: ""
      });
    }
  }

  const runner = new FlakyNetworkRunner();
  const result = await runner.spawnAgentWithNetworkRetries([], { workspace: "workspace" });

  assert.equal(runner.calls, 2);
  assert.equal(result.latestRunDir, "retry-succeeded");
  assert.equal(result.attempts.length, 1);
  assert.equal(result.attempts[0].category, "vlm-api");
});

test("VLM agent service runner calls the Python service protocol and reads summary output", async () => {
  const root = path.join(tmpdir(), "inputdemo-vlm-service-runner");
  const summaryPath = path.join(root, "runs", "summary.json");
  await mkdir(path.dirname(summaryPath), { recursive: true });
  await writeFile(summaryPath, JSON.stringify({
    final_answer: {
      tp_id: "TP12",
      pixel: [100.4, 205.6],
      confidence: 0.88
    },
    stopped_reason: "finish-tool-called"
  }), "utf8");

  const requests = [];
  const server = http.createServer((request, response) => {
    requests.push({ method: request.method, url: request.url });
    if (request.method === "POST" && request.url === "/v1/runs") {
      let body = "";
      request.setEncoding("utf8");
      request.on("data", (chunk) => {
        body += chunk;
      });
      request.on("end", () => {
        const payload = JSON.parse(body);
        response.writeHead(202, { "Content-Type": "application/json" });
        response.end(JSON.stringify({
          run_id: payload.run_id,
          status: "queued",
          event_count: 1
        }));
      });
      return;
    }
    if (request.method === "GET" && request.url === "/v1/runs/run-service-001") {
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(JSON.stringify({
        run_id: "run-service-001",
        status: "succeeded",
        run_dir: path.join(root, "runs"),
        summary_path: summaryPath,
        final_answer: {
          tp_id: "TP12",
          pixel: [100.4, 205.6],
          confidence: 0.88
        },
        stopped_reason: "finish-tool-called"
      }));
      return;
    }
    response.writeHead(404, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ detail: "not found" }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();

  try {
    const runner = new VlmAgentServiceRunner({
      baseUrl: `http://127.0.0.1:${port}`,
      runsRoot: path.join(root, "vlm-agent-runs"),
      pollIntervalMs: 1,
      timeoutMs: 1000
    });
    const result = await runner.runOnce({
      input: { caseId: "case-001", command: "measure" },
      vlmAgentCase: {
        runId: "run-service-001",
        taskFile: path.join(root, "task.yaml")
      }
    });

    assert.equal(result.ok, true);
    assert.equal(result.serviceUrl, `http://127.0.0.1:${port}`);
    assert.deepEqual(result.pixel, [100, 206]);
    assert.equal(result.finalAnswer.tp_id, "TP12");
    assert.equal(result.summary.stopped_reason, "finish-tool-called");
    assert.deepEqual(requests.map((item) => `${item.method} ${item.url}`), [
      "POST /v1/runs",
      "GET /v1/runs/run-service-001"
    ]);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("VLM agent service runner reports service errors without CLI fallback", async () => {
  const runner = new VlmAgentServiceRunner({
    baseUrl: "http://127.0.0.1:1",
    timeoutMs: 100,
    pollIntervalMs: 1
  });
  const payload = {
    input: { caseId: "case-001", command: "measure" },
    vlmAgentCase: {
      runId: "run-service-fallback",
      taskFile: path.join(tmpdir(), "task.yaml")
    }
  };

  await assert.rejects(
    () => runner.run(payload),
    (error) => error.details?.category === "vlm-agent-service-unavailable"
  );
});

test("real VLM model client derives temporary MG400 pose from pixel when pose is missing", async () => {
  const client = new RealVlmAgentModelClient({
    runner: {
      run() {
        return Promise.resolve({
          model: "summary-replay",
          finalAnswer: {
            tp_id: "TP1",
            pixel: [237, 1243],
            confidence: 0.7
          },
          pixel: [237, 1243],
          summaryPath: "summary.json",
          runDir: "runs/latest",
          workspace: "workspace",
          precheck: { ok: true },
          attempts: []
        });
      }
    }
  });

  const result = await client.generateMg400Pose({
    input: {},
    vlmAgentCase: {}
  });

  assert.deepEqual(result.mg400Pose, { x: 237, y: 1243, z: 0, r: 0 });
  assert.deepEqual(result.pixel, { x: 237, y: 1243 });
});

test("robot gateway client sends robot execution over HTTP/TCP service boundary", async () => {
  const requests = [];
  const server = http.createServer((request, response) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => {
      body += chunk;
    });
    request.on("end", () => {
      requests.push({
        method: request.method,
        url: request.url,
        body: body ? JSON.parse(body) : null
      });
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(JSON.stringify({
        ok: true,
        stepId: requests.at(-1).body.step.id,
        status: "COMPLETED",
        controller: "robot-gateway-test",
        tcpCommand: "MovJ(1,2,3,4)"
      }));
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();

  try {
    const client = new RobotGatewayClient({
      baseUrl: `http://127.0.0.1:${port}`,
      timeoutMs: 1000
    });
    const result = await client.execute({
      id: "step-001",
      command: "move",
      targetPose: { x: 1, y: 2, z: 3, r: 4 }
    });

    assert.equal(result.status, "COMPLETED");
    assert.equal(result.controller, "robot-gateway-test");
    assert.deepEqual(requests.map((item) => `${item.method} ${item.url}`), [
      "POST /v1/robot/execute"
    ]);
    assert.equal(requests[0].body.step.id, "step-001");
    assert.match(requests[0].body.config.mode, /^(simulation|mg400)$/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});
