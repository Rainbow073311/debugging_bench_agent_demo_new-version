import "dotenv/config";
import { RobotGatewayClient } from "../src/adapters/robotGatewayClient.js";
import { Dsox1204gEquipmentController } from "../src/adapters/dsox1204gEquipmentController.js";
import {
  executeVlmCompletionReadyMove,
  vlmCompletionReadyMoveSucceeded
} from "../src/agent/vlmCompletionReadyMove.js";
import { executeProbeSignalSearch } from "../src/agent/probeSignalSearch.js";

const armController = new RobotGatewayClient();
const equipmentController = new Dsox1204gEquipmentController();
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
let search = null;
let report = null;
let primaryError = null;

function emit(type, payload = {}) {
  process.stdout.write(`${JSON.stringify({ type, at: new Date().toISOString(), ...payload })}\n`);
}

try {
  emit("demo.started");
  const readyMove = await executeVlmCompletionReadyMove(armController);
  emit("robot.ready_move_finished", {
    status: readyMove.result.status,
    pose: readyMove.result.robot?.pose || readyMove.result.executedPose
  });
  if (!vlmCompletionReadyMoveSucceeded(readyMove.result)) {
    throw new Error(
      readyMove.result.message
      || readyMove.result.error
      || "MG400 failed to reach the fixed probe-search start pose."
    );
  }

  search = await executeProbeSignalSearch({
    armController,
    equipmentController,
    onEvent(event) {
      if (event.type === "probe.signal_sampled") {
        emit(event.type, event.payload);
      }
    }
  });
  emit("probe.search_finished", {
    status: search.status,
    stopReason: search.stopReason,
    finalPose: search.finalPose,
    sampleCount: search.samples.length,
    lastSample: search.samples.at(-1),
    signalConfirmed: search.signalConfirmed
  });

  if (!search.signalDetected) {
    throw new Error("Probe search reached the fixed Z lower limit without detecting 3 V.");
  }

  report = await equipmentController.captureCurrentDisplayReport({
    runId: `manual-probe-demo-${stamp}`,
    caseId: "manual-vlm-completion-demo",
    targetPoints: ["VLM_DEMO_TARGET"],
    robotPose: search.finalPose
  });
  emit("equipment.report_finished", {
    value: report.value,
    unit: report.unit,
    screenshotPath: report.screenshotPath,
    workbookPath: report.workbookPath
  });
} catch (error) {
  primaryError = error;
  emit("demo.failed", {
    error: error.message,
    details: error.details || null
  });
}

if (primaryError) {
  process.exitCode = 1;
} else {
  emit("demo.completed", {
    searchStatus: search.status,
    workbookPath: report.workbookPath
  });
}
