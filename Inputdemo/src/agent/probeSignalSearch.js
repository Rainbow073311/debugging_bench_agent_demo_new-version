import { evaluateMg400PoseReachability } from "../domain/mg400Reachability.js";
import {
  PROBE_SIGNAL_SEARCH_CONFIG,
  probeSignalSearchStartPose
} from "../domain/probeSignalSearchConfig.js";

const sleepDefault = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function rounded(value) {
  return Number(Number(value).toFixed(6));
}
function requireFiniteVoltage(reading) {
  const value = Number(reading?.value);
  if (!Number.isFinite(value)) {
    throw new Error("DSOX1204G returned an invalid voltage; probe descent was stopped.");
  }
  return { ...reading, value };
}

function requireIdleStatus(result, action) {
  const mode = result?.robot?.mode;
  if (mode?.code !== 5) {
    throw new Error(
      `MG400 is not ENABLED_IDLE before ${action}: ${mode?.label || "UNKNOWN"} (${mode?.code ?? "?"}).`
    );
  }
  return result.robot;
}

function poseMatches(actual, expected, toleranceMm) {
  if (!actual) return false;
  return ["x", "y", "z", "r"].every(
    (key) => Math.abs(Number(actual[key]) - Number(expected[key])) <= toleranceMm
  );
}

function buildTargetPose(config, z) {
  const pose = {
    x: config.x,
    y: config.y,
    z: rounded(z),
    r: config.r
  };
  const reachability = evaluateMg400PoseReachability(pose, {
    allowAdjustment: false,
    enforceLowZStallGuard: false
  });
  if (!reachability.reachable) {
    throw new Error(`Probe descent target is not reachable: ${reachability.message}`);
  }
  return pose;
}

async function readVoltage(equipmentController) {
  if (typeof equipmentController?.readMeanVoltage !== "function") {
    throw new Error("Probe signal search requires the DSOX1204G fast MEAN-voltage reader.");
  }
  return requireFiniteVoltage(await equipmentController.readMeanVoltage());
}

async function confirmLatchedSignal({
  equipmentController,
  firstReading,
  config,
  sleep
}) {
  const readings = [firstReading];
  while (readings.length < config.confirmationSamples) {
    await sleep(config.settleMs);
    readings.push(await readVoltage(equipmentController));
  }
  return {
    readings,
    confirmed: readings.every((reading) => reading.value >= config.thresholdV)
  };
}

export async function executeProbeSignalSearch({
  armController,
  equipmentController,
  config = PROBE_SIGNAL_SEARCH_CONFIG,
  sleep = sleepDefault,
  onEvent = () => {}
}) {
  const startPose = probeSignalSearchStartPose(config);
  const status = await armController.runCommand("status");
  const robot = requireIdleStatus(status, "probe signal search");
  if (!poseMatches(robot.pose, startPose, config.poseToleranceMm)) {
    throw new Error(
      "MG400 is not at the fixed probe-search start pose; descent was not started."
    );
  }

  const samples = [];
  const movements = [];
  let reading = await readVoltage(equipmentController);
  samples.push({ index: 0, z: startPose.z, value: reading.value, unit: reading.unit });
  onEvent({ type: "probe.signal_sampled", payload: samples.at(-1) });

  if (reading.value >= config.thresholdV) {
    const confirmation = await confirmLatchedSignal({
      equipmentController,
      firstReading: reading,
      config,
      sleep
    });
    return {
      status: confirmation.confirmed ? "SIGNAL_FOUND" : "SIGNAL_LATCHED_UNSTABLE",
      signalDetected: true,
      signalConfirmed: confirmation.confirmed,
      stopReason: "threshold-reached-at-start",
      thresholdV: config.thresholdV,
      finalPose: startPose,
      samples,
      confirmationReadings: confirmation.readings,
      movements
    };
  }

  const maxSteps = Math.ceil((config.startZ - config.minimumZ) / config.stepMm);
  for (let index = 1; index <= maxSteps; index += 1) {
    const nextZ = Math.max(
      config.minimumZ,
      rounded(config.startZ - (index * config.stepMm))
    );
    const targetPose = buildTargetPose(config, nextZ);
    const motion = await armController.runCommand("command", {
      command: {
        name: "probeStep",
        pose: targetPose,
        speedL: config.speedL,
        accL: config.accL,
        cp: config.cp
      }
    });
    const movedRobot = requireIdleStatus(motion, `probe step ${index}`);
    if (!poseMatches(movedRobot.pose, targetPose, config.poseToleranceMm)) {
      throw new Error(`MG400 probe step ${index} did not settle at the requested pose.`);
    }
    movements.push({
      stepId: `probe-signal-descent-${index}`,
      status: "COMPLETED",
      targetPose,
      actualPose: movedRobot.pose,
      speedL: config.speedL,
      accL: config.accL,
      cp: config.cp,
      result: motion
    });
    onEvent({
      type: "robot.probe_descent_step_finished",
      payload: movements.at(-1)
    });

    await sleep(config.settleMs);
    reading = await readVoltage(equipmentController);
    samples.push({ index, z: targetPose.z, value: reading.value, unit: reading.unit });
    onEvent({ type: "probe.signal_sampled", payload: samples.at(-1) });

    if (reading.value >= config.thresholdV) {
      // The robot is already stopped by probeStep's Sync(). Latch the first
      // threshold crossing and never issue another downward command.
      const confirmation = await confirmLatchedSignal({
        equipmentController,
        firstReading: reading,
        config,
        sleep
      });
      return {
        status: confirmation.confirmed ? "SIGNAL_FOUND" : "SIGNAL_LATCHED_UNSTABLE",
        signalDetected: true,
        signalConfirmed: confirmation.confirmed,
        stopReason: "threshold-reached",
        thresholdV: config.thresholdV,
        finalPose: movedRobot.pose,
        samples,
        confirmationReadings: confirmation.readings,
        movements
      };
    }

    if (nextZ <= config.minimumZ) break;
  }

  return {
    status: "LIMIT_REACHED",
    signalDetected: false,
    signalConfirmed: false,
    stopReason: "minimum-z-reached",
    thresholdV: config.thresholdV,
    finalPose: movements.at(-1)?.actualPose || startPose,
    samples,
    confirmationReadings: [],
    movements
  };
}

export async function retractProbeToStart({
  armController,
  config = PROBE_SIGNAL_SEARCH_CONFIG
}) {
  const startPose = probeSignalSearchStartPose(config);
  const result = await armController.runCommand("command", {
    command: {
      name: "probeStep",
      pose: startPose,
      speedL: config.retractSpeedL,
      accL: config.accL,
      cp: config.cp
    }
  });
  const robot = requireIdleStatus(result, "probe retraction");
  if (!poseMatches(robot.pose, startPose, config.poseToleranceMm)) {
    throw new Error("MG400 did not settle at the fixed probe-search start pose after retraction.");
  }
  return {
    stepId: "probe-signal-retract-to-start",
    status: "COMPLETED",
    targetPose: startPose,
    actualPose: robot.pose,
    result
  };
}
