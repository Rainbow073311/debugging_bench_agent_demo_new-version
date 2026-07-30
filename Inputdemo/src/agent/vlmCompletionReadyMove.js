import { StepKind } from "../domain/states.js";
import { evaluateMg400PoseReachability } from "../domain/mg400Reachability.js";
import { probeSignalSearchStartPose } from "../domain/probeSignalSearchConfig.js";

export const VLM_COMPLETION_READY_POSE = Object.freeze(probeSignalSearchStartPose());

export const VLM_COMPLETION_SAFE_TRAJECTORY = Object.freeze({
  mode: "safe-lift-traverse-descend",
  safeTravelZ: 60,
  travelSpeed: 30,
  descentSpeed: 10
});

export function postReadyTargetExecutionEnabled(env = process.env) {
  return env.ENABLE_POST_READY_TARGET_EXECUTION === "true";
}

export function createVlmCompletionReadyStep() {
  return {
    id: "step-vlm-complete-ready-position",
    kind: StepKind.ARM_MOTION,
    command: "MOVE_TO_VLM_COMPLETION_READY_POSE",
    targetLocationId: "vlm-completion-ready-position",
    targetPose: { ...VLM_COMPLETION_READY_POSE },
    trajectory: null,
    reachabilityPrecheck: evaluateMg400PoseReachability(VLM_COMPLETION_READY_POSE)
  };
}

export async function executeVlmCompletionReadyMove(armController) {
  const step = createVlmCompletionReadyStep();
  try {
    return {
      step,
      result: await armController.execute(step)
    };
  } catch (error) {
    return {
      step,
      result: {
        stepId: step.id,
        status: "FAILED",
        motionCommand: step.command,
        targetLocationId: step.targetLocationId,
        targetPose: step.targetPose,
        tcpCommand: null,
        error: error.message,
        details: error.details || null
      }
    };
  }
}

export function vlmCompletionReadyMoveSucceeded(result) {
  return result?.status === "COMPLETED";
}
