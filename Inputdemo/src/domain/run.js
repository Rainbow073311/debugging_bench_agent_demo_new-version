import { AgentState } from "./states.js";
import { createId } from "../utils/id.js";

export function createBenchRun(input) {
  return {
    runId: createId("run"),
    state: AgentState.IDLE,
    input,
    vlmAgentCase: null,
    modelOutput: null,
    plan: null,
    vlmObservation: null,
    ragEvidence: [],
    execution: {
      arm: [],
      equipment: [],
      camera: null
    },
    report: null,
    timeline: []
  };
}

export function transition(run, state, message) {
  run.state = state;
  run.timeline.push({
    at: new Date().toISOString(),
    state,
    message
  });
}
