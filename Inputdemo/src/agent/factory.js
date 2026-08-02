import { BenchAgent } from "./benchAgent.js";
import { MockRagRepository } from "../adapters/mockRagRepository.js";
import { RobotGatewayClient } from "../adapters/robotGatewayClient.js";
import { MockEquipmentController } from "../adapters/mockEquipmentController.js";
import { createEquipmentController } from "../adapters/equipmentControllerFactory.js";
import { ReportGenerator } from "../adapters/reportGenerator.js";
import { VlmAgentCaseAdapter } from "../adapters/vlmAgentCaseAdapter.js";
import { RealVlmAgentClient, RealVlmAgentModelClient } from "../adapters/realVlmAgentClient.js";
import { VlmAgentServiceRunner } from "../adapters/vlmAgentServiceRunner.js";
import { RemoteVlmAgentServiceRunner } from "../adapters/remoteVlmAgentServiceRunner.js";
import { EyeInHandCameraController } from "../adapters/eyeInHandCameraController.js";

/**
 * Decide which VLM runner to use based on environment:
 *
 *   VLM_AGENT_RUNNER=local-svc  → local FastAPI service (VlmAgentServiceRunner)
 *   VLM_AGENT_RUNNER=remote-svc → remote FastAPI service (RemoteVlmAgentServiceRunner)
 *
 *   (default / unset)
 *   ─ If VLM_AGENT_SERVICE_URL points to a non-localhost address
 *     → RemoteVlmAgentServiceRunner (remote server)
 *   ─ Else → VlmAgentServiceRunner (local dev service)
 */
export function createDefaultAgent({ vlmRunner } = {}) {
  const runnerMode = process.env.VLM_AGENT_RUNNER || "";

  let realVlmRunner = vlmRunner;

  if (!realVlmRunner) {
    if (runnerMode === "cli" || runnerMode === "local-svc") {
      // Force local service. CLI fallback is intentionally disabled.
      realVlmRunner = new VlmAgentServiceRunner();
    } else if (runnerMode === "remote-svc") {
      // Force remote service (no fallback)
      realVlmRunner = new RemoteVlmAgentServiceRunner();
    } else {
      // Auto-detect: remote if VLM_AGENT_SERVICE_URL is non-localhost
      const serviceUrl = (process.env.VLM_AGENT_SERVICE_URL || "").trim();
      if (serviceUrl && !isLocalhostUrl(serviceUrl)) {
        realVlmRunner = new RemoteVlmAgentServiceRunner();
      } else {
        realVlmRunner = new VlmAgentServiceRunner();
      }
    }
  }

  return new BenchAgent({
    vlmClient: new RealVlmAgentClient({ runner: realVlmRunner }),
    largeModelClient: new RealVlmAgentModelClient({ runner: realVlmRunner }),
    ragRepository: new MockRagRepository(),
    armController: new RobotGatewayClient(),
    equipmentController: createEquipmentController(),
    reportGenerator: new ReportGenerator(),
    vlmAgentCaseAdapter: new VlmAgentCaseAdapter(),
    eyeInHandCameraController: new EyeInHandCameraController(),
  });
}

function isLocalhostUrl(url) {
  try {
    const hostname = new URL(url).hostname.toLowerCase();
    return (
      hostname === "localhost" ||
      hostname === "127.0.0.1" ||
      hostname === "::1" ||
      hostname === "0.0.0.0"
    );
  } catch {
    return true; // treat unparsable as localhost for safety
  }
}
