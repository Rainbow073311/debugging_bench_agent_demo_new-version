import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { readMg400Config } from "./mg400Config.js";
import { SimulationArmController } from "./simulationArmController.js";
import { evaluateMg400PoseReachability } from "../domain/mg400Reachability.js";

const bridgePath = path.resolve("scripts", "mg400_bridge.py");

function getPythonCommand() {
  const homeDir = process.env.USERPROFILE
    || (process.env.HOMEDRIVE && process.env.HOMEPATH ? `${process.env.HOMEDRIVE}${process.env.HOMEPATH}` : null);
  const localAppData = process.env.LOCALAPPDATA || (homeDir ? path.join(homeDir, "AppData", "Local") : null);
  const candidates = [
    process.env.PYTHON,
    process.env.PYTHON_EXE,
    homeDir && path.join(homeDir, ".cache", "codex-runtimes", "codex-primary-runtime", "dependencies", "python", "python.exe"),
    localAppData && path.join(localAppData, "Programs", "Python", "Python312", "python.exe"),
    localAppData && path.join(localAppData, "Programs", "Python", "Python311", "python.exe"),
    "python"
  ].filter(Boolean);
  return candidates.find((candidate) => (
    path.isAbsolute(candidate) ? existsSync(candidate) : true
  )) || "python";
}

function runBridge(action, payload) {
  return new Promise((resolve, reject) => {
    const child = spawn(getPythonCommand(), [bridgePath, action], {
      cwd: process.cwd(),
      stdio: ["pipe", "pipe", "pipe"]
    });
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", reject);
    child.on("close", (code) => {
      let parsed = null;
      try {
        parsed = stdout.trim() ? JSON.parse(stdout) : null;
      } catch (error) {
        reject(new Error(`MG400 bridge returned invalid JSON: ${stdout || stderr}`));
        return;
      }

      if (code !== 0 || parsed?.ok === false) {
        const message = parsed?.error || stderr.trim() || `MG400 bridge failed with code ${code}`;
        const error = new Error(message);
        error.details = parsed || { stderr };
        reject(error);
        return;
      }

      resolve(parsed);
    });

    child.stdin.end(`${JSON.stringify(payload)}\n`);
  });
}

function blockedMotionResult(step, reachability, controller) {
  return {
    stepId: step.id,
    status: "BLOCKED",
    motionCommand: step.command,
    targetLocationId: step.targetLocationId,
    targetPose: reachability.requestedPose,
    tcpCommand: null,
    controller,
    reachability,
    fallbackPose: reachability.fallbackPose || null,
    fallbackAction: reachability.fallbackAction || "Request a recalibrated measurement target inside the MG400 workspace.",
    message: reachability.message
  };
}

export class Mg400ArmController {
  constructor({ config = null } = {}) {
    this.config = config;
    this.simulationController = new SimulationArmController(config || {});
  }

  async runCommand(action, payload = {}) {
    return Mg400ArmController.runCommand(action, { ...payload, config: this.config || payload.config });
  }

  async execute(step) {
    const config = this.config || await readMg400Config();

    if (config.mode === "simulation") {
      return this.simulationController.execute(step);
    }

    if (!step.targetPose) {
      return {
        stepId: step.id,
        status: "SKIPPED",
        motionCommand: step.command,
        targetLocationId: step.targetLocationId,
        targetPose: null,
        tcpCommand: null,
        controller: config.mode,
        message: "No target pose was provided for this step."
      };
    }

    const reachability = evaluateMg400PoseReachability(step.targetPose);
    if (!reachability.reachable) {
      return blockedMotionResult(step, reachability, "mg400");
    }

    const startedAt = Date.now();
    const result = await runBridge("execute", {
      config,
      pose: reachability.pose,
      trajectory: step.trajectory || null
    });

    return {
      stepId: step.id,
      status: "COMPLETED",
      motionCommand: step.command,
      targetLocationId: step.targetLocationId,
      targetPose: step.targetPose,
      executedPose: reachability.pose,
      reachability,
      tcpCommand: result.command,
      controller: "mg400",
      robot: result.robot,
      responses: result.responses,
      trajectory: result.trajectory || null,
      durationMs: Date.now() - startedAt
    };
  }

  static runCommand(action, payload = {}) {
    if (payload.config?.mode === "simulation") {
      return SimulationArmController.runCommand(action, payload);
    }
    let commandReachability = null;
    if (
      action === "command"
      && ["move", "probeStep"].includes(payload.command?.name)
    ) {
      const pose = payload.command.pose || {};
      const completePose = ["x", "y", "z", "r"].every((key) => pose[key] !== undefined && pose[key] !== null && pose[key] !== "");
      if (completePose) {
        const isProbeStep = payload.command.name === "probeStep";
        const reachability = evaluateMg400PoseReachability(pose, isProbeStep
          ? { allowAdjustment: false, enforceLowZStallGuard: false }
          : undefined);
        if (!reachability.reachable) {
          return Promise.resolve({
            ok: false,
            action: payload.command.name,
            error: reachability.message,
            reachability,
            fallbackPose: reachability.fallbackPose || null,
            fallbackAction: reachability.fallbackAction || "Request a recalibrated measurement target inside the MG400 workspace."
          });
        }
        commandReachability = reachability;
        payload = {
          ...payload,
          command: {
            ...payload.command,
            pose: reachability.pose
          }
        };
      }
    }
    return runBridge(action, payload).then((result) => (
      commandReachability
        ? { ...result, reachability: commandReachability, executedPose: commandReachability.pose }
        : result
    ));
  }
}
