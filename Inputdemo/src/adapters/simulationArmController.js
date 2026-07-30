import { createWriteStream, existsSync, readFileSync } from "node:fs";
import net from "node:net";
import { spawn } from "node:child_process";
import path from "node:path";
import { evaluateMg400PoseReachability } from "../domain/mg400Reachability.js";
import { createMg400SafeTrajectory } from "../domain/mg400SafeTrajectory.js";

function resolveSimulationDir() {
  const candidates = [
    path.resolve("..", "MG400stimulation", "MG400simulation-master"),
    path.resolve("MG400stimulation", "MG400simulation-master")
  ];
  return candidates.find((candidate) => existsSync(path.join(candidate, "simulate_slider.py"))) || candidates[0];
}

const simulationDir = resolveSimulationDir();
const simulationScript = path.join(simulationDir, "simulate_slider.py");
const simulationEnv = loadSimulationEnv();
const DEFAULT_HOST = simulationEnv.MG400_SIM_HOST || process.env.MG400_SIM_HOST || "127.0.0.1";
const DEFAULT_PORT = Number(simulationEnv.MG400_SIM_DASHBOARD_PORT || simulationEnv.MG400_SIM_PORT || process.env.MG400_SIM_DASHBOARD_PORT || process.env.MG400_SIM_PORT || 29999);
const DEFAULT_FEEDBACK_PORT = Number(simulationEnv.MG400_SIM_FEEDBACK_PORT || process.env.MG400_SIM_FEEDBACK_PORT || 30004);
const DEFAULT_TIMEOUT_MS = Number(simulationEnv.MG400_SIM_TIMEOUT_MS || process.env.MG400_SIM_TIMEOUT_MS || 10000);
const bundledPython = path.resolve(
  process.env.USERPROFILE || "",
  ".cache",
  "codex-runtimes",
  "codex-primary-runtime",
  "dependencies",
  "python",
  "python.exe"
);

let simulationProcess = null;

function parseDotenv(filePath) {
  if (!existsSync(filePath)) return {};
  return Object.fromEntries(
    readFileSync(filePath, "utf8")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#") && line.includes("="))
      .map((line) => {
        const index = line.indexOf("=");
        const key = line.slice(0, index).trim();
        const value = line.slice(index + 1).trim().replace(/^['"]|['"]$/g, "");
        return [key, value];
      })
  );
}

function loadSimulationEnv() {
  const envFile = process.env.MG400_SIM_ENV_FILE
    ? path.resolve(process.env.MG400_SIM_ENV_FILE)
    : path.resolve(simulationDir, "..", ".env");
  return parseDotenv(envFile);
}

function getPythonCommand() {
  const candidates = [
    process.env.MG400_SIM_PYTHON,
    simulationEnv.MG400_SIM_PYTHON,
    process.env.PYTHON,
    process.env.PYTHON_EXE,
    process.env.LOCALAPPDATA ? path.join(process.env.LOCALAPPDATA, "Programs", "Python", "Python312", "python.exe") : null,
    bundledPython
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (candidate === "python" || existsSync(candidate)) return candidate;
  }
  if (existsSync(bundledPython)) return bundledPython;
  return "python";
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isInteger(number) ? String(number) : String(Number(number.toFixed(6)));
}

function speedValue(config) {
  return Math.max(1, Math.min(100, Math.round(Number(config.speed || 30))));
}

function commandHost(config) {
  if (config.mode === "simulation") return config.host || config.simHost || DEFAULT_HOST;
  return config.ip && config.ip !== "192.168.2.6" ? config.ip : DEFAULT_HOST;
}

function commandPort(config) {
  if (config.mode === "simulation") return Number(config.port || config.simDashboardPort || DEFAULT_PORT);
  return Number(config.port || config.dashboardPort || DEFAULT_PORT);
}

function socketProbe(host, port, timeoutMs = 500) {
  return new Promise((resolve) => {
    const socket = net.createConnection({ host, port });
    const timer = setTimeout(() => {
      socket.destroy();
      resolve(false);
    }, timeoutMs);
    socket.on("connect", () => {
      clearTimeout(timer);
      socket.end();
      resolve(true);
    });
    socket.on("error", () => {
      clearTimeout(timer);
      resolve(false);
    });
  });
}

async function waitForDashboard(host, port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await socketProbe(host, port)) return true;
    await new Promise((resolve) => setTimeout(resolve, 300));
  }
  return false;
}

export async function ensureSimulationStarted(config = {}) {
  const host = commandHost(config);
  const port = commandPort(config);
  if (await socketProbe(host, port)) {
    return { started: false, host, port, message: "Simulation Dashboard is already reachable." };
  }

  if (!simulationProcess || simulationProcess.exitCode !== null) {
    const stdout = createWriteStream(path.join(simulationDir, "inputdemo-simulation.log"), { flags: "a" });
    const stderr = createWriteStream(path.join(simulationDir, "inputdemo-simulation.err.log"), { flags: "a" });
    simulationProcess = spawn(getPythonCommand(), [simulationScript], {
      cwd: simulationDir,
      env: {
        ...process.env,
        ...simulationEnv,
        MG400_SIM_HOST: host,
        MG400_SIM_DASHBOARD_PORT: String(port),
        MG400_SIM_FEEDBACK_PORT: String(config.simFeedbackPort || config.feedbackPort || DEFAULT_FEEDBACK_PORT),
        MG400_SIM_MODE: "simulation"
      },
      detached: false,
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: false
    });
    simulationProcess.on("error", (error) => {
      simulationProcess = null;
      stderr.write(`Failed to start simulation: ${error.message}\n`);
    });
    simulationProcess.stdout.pipe(stdout);
    simulationProcess.stderr.pipe(stderr);
  }

  const ready = await waitForDashboard(host, port, Number(config.timeoutMs || DEFAULT_TIMEOUT_MS));
  if (!ready) {
    throw new Error(`MuJoCo simulation did not open Dashboard at ${host}:${port}. Check MG400stimulation/MG400simulation-master/inputdemo-simulation.err.log.`);
  }

  return { started: true, host, port, pid: simulationProcess.pid, message: "MuJoCo simulation started." };
}

function sendDashboardCommand(command, config) {
  const host = commandHost(config);
  const port = commandPort(config);
  const timeoutMs = Number(config.timeoutMs || DEFAULT_TIMEOUT_MS);

  return new Promise((resolve, reject) => {
    const socket = net.createConnection({ host, port });
    let buffer = "";
    let settled = false;
    const timer = setTimeout(() => {
      socket.destroy();
      reject(new Error(`Simulation Dashboard timed out at ${host}:${port}`));
    }, timeoutMs);

    function finish(error, payload) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      socket.end();
      if (error) reject(error);
      else resolve(payload);
    }

    socket.setEncoding("utf8");
    socket.on("connect", () => socket.write(`${command}\n`));
    socket.on("data", (chunk) => {
      buffer += chunk;
      const newline = buffer.indexOf("\n");
      if (newline === -1) return;
      const response = buffer.slice(0, newline).trim();
      const ok = !response.startsWith("-1");
      if (!ok) {
        finish(new Error(`Simulation rejected ${command}: ${response}`));
        return;
      }
      finish(null, { command, response, errorId: 0, ok: true });
    });
    socket.on("error", (error) => {
      finish(new Error(`Cannot connect to simulation Dashboard at ${host}:${port}: ${error.message}`));
    });
  });
}

async function sendDashboardCommands(commands, config) {
  const responses = [];
  for (const command of commands) {
    responses.push(await sendDashboardCommand(command, config));
  }
  return responses;
}

function poseCommand(pose, commandName = "MovJ") {
  const moveName = commandName === "MovL" ? "MovL" : "MovJ";
  return `${moveName}(pose={${formatNumber(pose.x)},${formatNumber(pose.y)},${formatNumber(pose.z)},0,0,${formatNumber(pose.r || 0)}})`;
}

function blockedMotionResult(step, reachability) {
  return {
    stepId: step.id,
    status: "BLOCKED",
    motionCommand: step.command,
    targetLocationId: step.targetLocationId,
    targetPose: reachability.requestedPose,
    tcpCommand: null,
    controller: "simulation",
    reachability,
    fallbackPose: reachability.fallbackPose || null,
    fallbackAction: reachability.fallbackAction || "Request a recalibrated measurement target inside the MG400 workspace.",
    message: reachability.message
  };
}

function parsePose(response) {
  const match = String(response || "").match(/\{([^}]+)\}/);
  if (!match) return null;
  const values = match[1].split(",").map((value) => Number(value.trim()));
  if (values.length < 4 || values.some((value) => !Number.isFinite(value))) return null;
  return { x: values[0], y: values[1], z: values[2], r: values[5] ?? values[3] };
}

async function currentPose(config) {
  const result = await sendDashboardCommand("GetPose()", config);
  return { pose: parsePose(result.response), raw: result };
}

async function resolvePartialPose(config, pose = {}) {
  const current = await currentPose(config);
  if (!current.pose) throw new Error("Current simulation pose is unavailable; fill X/Y/Z/R explicitly.");
  return {
    x: pose.x ?? current.pose.x,
    y: pose.y ?? current.pose.y,
    z: pose.z ?? current.pose.z,
    r: pose.r ?? current.pose.r
  };
}

export class SimulationArmController {
  constructor(options = {}) {
    this.options = options;
  }

  async runCommand(action, payload = {}) {
    return SimulationArmController.runCommand(action, { ...payload, config: { ...this.options, ...payload.config } });
  }

  async execute(step) {
    const { readMg400Config } = await import("./mg400Config.js");
    const config = { ...(await readMg400Config()), ...this.options, mode: "simulation" };
    if (!step.targetPose) {
      await ensureSimulationStarted(config);
      const responses = await sendDashboardCommands(["GetPose()"], config);
      return {
        stepId: step.id,
        status: "COMPLETED",
        motionCommand: step.command,
        targetLocationId: step.targetLocationId,
        targetPose: null,
        tcpCommand: "GetPose()",
        controller: "simulation",
        simulation: { host: commandHost(config), port: commandPort(config), protocol: "dashboard" },
        responses
      };
    }

    const startedAt = Date.now();
    const reachability = evaluateMg400PoseReachability(step.targetPose);
    if (!reachability.reachable) {
      return blockedMotionResult(step, reachability);
    }
    await ensureSimulationStarted(config);
    const motionPose = reachability.pose;
    let trajectory = null;
    const commands = ["EnableRobot()", `SpeedFactor(${speedValue(config)})`];
    if (step.trajectory?.mode === "safe-lift-traverse-descend") {
      const current = await currentPose(config);
      if (!current.pose) throw new Error("Current simulation pose is unavailable for safe trajectory planning.");
      trajectory = createMg400SafeTrajectory(current.pose, motionPose, step.trajectory);
      for (const stage of trajectory.stages) {
        commands.push(poseCommand(stage.targetPose, "MovL"), "Sync()");
      }
    } else {
      commands.push(poseCommand(motionPose, config.motionCommand));
    }
    const responses = await sendDashboardCommands(commands, config);
    return {
      stepId: step.id,
      status: "COMPLETED",
      motionCommand: step.command,
      targetLocationId: step.targetLocationId,
      targetPose: step.targetPose,
      executedPose: motionPose,
      reachability,
      tcpCommand: trajectory
        ? poseCommand(trajectory.stages.at(-1)?.targetPose || motionPose, "MovL")
        : commands[commands.length - 1],
      controller: "simulation",
      simulation: { host: commandHost(config), port: commandPort(config), protocol: "dashboard" },
      responses,
      trajectory,
      durationMs: Date.now() - startedAt
    };
  }

  static async runCommand(action, payload = {}) {
    const config = payload.config || {};
    const startup = await ensureSimulationStarted(config);
    if (action === "test") {
      const pose = await currentPose(config);
      return { ok: true, action, startup, robot: { pose: pose.pose, poseRaw: pose.raw } };
    }
    if (action === "status") {
      const pose = await currentPose(config);
      const mode = await sendDashboardCommand("RobotMode()", config);
      return { ok: true, action, startup, robot: { mode, pose: pose.pose, poseRaw: pose.raw } };
    }
    if (action !== "command") {
      throw new Error(`Unsupported simulation action: ${action}`);
    }

    const command = payload.command || {};
    const name = command.name;
    if (name === "pose") return SimulationArmController.runCommand("status", payload);
    if (name === "clearError") return { ok: true, action: name, startup, result: await sendDashboardCommand("ClearError()", config) };
    if (name === "enable") return { ok: true, action: name, startup, result: await sendDashboardCommand("EnableRobot()", config) };
    if (name === "disable") return { ok: true, action: name, startup, result: await sendDashboardCommand("DisableRobot()", config) };
    if (name === "pause") return { ok: true, action: name, startup, result: await sendDashboardCommand("Stop()", config) };
    if (name === "continue") return { ok: true, action: name, startup, result: await sendDashboardCommand("EnableRobot()", config) };
    if (name === "reset") return { ok: true, action: name, startup, result: await sendDashboardCommand("MovJ(joint={0,0,0,0,0,0})", config) };
    if (name === "jog") return { ok: true, action: name, startup, result: await sendDashboardCommand(`MoveJog(${String(command.axis || "").toUpperCase()})`, config) };
    if (name === "jogStop") return { ok: true, action: name, startup, result: await sendDashboardCommand("MoveJog()", config) };
    if (name === "probeStep") {
      const pose = await resolvePartialPose(config, command.pose);
      const reachability = evaluateMg400PoseReachability(pose, { allowAdjustment: false, enforceLowZStallGuard: false });
      if (!reachability.reachable) {
        return {
          ok: false,
          action: name,
          startup,
          error: reachability.message,
          reachability,
          fallbackPose: reachability.fallbackPose || null,
          fallbackAction: reachability.fallbackAction || "Request a recalibrated measurement target inside the MG400 workspace."
        };
      }
      const motionPose = reachability.pose;
      const commands = [
        "EnableRobot()",
        `SpeedFactor(${command.speedL !== undefined ? Math.round(command.speedL * 10) : speedValue(config)})`,
        poseCommand(motionPose, "MovL"),
        "Sync()"
      ];
      const responses = await sendDashboardCommands(commands, config);
      const mode = await sendDashboardCommand("RobotMode()", config);
      const current = await currentPose(config);
      return {
        ok: true,
        action: name,
        startup,
        robot: { mode, pose: current.pose },
        resolvedPose: pose,
        executedPose: motionPose,
        reachability,
        command: commands[commands.length - 1],
        responses
      };
    }
    if (name === "move") {
      const pose = await resolvePartialPose(config, command.pose);
      const reachability = evaluateMg400PoseReachability(pose);
      if (!reachability.reachable) {
        return {
          ok: false,
          action: name,
          startup,
          error: reachability.message,
          reachability,
          fallbackPose: reachability.fallbackPose || null,
          fallbackAction: reachability.fallbackAction || "Request a recalibrated measurement target inside the MG400 workspace."
        };
      }
      const motionPose = reachability.pose;
      const commands = [
        "EnableRobot()",
        `SpeedFactor(${speedValue(config)})`,
        poseCommand(motionPose, command.motionCommand || config.motionCommand)
      ];
      const responses = await sendDashboardCommands(commands, config);
      return { ok: true, action: name, startup, resolvedPose: pose, executedPose: motionPose, reachability, command: commands[commands.length - 1], responses };
    }
    throw new Error(`Unsupported simulation command: ${name}`);
  }
}
