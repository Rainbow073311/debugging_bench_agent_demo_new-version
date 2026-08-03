import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const bridgePath = fileURLToPath(new URL("../../scripts/eye_in_hand_camera.py", import.meta.url));

function pythonCommand() {
  const userDir = process.env.USERPROFILE;
  const localAppData = process.env.LOCALAPPDATA
    || (userDir ? path.join(userDir, "AppData", "Local") : null);
  const candidates = [
    process.env.EYE_IN_HAND_PYTHON,
    process.env.PYTHON,
    process.env.PYTHON_EXE,
    localAppData && path.join(localAppData, "Programs", "Python", "Python312", "python.exe"),
    localAppData && path.join(localAppData, "Programs", "Python", "Python311", "python.exe"),
    userDir && path.join(userDir, ".cache", "codex-runtimes", "codex-primary-runtime", "dependencies", "python", "python.exe"),
    "python"
  ].filter(Boolean);
  return candidates.find((candidate) => !path.isAbsolute(candidate) || existsSync(candidate)) || "python";
}

function runBridge(action, payload) {
  return new Promise((resolve, reject) => {
    const child = spawn(pythonCommand(), [bridgePath, action], {
      cwd: process.cwd(), stdio: ["pipe", "pipe", "pipe"], windowsHide: true
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk.toString("utf8"); });
    child.stderr.on("data", (chunk) => { stderr += chunk.toString("utf8"); });
    child.on("error", reject);
    child.on("close", (code) => {
      let result;
      try { result = stdout.trim() ? JSON.parse(stdout) : null; }
      catch {
        reject(new Error(`Eye-in-hand camera bridge returned invalid JSON: ${stdout || stderr}`));
        return;
      }
      if (code !== 0 || result?.ok === false) {
        const error = new Error(result?.error || stderr.trim() || `Camera bridge exited with code ${code}.`);
        error.details = result || { stderr };
        reject(error);
        return;
      }
      resolve(result);
    });
    child.stdin.end(`${JSON.stringify(payload)}\n`);
  });
}

async function imageFile(filePath) {
  const data = await readFile(filePath);
  const info = await stat(filePath);
  const type = path.extname(filePath).toLowerCase() === ".png" ? "image/png" : "image/jpeg";
  return {
    kind: "camera_image",
    name: path.basename(filePath),
    type,
    size: info.size,
    dataUrl: `data:${type};base64,${data.toString("base64")}`
  };
}

export class EyeInHandCameraController {
  async health(config) { return runBridge("health", config); }
  async capture({ config, robotPose, label }) {
    return runBridge("capture", { ...config, robotPose, label });
  }
  async coarseLocalize({ config, capture }) {
    return runBridge("coarse-localize", { ...config, capture });
  }
  async pixelToBase({ calibrationFile, pixel, robotPose }) {
    return runBridge("pixel-to-base", { calibrationFile, pixel, robotPose });
  }
  async captureBurst({ config, robotPose }) {
    const result = await runBridge("capture-burst", { ...config, robotPose });
    return { ...result, selectedFile: await imageFile(result.selected.path) };
  }
}
