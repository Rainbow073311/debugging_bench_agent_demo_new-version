import "dotenv/config";
import { appendFileSync, existsSync, readdirSync, readFileSync, createReadStream } from "node:fs";
import http from "node:http";
import path from "node:path";
import { mapVlmTargetToExecution } from "../agent/vlmTargetExecutionMapper.js";
import { parseUserCommand } from "../agent/inputCore.js";
import { createBenchRun, transition } from "../domain/run.js";
import { AgentState, StepKind } from "../domain/states.js";
import { getRun, saveRun, updateRun, appendRunEvent, getRunEvents } from "./runStore.js";
import { readJsonBody, sendJson } from "../utils/http.js";
import { webPage } from "./webPage.js";
import { RobotGatewayClient } from "../adapters/robotGatewayClient.js";
import { readMg400Config, writeMg400Config } from "../adapters/mg400Config.js";
import { VlmAgentCaseAdapter } from "../adapters/vlmAgentCaseAdapter.js";
import { VlmAgentServiceRunner } from "../adapters/vlmAgentServiceRunner.js";
import { RemoteVlmAgentServiceRunner } from "../adapters/remoteVlmAgentServiceRunner.js";
import { createEquipmentController } from "../adapters/equipmentControllerFactory.js";
import { ReportGenerator } from "../adapters/reportGenerator.js";
import { getEthernetInfo, autoDetectAdapter } from "../adapters/ethernetConfig.js";
import { defaultVlmAgentRunsDir } from "../adapters/defaultPaths.js";
import { VlmStatusMonitor } from "./vlmStatus.js";
import {
  executeVlmCompletionReadyMove,
  postReadyTargetExecutionEnabled,
  vlmCompletionReadyMoveSucceeded
} from "../agent/vlmCompletionReadyMove.js";
import { executeProbeSignalSearch } from "../agent/probeSignalSearch.js";
import { executeEyeInHandCaptureWorkflow } from "../agent/eyeInHandCaptureWorkflow.js";
import { EyeInHandCameraController } from "../adapters/eyeInHandCameraController.js";
import { projectEyeInHandVlmPixel } from "../agent/calibratedVisionTarget.js";
import { evaluateMg400PoseReachability } from "../domain/mg400Reachability.js";
import { PROBE_SIGNAL_SEARCH_CONFIG } from "../domain/probeSignalSearchConfig.js";
import { readEyeInHandCaptureConfig } from "../domain/eyeInHandCaptureConfig.js";

const port = Number(process.env.PORT || 3000);

const eyeInHandProbeHoverOffsetMm = Number(process.env.EYE_IN_HAND_PROBE_HOVER_OFFSET_MM || 10);
const eyeInHandProbeMarginMm = Number(process.env.EYE_IN_HAND_PROBE_MARGIN_MM || 2);

// Decide runner mode based on VLM_AGENT_RUNNER env var:
//   "cli"        → no service runner (use CLI fallback directly)
//   "remote-svc" → RemoteVlmAgentServiceRunner (upload to remote server)
//   (default)    → VlmAgentServiceRunner (local FastAPI + CLI fallback)
const runnerMode = (process.env.VLM_AGENT_RUNNER || "").trim();
let serviceRunner;
if (runnerMode === "remote-svc" || shouldUseRemoteVlmService(process.env.VLM_AGENT_SERVICE_URL)) {
  serviceRunner = new RemoteVlmAgentServiceRunner();
} else {
  serviceRunner = new VlmAgentServiceRunner();
}
const serviceCaseAdapter = new VlmAgentCaseAdapter();
const robotGateway = new RobotGatewayClient();
const serviceArmController = robotGateway;
const serviceEquipmentController = createEquipmentController();
const serviceReportGenerator = new ReportGenerator();
const serviceCameraController = new EyeInHandCameraController();
const vlmStatus = new VlmStatusMonitor({
  workerCount: Number(process.env.VLM_MONITOR_WORKERS || 2)
});

async function route(request, response) {
  const url = new URL(request.url, `http://${request.headers.host}`);

  if (request.method === "GET" && url.pathname === "/health") {
    sendJson(response, 200, { ok: true, service: "debugging-bench-agent" });
    return;
  }

  if (request.method === "GET" && url.pathname === "/") {
    response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    response.end(webPage);
    return;
  }

  if (request.method === "GET" && url.pathname === "/api/vlm/status") {
    sendJson(response, 200, vlmStatus.snapshot());
    return;
  }

  if (request.method === "POST" && url.pathname === "/api/runs") {
    const payload = await readJsonBody(request);
    const input = parseUserCommand(payload);
    const run = await startServiceBackedRun(input);
    sendJson(response, 202, run);
    return;
  }

  if (request.method === "GET" && url.pathname === "/api/mg400/config") {
    sendJson(response, 200, await readMg400Config());
    return;
  }

  if (request.method === "POST" && url.pathname === "/api/mg400/config") {
    const payload = await readJsonBody(request);
    sendJson(response, 200, await writeMg400Config(payload));
    return;
  }

  if (request.method === "POST" && url.pathname === "/api/mg400/test") {
    const config = await readMg400Config();
    sendJson(response, 200, await robotGateway.runCommand("test", { config }));
    return;
  }

  if (request.method === "GET" && url.pathname === "/api/mg400/status") {
    const config = await readMg400Config();
    sendJson(response, 200, await robotGateway.runCommand("status", { config }));
    return;
  }

  if (request.method === "POST" && url.pathname === "/api/mg400/command") {
    const command = await readJsonBody(request);
    const config = await readMg400Config();
    sendJson(response, 200, await robotGateway.runCommand("command", { config, command }));
    return;
  }

  // 避障服务代理 → Python avoidance_service on port 8001
  const avoidMatch = url.pathname.match(/^\/api\/avoidance(\/.*)?$/);
  if (avoidMatch) {
    const avoidPath = avoidMatch[1] || "/health";
    const avoidUrl = `http://127.0.0.1:8001${avoidPath}`;
    try {
      const avoidResp = await fetch(avoidUrl, {
        method: request.method,
        headers: { "Content-Type": "application/json" },
        body: request.method === "POST" ? JSON.stringify(await readJsonBody(request)) : undefined
      });
      const avoidData = await avoidResp.json();
      sendJson(response, avoidResp.status, avoidData);
    } catch {
      sendJson(response, 503, { ok: false, error: "避障服务未启动 (端口8001)" });
    }
    return;
  }

  // Proxy cloud VLM status to avoid browser CORS issues
  const vlmProxyMatch = url.pathname.match(/^\/api\/vlm-proxy\/runs\/([^/]+)$/);
  if (request.method === "GET" && vlmProxyMatch) {
    try {
      const cloudUrl = `${serviceRunner.baseUrl}/v1/runs/${encodeURIComponent(vlmProxyMatch[1])}`;
      const resp = await fetch(cloudUrl);
      const data = await resp.json();
      sendJson(response, resp.status, data);
    } catch {
      sendJson(response, 502, { error: "Cloud VLM unreachable" });
    }
    return;
  }

  if (request.method === "GET" && url.pathname === "/api/ethernet/info") {
    let name = url.searchParams.get("name") || "";
    if (!name || name === "以太网") {
      const detected = await autoDetectAdapter();
      if (detected) name = detected;
    }
    if (!name) {
      sendJson(response, 400, { ok: false, message: "未找到活跃网卡，请手动输入网卡名称。" });
      return;
    }
    sendJson(response, 200, { ok: true, adapter: await getEthernetInfo(name) });
    return;
  }

  const runMatch = url.pathname.match(/^\/api\/runs\/([^/]+)$/);
  if (request.method === "GET" && runMatch) {
    const run = getRun(runMatch[1]);
    if (!run) {
      sendJson(response, 404, { error: "Run not found." });
      return;
    }

    // Strip large base64 fields to keep response small (polled every 5s)
    sendJson(response, 200, stripBinaryForApi(run));
    return;
  }

  const eventsMatch = url.pathname.match(/^\/api\/runs\/([^/]+)\/events$/);
  if (request.method === "GET" && eventsMatch) {
    await sendRunEvents(request, response, eventsMatch[1], Number(url.searchParams.get("since") || 0));
    return;
  }

  // Serve workspace artifact images for frontend display
  if (request.method === "GET" && url.pathname === "/api/files") {
    const filePath = url.searchParams.get("path") || "";
    if (!filePath) { sendJson(response, 400, { error: "Missing path param" }); return; }
    serveStaticFile(response, filePath);
    return;
  }

  sendJson(response, 404, { error: "Route not found." });
}

const server = http.createServer((request, response) => {
  route(request, response).catch((error) => {
    if (response.headersSent) return;
    const statusCode = error.statusCode || 500;
    sendJson(response, statusCode, {
      ok: false,
      error: error.message,
      details: error.details
    });
  });
});

server.listen(port, () => {
  console.log(`Debugging Bench Agent listening on http://localhost:${port}`);
});

async function startServiceBackedRun(input) {
  const run = createBenchRun(input);
  run.serviceMode = true;
  transition(run, AgentState.PREPARING, "Parsed input; creating VLM task with split planner.");
  saveRun(run);
  const workspace = serviceWorkspace(input, run.runId);
  await executeEyeInHandCaptureWorkflow({
    run,
    input,
    armController: serviceArmController,
    cameraController: serviceCameraController,
    outputDir: workspace,
    onEvent: (type, payload) => appendRunEvent(run.runId, type, payload)
  });
  run.vlmAgentCase = await serviceCaseAdapter.adapt({ runId: run.runId, input });
  const runnerMode = process.env.VLM_AGENT_RUNNER || "";
  const isRemote = serviceRunner instanceof RemoteVlmAgentServiceRunner;

  if (isRemote) {
    const uploadId = `${run.runId}_upload`;
    const uploaded = await serviceRunner.createRunRemote({
      vlmAgentCase: run.vlmAgentCase,
      input,
      runId: uploadId,
      workspace,
    });
    try { await serviceRunner.cancelRun(uploaded.run_id); } catch {}
    const serverTaskFile = `data/cases/${uploadId}/task.yaml`;
    run.vlmService = await serviceRunner.createSplitRun({
      task_file: serverTaskFile,
      run_id: run.runId,
      name: run.runId,
      workspace,
      max_steps: serviceRunner.maxSteps,
    });
  } else {
    run.vlmService = await serviceRunner.createSplitRun({
      task_file: run.vlmAgentCase.taskFile,
      run_id: run.runId,
      name: run.runId,
      workspace,
      max_steps: serviceRunner.maxSteps,
      ...(serviceRunner.envFile ? { env_file: serviceRunner.envFile } : {})
    });
  }

  vlmStatus.registerRun(run, {
    maxSteps: serviceRunner.maxSteps,
    serviceUrl: serviceRunner.baseUrl,
    runnerMode
  });
  saveRun(run);
  appendRunEvent(run.runId, "node.vlm_forwarded", {
    serviceUrl: serviceRunner.baseUrl,
    serviceStatus: run.vlmService.status,
    taskFile: run.vlmAgentCase.taskFile,
    runnerMode,
    workspace,
    splitMode: true,
  });
  vlmStatus.markSubmitted(run.runId, {
    serviceUrl: serviceRunner.baseUrl,
    serviceStatus: run.vlmService.status,
    taskFile: run.vlmAgentCase.taskFile,
    runnerMode,
    workspace,
  });

  finalizeSplitServiceRun(run.runId).catch((error) => {
    const stored = getRun(run.runId);
    if (stored) {
      stored.error = error.message;
      transition(stored, AgentState.REPORTING, `Service-backed split run failed: ${error.message}`);
      appendRunEvent(run.runId, "node.failed", {
        error: error.message,
        details: error.details || null
      });
      vlmStatus.markRunFailed(run.runId, error);
    }
  });
  return run;
}

async function finalizeSplitServiceRun(parentRunId) {
  try { appendFileSync("debug_eye_in_hand.log", `${new Date().toISOString()} finalizeSplitServiceRun called parentRunId=${parentRunId}\n`); } catch {}
  const run = getRun(parentRunId);
  if (!run) { try { appendFileSync("debug_eye_in_hand.log", `${new Date().toISOString()} run not found in store\n`); } catch {} return; }

  const workspace = serviceWorkspace(run.input, parentRunId);
  try { appendFileSync("debug_eye_in_hand.log", `${new Date().toISOString()} workspace=${workspace}\n`); } catch {}

  // VLM service events → push to web UI (best-effort).
  const servicePromise = serviceRunner.waitForRun(parentRunId, {
    onEvent: (event) => {
      try {
        appendRunEvent(parentRunId, event.type, { ...(event.payload || {}), serviceSeq: event.seq, serviceEventId: event.event_id, serviceTimestamp: event.timestamp });
        vlmStatus.handleEvent(parentRunId, event.type, event.payload || {});
        const childRunId = event.payload?.child_run_id;
        const targetPoint = event.payload?.target_point;
        if (childRunId && !vlmStatus.runs.has(childRunId)) {
          const childStatusRun = statusRunForChild(run, childRunId, targetPoint);
          vlmStatus.registerRun(childStatusRun, { maxSteps: serviceRunner.maxSteps, serviceUrl: serviceRunner.baseUrl, runnerMode: process.env.VLM_AGENT_RUNNER || "" });
          vlmStatus.markSubmitted(childRunId, { serviceUrl: serviceRunner.baseUrl, serviceStatus: "running", taskFile: run.vlmAgentCase.taskFile, runnerMode: process.env.VLM_AGENT_RUNNER || "", workspace: serviceWorkspace(run.input, childRunId) });
        }
        if (childRunId) vlmStatus.handleEvent(childRunId, event.type.replace(/^child\./, ""), event.payload || {});
      } catch {}
    }
  }).catch(() => null);

  // Filesystem-based step08 detection (runs in parallel).
  const fsPromise = (async () => {
    const deadline = Date.now() + serviceRunner.timeoutMs;
    while (Date.now() < deadline) {
      const step08Files = [];
      try {
        const entries = readdirSync(workspace, { withFileTypes: true });
        for (const e of entries) {
          if (!e.isDirectory() || !e.name.startsWith("TP")) continue;
          const p = `${workspace}/${e.name}/debug/step08_result.json`.replace(/\\/g, "/");
          if (existsSync(p)) { try { step08Files.push({ tp: e.name, data: JSON.parse(readFileSync(p, "utf8")) }); } catch {} }
        }
      } catch {}
      if (step08Files.length >= 2) {
        try { appendFileSync("debug_eye_in_hand.log", `${new Date().toISOString()} step08 found: ${step08Files.map(f=>f.tp).join(",")}\n`); } catch {}
        return { status: "succeeded", final_answer: { children: step08Files.map(f => ({ target_point: f.tp, child_run_id: null, status: "succeeded", final_answer: { pixel: f.data.pixel, camera_view: f.data.camera_view, tp_id: f.tp } })) }, summary_path: null };
      }
      await new Promise(r => setTimeout(r, 2000));
    }
    return null;
  })();

  const completed = await Promise.race([servicePromise, fsPromise].filter(Boolean));
  if (!completed || completed.status !== "succeeded") {
    run.error = completed.error || `Split service status: ${completed.status}`;
    transition(run, AgentState.REPORTING, run.error);
    appendRunEvent(parentRunId, "node.failed", { error: run.error, service: completed });
    vlmStatus.handleEvent(parentRunId, "node.failed", { error: run.error });
    return;
  }

  const finalAnswer = completed.final_answer || {};
  const children = finalAnswer.children || [];
  const dbg = (msg) => { try { appendFileSync("debug_eye_in_hand.log", `${new Date().toISOString()} ${msg}\n`); } catch {} };
  dbg(`VLM completed, status=${completed.status} children=${children.length}`);

  const allPoints = children
    .filter((c) => c.final_answer)
    .map((c) => ({
      id: c.target_point,
      label: c.target_point,
      final_answer: c.final_answer,
      pixel: normalizePixelFromAnswer(c.final_answer),
      childRunId: c.child_run_id,
    }));

  if (allPoints.length === 0) {
    transition(run, AgentState.REPORTING, "Split VLM completed but no child runs succeeded.");
    appendRunEvent(parentRunId, "node.failed", { error: "No successful child runs", points: allPoints });
    return;
  }

  dbg(`camera.status=${run.execution.camera?.status} allPoints=${allPoints.length}`);
  if (run.execution.camera?.status === "COMPLETED") {
    dbg("entering auto-probe path");
    for (const point of allPoints) {
      const projection = await projectEyeInHandVlmPixel({
        cameraExecution: run.execution.camera,
        cameraController: serviceCameraController,
        pixel: point.pixel
      });
      point.basePoint = projection?.basePoint || null;
    }
    const projectedPoint = allPoints.find((point) => point.id === "TP9" && point.basePoint) || allPoints.find((point) => point.basePoint);
    dbg(`projectedPoint=${projectedPoint?.id} basePoint=${JSON.stringify(projectedPoint?.basePoint)}`);
    if (projectedPoint) {
      const projectedAnswer = { ...projectedPoint.final_answer, tp_id: projectedPoint.id };
      const bp = projectedPoint.basePoint;
      const cameraExec = run.execution.camera;
      const fixedR = cameraExec.fixedR;

      if (!Number.isFinite(fixedR)) {
        run.vlmObservation = buildVlmObservation({
          input: run.input, service: completed,
          finalAnswer: projectedAnswer, pixel: projectedPoint.pixel, points: allPoints
        });
        if (run.vlmObservation.locations?.[0]) run.vlmObservation.locations[0].basePoint = bp;
        run.error = "Eye-in-hand calibrated fixedR is unavailable; cannot compute probe descent pose.";
        transition(run, AgentState.REPORTING, run.error);
        run.report = serviceReportGenerator.create({
          run, ragEvidence: run.ragEvidence || [],
          vlmObservation: run.vlmObservation, measurements: run.execution.equipment
        });
        appendRunEvent(parentRunId, "camera.calibrated_target_blocked", {
          point: projectedPoint, basePoint: bp, reason: "missing-fixedR"
        });
        vlmStatus.markRunFailed(parentRunId, new Error(run.error));
        updateRun(parentRunId, run);
        return;
      }

      // ── Phase 2: ultra-close camera capture + VLM2 refinement ──────
      const ultraCloseZ = -115;
      let refinedBp = bp;
      try {
        // Camera-vs-probe offset computed from the eye-in-hand calibration:
        // ask the bridge where the image-center ray lands from this Z, then
        // shift the robot so that point coincides with the TP.
        let camDx = 0;
        let camDy = -50; // legacy fallback if the calibration query fails
        try {
          const trialPose = { x: bp.x, y: bp.y, z: ultraCloseZ, r: fixedR };
          const centerOffset = await serviceCameraController.cameraCenterOffset({
            calibrationFile: run.execution.camera.calibrationFile,
            robotPose: trialPose
          });
          if (Number.isFinite(centerOffset?.offset?.dx) && Number.isFinite(centerOffset?.offset?.dy)) {
            camDx = centerOffset.offset.dx;
            camDy = centerOffset.offset.dy;
          }
        } catch (e) {
          appendFileSync("debug_eye_in_hand.log", `${new Date().toISOString()} camera-center-offset failed, using legacy (0,-50): ${e.message}\n`);
        }
        const camX = bp.x + camDx;
        const camY = bp.y + camDy;
        appendFileSync("debug_eye_in_hand.log", `${new Date().toISOString()} ultra-close: moving camera via probe (${camX.toFixed(1)},${camY.toFixed(1)},${ultraCloseZ})\n`);
        // Move robot so camera is above TP
        const camPose = { x: camX, y: camY, z: ultraCloseZ, r: fixedR };
        const camReach = evaluateMg400PoseReachability(camPose, { allowAdjustment: true });
        appendFileSync("debug_eye_in_hand.log", `${new Date().toISOString()} ultra cam reachable=${camReach.reachable} pose=${JSON.stringify(camReach.pose)}\n`);
        if (camReach.reachable) {
          const camStep = { id: "ultra-close-camera", kind: StepKind.ARM_MOTION, command: "MOVE_TO_ULTRA_CLOSE", targetLocationId: projectedPoint.id, targetPose: camReach.pose, trajectory: { mode: "direct" } };
          const camMove = await serviceArmController.execute(camStep);
          appendFileSync("debug_eye_in_hand.log", `${new Date().toISOString()} ultra cam move status=${camMove.status}\n`);
          if (camMove.status === "COMPLETED") {
            await new Promise(r => setTimeout(r, 500));
            const ultraCfg = { ...readEyeInHandCaptureConfig(), outputDir: workspace };
            const ultraCapture = await serviceCameraController.capture({ config: ultraCfg, robotPose: camReach.pose, label: "ultra_close" });
            appendRunEvent(parentRunId, "camera.ultra_close_captured", { path: ultraCapture.path });
            // VLM-based TP refinement: project coarse basePoint to ultra-close pixel, VLM finds TP text + silver pad
            const ultraImgPath = ultraCapture.path;
            const { spawn } = await import("node:child_process");
            const coarseBpStr = `${bp.x},${bp.y},${bp.z}`;
            const camPoseStr = `${camReach.pose.x},${camReach.pose.y},${camReach.pose.z},${camReach.pose.r}`;
            const tpId = projectedPoint.id || "TP9";
            appendRunEvent(parentRunId, "camera.ultra_vlm_started", {
              tp_id: tpId,
              message: `VLM refining ${tpId} on ultra-close image...`
            });
            const vlmResult = await new Promise((resolve) => {
              const py = spawn("python", ["C:/Users/32825/Desktop/new_version_demo/calibration/vlm_refine_ultra.py", ultraImgPath, coarseBpStr, camPoseStr, tpId], { env: process.env, timeout: 300000 });
              let out = ""; py.stdout.on("data", d => out += d); py.stderr.on("data", () => {});
              py.on("close", () => { try { resolve(JSON.parse(out)); } catch { resolve(null); } });
              py.on("error", () => resolve(null));
            });
            appendFileSync("debug_eye_in_hand.log", `${new Date().toISOString()} ultra VLM result: ${JSON.stringify(vlmResult)}\n`);
            if (vlmResult?.ok && vlmResult.pad_pixel) {
              const ultraPixel = vlmResult.pad_pixel;
              const textPixel = vlmResult.text_pixel;
              const artifacts = vlmResult.artifacts || {};
              const artifactUrls = {};
              for (const [key, name] of Object.entries(artifacts)) {
                artifactUrls[key] = `/api/files?path=${encodeURIComponent(workspace + "/" + name)}`;
              }
              appendRunEvent(parentRunId, "camera.ultra_vlm_completed", {
                tp_id: tpId,
                text_pixel: textPixel,
                pad_pixel: ultraPixel,
                pad_candidates: vlmResult.pad_candidates,
                artifacts: artifactUrls,
                message: `${tpId} refined: pad=(${ultraPixel[0]},${ultraPixel[1]})`
              });
              appendFileSync("debug_eye_in_hand.log", `${new Date().toISOString()} ultra VLM text=(${textPixel}) pad=(${ultraPixel})\n`);
              try {
                const ultraProjection = await projectEyeInHandVlmPixel({ cameraExecution: { ...run.execution.camera, selectedImage: { robotPose: camReach.pose } }, cameraController: serviceCameraController, pixel: ultraPixel });
                if (ultraProjection?.basePoint) {
                  refinedBp = ultraProjection.basePoint;
                  try { appendFileSync(`${workspace}/ultra_refined_result.json`, JSON.stringify({ pixel: ultraPixel, text_pixel: textPixel, basePoint: refinedBp, source: "vlm_text_pad_refinement", cameraZ: ultraCloseZ, vlm_raw: vlmResult.vlm_raw }, null, 2)); } catch {}
                  appendFileSync("debug_eye_in_hand.log", `${new Date().toISOString()} refined basePoint: x=${refinedBp.x.toFixed(2)} y=${refinedBp.y.toFixed(2)}\n`);
                }
              } catch (projErr) {
                // Keep coarse bp and continue to probe descent; do not abort the whole ultra-close path.
                appendFileSync("debug_eye_in_hand.log", `${new Date().toISOString()} ultra pad projection failed, keep coarse bp: ${projErr.message}\n`);
                appendRunEvent(parentRunId, "camera.ultra_projection_failed", {
                  tp_id: tpId,
                  pad_pixel: ultraPixel,
                  error: projErr.message,
                  message: `${tpId} pad found but projection failed; using coarse basePoint for descent`
                });
              }
            } else {
              appendRunEvent(parentRunId, "camera.ultra_vlm_failed", {
                tp_id: tpId,
                error: vlmResult?.error || "VLM returned no result",
                message: `${tpId} ultra VLM refinement failed`
              });
            }
          }
        }
      } catch (e) {
        appendFileSync("debug_eye_in_hand.log", `${new Date().toISOString()} ultra-close failed: ${e.message}\n`);
      }

      // Compute hover and minimum Z — start from fixed height, descend to PCB
      const safeHoverZ = -120;
      const minimumZ = refinedBp.z - eyeInHandProbeMarginMm;

      // Check reachability and get adjusted pose (use refined basePoint)
      const hoverPose = { x: refinedBp.x, y: refinedBp.y, z: safeHoverZ, r: fixedR };
      const reachability = evaluateMg400PoseReachability(hoverPose, { allowAdjustment: true });
      if (!reachability.reachable) {
        run.error = `Calibrated probe hover pose is unreachable: ${reachability.message}`;
        transition(run, AgentState.REPORTING, run.error);
        appendRunEvent(parentRunId, "node.failed", { error: run.error });
        vlmStatus.markRunFailed(parentRunId, new Error(run.error));
        updateRun(parentRunId, run);
        return;
      }
      const actualHoverPose = reachability.pose;

      // Build dynamic probe search config using actual pose
      const probeConfig = {
        ...PROBE_SIGNAL_SEARCH_CONFIG,
        x: actualHoverPose.x,
        y: actualHoverPose.y,
        r: actualHoverPose.r,
        startZ: actualHoverPose.z,
        minimumZ
      };

      // Build observation and model output
      run.vlmObservation = buildVlmObservation({
        input: run.input, service: completed,
        finalAnswer: projectedAnswer, pixel: projectedPoint.pixel, points: allPoints
      });
      if (run.vlmObservation.locations?.[0]) {
        run.vlmObservation.locations[0].basePoint = bp;
      }
      run.modelOutput = buildModelOutput({
        service: completed, finalAnswer: projectedAnswer,
        pixel: projectedPoint.pixel, pointId: projectedPoint.id,
        basePoint: bp
      });

      // Move to calibrated hover pose
      const hoverStep = {
        id: "eye-in-hand-probe-hover",
        kind: StepKind.ARM_MOTION,
        command: "MOVE_TO_CALIBRATED_PROBE_HOVER",
        targetLocationId: projectedPoint.id,
        targetPose: { ...actualHoverPose },
        trajectory: { mode: "direct" }
      };

      transition(run, AgentState.EXECUTING,
        "Moving MG400 to calibrated hover pose then executing probe signal descent."
      );

      const hoverMove = await serviceArmController.execute(hoverStep);
      run.execution.arm.push(hoverMove);
      appendRunEvent(parentRunId, "robot.eye_in_hand_hover_move_finished", {
        step: hoverStep, result: hoverMove, targetPoint: projectedPoint.id
      });

      if (hoverMove.status !== "COMPLETED") {
        run.error = hoverMove.message || hoverMove.error || "MG400 failed to reach the calibrated probe hover pose.";
        transition(run, AgentState.REPORTING, "Eye-in-hand hover move failed; probe descent was not started.");
        run.report = serviceReportGenerator.create({
          run, ragEvidence: run.ragEvidence || [],
          vlmObservation: run.vlmObservation, measurements: run.execution.equipment
        });
        appendRunEvent(parentRunId, "node.failed", { error: run.error, step: hoverStep, result: hoverMove });
        vlmStatus.markRunFailed(parentRunId, new Error(run.error));
        updateRun(parentRunId, run);
        return;
      }

      vlmStatus.markMg400("executing", {
        runId: parentRunId, stepId: hoverStep.id,
        action: `MG400 moved to eye-in-hand hover pose → ${projectedPoint.id}`, result: hoverMove
      });

      // Execute probe signal search (Z-axis step descent until signal)
      let probeSearch;
      let measurement;
      let flowError = null;
      try {
        probeSearch = await executeProbeSignalSearch({
          armController: serviceArmController,
          equipmentController: serviceEquipmentController,
          config: probeConfig,
          onEvent: ({ type, payload: eventPayload }) => appendRunEvent(parentRunId, type, eventPayload)
        });
        run.execution.arm.push(...probeSearch.movements);
        appendRunEvent(parentRunId, "probe.signal_search_finished", {
          status: probeSearch.status, stopReason: probeSearch.stopReason,
          thresholdV: probeSearch.thresholdV, finalPose: probeSearch.finalPose,
          basePoint: bp, hoverPose: actualHoverPose, minimumZ,
          sampleCount: probeSearch.samples.length, signalConfirmed: probeSearch.signalConfirmed
        });

        if (!probeSearch.signalDetected) {
          throw new Error(
            `Probe search reached minimum Z (${minimumZ.toFixed(2)} mm) from start Z (${safeHoverZ.toFixed(2)} mm) without detecting CHANNEL2 MEAN voltage >= ${probeConfig.thresholdV} V.`
          );
        }

        measurement = await serviceEquipmentController.captureCurrentDisplayReport({
          runId: parentRunId,
          caseId: run.input.caseId,
          targetPoints: allPoints.map((p) => p.id),
          robotPose: probeSearch.finalPose
        });
        run.execution.equipment.push(measurement);
        appendRunEvent(parentRunId, "equipment.dsox1204g_capture_finished", {
          targetPoints: allPoints.map((p) => p.id), result: measurement
        });
      } catch (error) {
        flowError = error;
      }

      if (flowError) {
        run.error = flowError.message;
        transition(run, AgentState.REPORTING,
          "Probe signal search stopped safely; the measurement report was not completed."
        );
        run.report = serviceReportGenerator.create({
          run, ragEvidence: run.ragEvidence || [],
          vlmObservation: run.vlmObservation, measurements: run.execution.equipment
        });
        appendRunEvent(parentRunId, "node.failed", {
          error: run.error, details: flowError.details || null, probeSearch
        });
        vlmStatus.markRunFailed(parentRunId, flowError);
        updateRun(parentRunId, run);
        return;
      }

      vlmStatus.markMg400("idle", {
        runId: parentRunId, stepId: hoverStep.id,
        action: "MG400 probe descent completed; at signal contact point.", result: probeSearch
      });
      if (run.vlmObservation.locations?.[0]) {
        run.vlmObservation.locations[0].probeContactZ = probeSearch.finalPose?.z ?? null;
      }
      transition(run, AgentState.REPORTING,
        `Eye-in-hand VLM completed; probe descended from ${safeHoverZ.toFixed(2)} mm to ${probeSearch.finalPose.z.toFixed(2)} mm at calibrated Base (${bp.x.toFixed(2)}, ${bp.y.toFixed(2)}) and detected CHANNEL2 MEAN voltage >= ${probeConfig.thresholdV} V. ${measurement?.instrument || "Oscilloscope"} display saved to Excel.`
      );
      run.report = serviceReportGenerator.create({
        run, ragEvidence: run.ragEvidence || [],
        vlmObservation: run.vlmObservation, measurements: run.execution.equipment
      });
      appendRunEvent(parentRunId, "node.completed", {
        points: allPoints, calibratedBasePoint: bp,
        probeSearchStatus: probeSearch.status, signalConfirmed: probeSearch.signalConfirmed,
        finalPose: probeSearch.finalPose
      });
      vlmStatus.handleEvent(parentRunId, "node.completed", {
        runId: parentRunId, calibratedBasePoint: bp,
        signalConfirmed: probeSearch.signalConfirmed
      });
      updateRun(parentRunId, run);
      return;
    }

    run.error = "The close-image VLM result did not contain a pixel that could be projected to calibrated Base XY.";
    transition(run, AgentState.REPORTING, `${run.error} No probe motion was issued.`);
    run.report = serviceReportGenerator.create({
      run,
      ragEvidence: run.ragEvidence || [],
      vlmObservation: run.vlmObservation,
      measurements: run.execution.equipment
    });
    appendRunEvent(parentRunId, "node.failed", { error: run.error, automaticProbePaused: true });
    vlmStatus.markRunFailed(parentRunId, new Error(run.error));
    updateRun(parentRunId, run);
    return;
  }

  transition(run, AgentState.EXECUTING,
    `Split VLM completed: ${allPoints.length}/${children.length} child runs succeeded; executing MG400 flow.`
  );

  // All VLM child runs are complete at this point. Move to the operator-recorded
  // ready pose exactly once before any per-target MG400 action.
  const readyMove = await executeVlmCompletionReadyMove(serviceArmController);
  run.execution.arm.push(readyMove.result);
  vlmStatus.markMg400(
    vlmCompletionReadyMoveSucceeded(readyMove.result) ? "executing" : "blocked",
    {
      runId: parentRunId,
      stepId: readyMove.step.id,
      action: "VLM 已完成，MG400 移动到固定起始位置",
      result: readyMove.result
    }
  );
  appendRunEvent(parentRunId, "robot.vlm_completion_ready_move_finished", {
    step: readyMove.step,
    result: readyMove.result
  });
  if (!vlmCompletionReadyMoveSucceeded(readyMove.result)) {
    run.error = readyMove.result.message
      || readyMove.result.error
      || "MG400 failed to reach the required VLM-completion ready position.";
    transition(
      run,
      AgentState.REPORTING,
      "VLM completed, but the required ready-position move failed; later hardware steps were stopped."
    );
    run.report = serviceReportGenerator.create({
      run,
      ragEvidence: run.ragEvidence || [],
      vlmObservation: run.vlmObservation,
      measurements: run.execution.equipment
    });
    appendRunEvent(parentRunId, "node.failed", {
      error: run.error,
      step: readyMove.step,
      result: readyMove.result
    });
    vlmStatus.markRunFailed(parentRunId, new Error(run.error));
    updateRun(parentRunId, run);
    return;
  }

  if (!postReadyTargetExecutionEnabled()) {
    const reportPoint = allPoints[0];
    const reportFinalAnswer = {
      ...reportPoint.final_answer,
      tp_id: reportPoint.id
    };
    run.vlmObservation = buildVlmObservation({
      input: run.input,
      service: completed,
      finalAnswer: reportFinalAnswer,
      pixel: reportPoint.pixel,
      points: allPoints
    });
    run.modelOutput = buildModelOutput({
      service: completed,
      finalAnswer: reportFinalAnswer,
      pixel: reportPoint.pixel,
      pointId: reportPoint.id
    });
    let measurement;
    let probeSearch;
    let flowError = null;
    try {
      probeSearch = await executeProbeSignalSearch({
        armController: serviceArmController,
        equipmentController: serviceEquipmentController,
        onEvent: ({ type, payload }) => appendRunEvent(parentRunId, type, payload)
      });
      run.execution.arm.push(...probeSearch.movements);
      appendRunEvent(parentRunId, "probe.signal_search_finished", {
        status: probeSearch.status,
        stopReason: probeSearch.stopReason,
        thresholdV: probeSearch.thresholdV,
        finalPose: probeSearch.finalPose,
        sampleCount: probeSearch.samples.length,
        signalConfirmed: probeSearch.signalConfirmed
      });
      if (!probeSearch.signalDetected) {
        throw new Error(
          "Probe search reached the fixed Z lower limit without detecting a CHANNEL2 MEAN voltage of 3 V or more."
        );
      }

      measurement = await serviceEquipmentController.captureCurrentDisplayReport({
        runId: parentRunId,
        caseId: run.input.caseId,
        targetPoints: allPoints.map((point) => point.id),
        robotPose: probeSearch.finalPose
      });
      run.execution.equipment.push(measurement);
      appendRunEvent(parentRunId, "equipment.dsox1204g_capture_finished", {
        targetPoints: allPoints.map((point) => point.id),
        result: measurement
      });
    } catch (error) {
      flowError = error;
    }

    if (flowError) {
      run.error = flowError.message;
      transition(
        run,
        AgentState.REPORTING,
        "Probe signal search stopped safely; the measurement report was not completed."
      );
      run.report = serviceReportGenerator.create({
        run,
        ragEvidence: run.ragEvidence || [],
        vlmObservation: run.vlmObservation,
        measurements: run.execution.equipment
      });
      appendRunEvent(parentRunId, "node.failed", {
        error: run.error,
        details: flowError.details || null,
        probeSearch
      });
      vlmStatus.markRunFailed(parentRunId, flowError);
      updateRun(parentRunId, run);
      return;
    }

    vlmStatus.markMg400("idle", {
      runId: parentRunId,
      stepId: readyMove.step.id,
      action: "MG400 已到达固定点并保持静止",
      result: probeSearch
    });
    transition(
      run,
      AgentState.REPORTING,
      `VLM completed; the probe stopped at the first 3 V threshold crossing and remained there, and the ${measurement.instrument} display was saved to Excel.`
    );
    run.report = serviceReportGenerator.create({
      run,
      ragEvidence: run.ragEvidence || [],
      vlmObservation: run.vlmObservation,
      measurements: run.execution.equipment
    });
    appendRunEvent(parentRunId, "node.completed", {
      points: allPoints,
      childCount: children.length,
      postReadyExecutionPaused: true
    });
    vlmStatus.handleEvent(parentRunId, "node.completed", {
      runId: parentRunId,
      points: allPoints,
      postReadyExecutionPaused: true
    });
    updateRun(parentRunId, run);
    return;
  }

  // Execute robot arm for each successful point
  const blockedLocations = new Map();
  for (const point of allPoints) {
    const finalAnswer = { ...point.final_answer, tp_id: point.id };
    const pixel = point.pixel;
    const childRunId = point.childRunId;

    run.vlmObservation = buildVlmObservation({ input: run.input, service: completed, finalAnswer, pixel, points: allPoints });
    run.modelOutput = buildModelOutput({ service: completed, finalAnswer, pixel, pointId: point.id });
    run.plan = mapVlmTargetToExecution({
      input: run.input,
      ragEvidence: run.ragEvidence || [],
      vlmObservation: run.vlmObservation,
      modelOutput: run.modelOutput
    });
    appendRunEvent(parentRunId, "node.execution_mapping_created", { plan: run.plan, targetPoint: point.id });

    for (const step of run.plan.steps) {
      if (step.kind === StepKind.ARM_MOTION || step.kind === StepKind.VISUAL_CAPTURE) {
        const armResult = await serviceArmController.execute(step);
        run.execution.arm.push(armResult);
        vlmStatus.markMg400("executing", { runId: parentRunId, stepId: step.id, action: `MG400 ${step.kind} → ${point.id}`, result: armResult });
        appendRunEvent(parentRunId, "robot.action_finished", { step, result: armResult, targetPoint: point.id });
        if (childRunId) {
          await serviceRunner.postObservation(childRunId, {
            type: "robot.observation",
            source: "node-mg400-gateway",
            payload: { ...armResult, target_point: point.id }
          }).catch(() => {});
        }
        if (armResult.status === "BLOCKED" && step.targetLocationId) {
          blockedLocations.set(step.targetLocationId, armResult);
        }
      }
      if (step.kind === StepKind.EQUIPMENT_MEASUREMENT) {
        const blockedArm = blockedLocations.get(step.locationId);
        if (blockedArm) {
          run.execution.equipment.push({
            stepId: step.id, status: "SKIPPED", instrument: step.instrument,
            signal: step.signal, locationId: step.locationId, pass: false,
            reason: blockedArm.message || blockedArm.error || "Arm motion did not reach the requested measurement point."
          });
        } else {
          run.execution.equipment.push(await serviceEquipmentController.measure(step));
        }
        appendRunEvent(parentRunId, "equipment.measurement_finished", { step, result: run.execution.equipment.at(-1), targetPoint: point.id });
      }
    }

    // Wait for robot to settle before next point
    await new Promise(resolve => setTimeout(resolve, 5000));
  }

  vlmStatus.markMg400("idle", { action: "MG400 空闲" });
  transition(run, AgentState.REPORTING, "Execution completed; generating report.");
  run.report = serviceReportGenerator.create({
    run,
    ragEvidence: run.ragEvidence || [],
    vlmObservation: run.vlmObservation,
    measurements: run.execution.equipment
  });
  appendRunEvent(parentRunId, "node.completed", {
    points: allPoints,
    childCount: children.length,
  });
  vlmStatus.handleEvent(parentRunId, "node.completed", { runId: parentRunId, points: allPoints });
  updateRun(parentRunId, run);
}

function reconcileSplitMonitor(parentRunId, completed) {
  const terminalEvent = completed.status === "succeeded" ? "agent.final" : "agent.failed";
  vlmStatus.handleEvent(parentRunId, terminalEvent, {
    final_answer: completed.final_answer || null,
    error: completed.error || null,
  });

  const children = completed.final_answer?.children || completed.vlm_split?.child_runs || [];
  for (const child of children) {
    const childRunId = child.child_run_id;
    if (!childRunId || !vlmStatus.runs.has(childRunId)) continue;
    const childStatus = child.status || (child.final_answer ? "succeeded" : "failed");
    vlmStatus.handleEvent(
      childRunId,
      childStatus === "succeeded" ? "agent.final" : "agent.failed",
      {
        final_answer: child.final_answer || null,
        error: child.error || null,
      }
    );
  }
}

function statusRunForChild(parentRun, childRunId, label) {
  const instruction = parentRun.input?.instruction || parentRun.input?.command || "";
  return {
    ...parentRun,
    runId: childRunId,
    input: {
      ...parentRun.input,
      instruction: `${instruction} / ${label}`,
      command: `${instruction} / ${label}`
    }
  };
}

function normalizePixelFromAnswer(finalAnswer) {
  if (!finalAnswer) return null;
  const pixel = finalAnswer.pixel || finalAnswer.pixel_array;
  if (Array.isArray(pixel) && pixel.length === 2) return [Math.round(pixel[0]), Math.round(pixel[1])];
  const points = finalAnswer.points;
  if (Array.isArray(points) && points.length > 0) {
    const p = points[0].pixel || points[0].pixel_array;
    if (Array.isArray(p)) return [Math.round(p[0]), Math.round(p[1])];
  }
  return null;
}

function serviceWorkspace(input, runId = null) {
  const caseId = sanitizeSegment(input.caseId) || sanitizeSegment(input.command) || "inputdemo-case";
  const workspaceName = runId ? `${caseId}-${sanitizeSegment(runId)}` : caseId;
  const base = defaultVlmAgentRunsDir();
  // Use forward-slash join to avoid Windows backslash in Linux paths
  const sep = base.includes("\\") ? "\\" : "/";
  return base + (base.endsWith(sep) || base.endsWith("/") ? "" : "/") + workspaceName;
}

function buildVlmObservation({ input, service, finalAnswer, pixel }) {
  const locationId = finalAnswer?.tp_id || finalAnswer?.test_point || "vlm-target";
  return {
    model: process.env.VLM_MODEL || "vlm-agent-service",
    provider: "debugging-agent-v2-service",
    realModelService: true,
    imageRef: input.cameraImage?.name || input.visualCapture?.imageRef || null,
    promptMode: input.prompt?.mode || "text",
    workspace: serviceWorkspace(input),
    runDir: service.run_dir,
    summaryPath: service.summary_path,
    stoppedReason: service.stopped_reason,
    finalAnswer,
    pixel: pixel ? { x: pixel[0], y: pixel[1] } : null,
    confidence: Number.isFinite(Number(finalAnswer?.confidence)) ? Number(finalAnswer.confidence) : null,
    modelInputSummary: {
      attachmentCount: input.modelAttachments.length,
      schematicCount: input.schematicDiagrams.length,
      bitImageCount: input.bitImages.length,
      attachmentNames: input.modelAttachments.map((item) => item.name)
    },
    benchOverview: {
      boardDetected: Boolean(pixel),
      instrumentsDetected: [],
      armReachableZones: pixel ? ["vlm-localized-test-point"] : []
    },
    locations: [
      {
        id: locationId,
        label: finalAnswer?.tp_id || finalAnswer?.label || "VLM localized test point",
        pixel: pixel ? { x: pixel[0], y: pixel[1] } : null,
        cameraView: finalAnswer?.camera_view || null
      }
    ],
    recommendedMeasurements: [
      {
        signal: finalAnswer?.signal || finalAnswer?.tp_id || "TARGET_SIGNAL",
        locationId,
        instrument: "oscilloscope",
        expectedRange: finalAnswer?.expected_range || "See real VLM final_answer.",
        reason: finalAnswer?.reason || `Real VLM agent result for: ${input.command}`
      }
    ]
  };
}

function buildModelOutput({ service, finalAnswer, pixel, pointId = null, basePoint = null }) {
  const pose = basePoint
    ? null
    : normalizePose(finalAnswer?.mg400Pose || finalAnswer?.mg400_pose || finalAnswer?.pose)
      || poseFromPixel(pixel || finalAnswer?.pixel, pointId);
  return {
    model: process.env.VLM_MODEL || "vlm-agent-service",
    provider: "debugging-agent-v2-service",
    realModelService: true,
    inputFormat: "debugging-agent-v2-task",
    testPoint: finalAnswer?.tp_id || finalAnswer?.test_point || null,
    confidence: Number.isFinite(Number(finalAnswer?.confidence)) ? Number(finalAnswer.confidence) : null,
    pixel: pixel ? { x: pixel[0], y: pixel[1] } : null,
    mg400Pose: pose,
    calibratedBasePoint: basePoint,
    finalAnswer,
    summaryPath: service.summary_path,
    runDir: service.run_dir,
    workspace: service.run_dir ? service.run_dir.replace(/\\runs\\.*$/, "") : null,
    reason: basePoint
      ? "Close-image pixel was projected to calibrated Base XY; contact motion remains paused for hover validation."
      : pose
      ? "VLM service returned localization output; pose was returned or derived from pixel."
      : "VLM service returned localization output."
  };
}

function normalizePixel(value) {
  if (!Array.isArray(value) || value.length !== 2) return null;
  const x = Number(value[0]);
  const y = Number(value[1]);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return [Math.round(x), Math.round(y)];
}

function normalizePose(value) {
  if (!value || typeof value !== "object") return null;
  const pose = {
    x: Number(value.x),
    y: Number(value.y),
    z: Number(value.z),
    r: Number(value.r)
  };
  if (Object.values(pose).some((item) => !Number.isFinite(item))) return null;
  return pose;
}

function poseFromPixel(value, pointId = null) {
  // Hardcoded test mapping: pixel → MG400 world coordinates
  if (pointId) {
    const known = HARDCODED_POSES[String(pointId).toUpperCase()];
    if (known) return { ...known };
  }
  if (!Array.isArray(value) || value.length !== 2) return null;
  const x = Number(value[0]);
  const y = Number(value[1]);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x: Math.round(x), y: Math.round(y), z: 0, r: 0 };
}

const HARDCODED_POSES = {
  "TP1": { x: 320, y: 50, z: 60, r: 0 },
  "TP6": { x: 280, y: -120, z: 70, r: 0 },
};

function stripBinaryForApi(run) {
  // Shallow-clone and remove large base64 dataUrl fields + nodeEvents
  // to avoid circular reference and 6+ MB responses on poll.
  if (!run || !run.input) return run;
  const { nodeEvents, ...rest } = run;
  const stripped = { ...rest };
  const input = { ...run.input };
  for (const key of ["cameraImage", "cameraImageBack", "bitImage", "bitPdf", "schematicImage", "schematicPdf"]) {
    const field = input[key];
    if (field && field.dataUrl && field.dataUrl.length > 1000) {
      input[key] = { ...field, dataUrl: "[stripped:" + (field.dataUrl.length || 0) + "]" };
    }
  }
  stripped.input = input;
  return stripped;
}

function sanitizeSegment(value) {
  return String(value || "")
    .trim()
    .replace(/[<>:"/\\|?*\x00-\x1F]/g, "_")
    .replace(/\s+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 120);
}

function shouldUseRemoteVlmService(serviceUrl) {
  if (!serviceUrl) return false;
  try {
    const hostname = new URL(serviceUrl).hostname.toLowerCase();
    return !["localhost", "127.0.0.1", "::1", "0.0.0.0"].includes(hostname);
  } catch {
    return false;
  }
}

async function sendRunEvents(request, response, runId, since) {
  const run = getRun(runId);
  if (!run) {
    sendJson(response, 404, { error: "Run not found." });
    return;
  }

  const wantsSse = String(request.headers.accept || "").includes("text/event-stream");
  if (!wantsSse) {
    sendJson(response, 200, { runId, events: getRunEvents(runId, since) || [] });
    return;
  }

  response.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive"
  });

  let seq = Number(since || 0);
  let closed = false;
  request.on("close", () => {
    closed = true;
  });

  while (!closed) {
    const events = getRunEvents(runId, seq);
    if (events === null) break;
    for (const event of events) {
      seq = Math.max(seq, Number(event.seq) || seq);
      response.write(`data: ${JSON.stringify(event)}\n\n`);
    }

    const current = getRun(runId);
    if (!current || current.report || current.error) break;
    await sleep(500);
  }

  response.end();
}

function serveStaticFile(response, filePath) {
  // Only allow image files from workspace directories
  const ext = path.extname(filePath).toLowerCase();
  const mimeTypes = { ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".bmp": "image/bmp" };
  const contentType = mimeTypes[ext];
  if (!contentType) { sendJson(response, 403, { error: "File type not allowed" }); return; }
  // Security: path must contain "workspace" or "calibration"
  const normalized = path.normalize(filePath).replace(/\\/g, "/");
  if (!normalized.includes("workspace") && !normalized.includes("calibration")) {
    sendJson(response, 403, { error: "Access denied" }); return;
  }
  if (!existsSync(filePath)) { sendJson(response, 404, { error: "File not found" }); return; }
  try {
    const stream = createReadStream(filePath);
    response.writeHead(200, { "Content-Type": contentType, "Cache-Control": "no-cache" });
    stream.pipe(response);
  } catch (e) {
    sendJson(response, 500, { error: "Failed to read file" });
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
