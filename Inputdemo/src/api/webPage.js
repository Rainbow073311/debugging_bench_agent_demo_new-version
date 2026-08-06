export const webPage = String.raw`<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Inputdemo MG400 Agent</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f7fb;
      --panel: #ffffff;
      --panel-soft: #f8fafc;
      --text: #172033;
      --muted: #667085;
      --line: #d9e0ea;
      --line-strong: #b9c5d4;
      --primary: #146c94;
      --primary-dark: #0e516f;
      --accent: #d97706;
      --good: #0f766e;
      --warn: #b45309;
      --danger: #b42318;
      --shadow: 0 18px 40px rgba(23, 32, 51, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background:
        linear-gradient(180deg, #eef6fb 0, rgba(238, 246, 251, 0) 360px),
        var(--bg);
      color: var(--text);
    }
    header {
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    .topbar {
      max-width: 1440px;
      margin: 0 auto;
      padding: 16px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    .brand { display: grid; gap: 3px; }
    h1 { margin: 0; font-size: 22px; line-height: 1.25; font-weight: 750; letter-spacing: 0; }
    .subtitle { color: var(--muted); font-size: 13px; }
    main {
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: minmax(480px, 1.08fr) minmax(420px, 0.92fr);
      gap: 20px;
      align-items: start;
    }
    .column { display: grid; gap: 20px; }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .section-head {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #fff 0, #fbfdff 100%);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .section-title { display: grid; gap: 3px; }
    h2 { margin: 0; font-size: 16px; font-weight: 750; letter-spacing: 0; }
    h3 { margin: 0 0 10px; font-size: 14px; font-weight: 720; letter-spacing: 0; }
    .section-note, .hint, .status { color: var(--muted); font-size: 13px; line-height: 1.55; }
    .section-body { padding: 18px; }
    .stack { display: grid; gap: 16px; }
    .mini-stack { display: grid; gap: 10px; }
    .grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .grid-4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .upload-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 14px; }
    .subpanel {
      padding: 14px;
      border: 1px solid #e5ebf3;
      border-radius: 8px;
      background: var(--panel-soft);
    }
    label {
      display: block;
      margin-bottom: 7px;
      font-size: 13px;
      font-weight: 650;
      color: #344054;
    }
    textarea, input[type="text"], input[type="number"], select {
      width: 100%;
      border: 1px solid var(--line-strong);
      border-radius: 7px;
      padding: 10px 11px;
      font: inherit;
      color: var(--text);
      background: #fff;
      outline: none;
    }
    textarea:focus, input:focus, select:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(20, 108, 148, 0.12);
    }
    textarea { min-height: 126px; resize: vertical; }
    input[type="file"] {
      width: 100%;
      min-height: 44px;
      border: 1px dashed #9fb0c6;
      border-radius: 7px;
      padding: 9px;
      background: #fff;
      color: var(--muted);
    }
    input[type="checkbox"] { accent-color: var(--primary); }
    .segmented {
      display: inline-grid;
      grid-template-columns: 1fr 1fr;
      border: 1px solid var(--line-strong);
      border-radius: 7px;
      overflow: hidden;
      background: #fff;
    }
    .segmented button {
      border: 0;
      border-radius: 0;
      min-height: 36px;
      background: transparent;
      color: var(--muted);
    }
    .segmented button.active { background: var(--primary); color: #fff; }
    .hidden { display: none; }
    .row {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }
    button {
      border: 1px solid var(--line-strong);
      border-radius: 7px;
      min-height: 38px;
      padding: 0 13px;
      background: #fff;
      color: var(--text);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { border-color: var(--primary); color: var(--primary); }
    button.primary { border-color: var(--primary); background: var(--primary); color: #fff; }
    button.primary:hover { background: var(--primary-dark); color: #fff; }
    button:disabled { opacity: 0.62; cursor: progress; }
    .status { max-width: 100%; white-space: pre-wrap; overflow-wrap: anywhere; }
    .status.error { color: var(--danger); }
    .status.ok { color: var(--good); }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 3px 9px;
      border-radius: 999px;
      background: #e6f4f1;
      border: 1px solid #b7ded7;
      color: var(--good);
      font-size: 12px;
      font-weight: 750;
      white-space: nowrap;
    }
    .badge.mock { background: #fff7ed; border-color: #fed7aa; color: var(--warn); }
    .file-list { display: grid; gap: 8px; margin-top: 10px; }
    .file-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 10px 11px;
      border: 1px solid #e5e8ef;
      border-radius: 7px;
      background: #fff;
      font-size: 13px;
    }
    .file-row span { overflow-wrap: anywhere; }
    .camera-preview { display: grid; gap: 8px; margin-top: 10px; }
    .camera-preview video, .camera-preview img {
      width: 100%;
      aspect-ratio: 4 / 3;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #111827;
      object-fit: cover;
    }
    .jog { display: grid; grid-template-columns: repeat(4, minmax(64px, 1fr)); gap: 8px; }
    .kv { display: grid; gap: 10px; margin: 0; }
    .kv div { display: grid; grid-template-columns: 118px minmax(0, 1fr); gap: 12px; font-size: 13px; }
    .kv dt { color: var(--muted); }
    .kv dd { margin: 0; overflow-wrap: anywhere; }
    pre {
      margin: 0;
      max-height: 460px;
      overflow: auto;
      border-radius: 7px;
      padding: 14px;
      background: #111827;
      color: #f8fafc;
      font-size: 12px;
      line-height: 1.55;
    }
    .monitor-overview {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid #e3eaf2;
      border-radius: 8px;
      background: #fbfdff;
      padding: 12px;
      display: grid;
      gap: 4px;
      min-width: 0;
    }
    .metric strong { font-size: 20px; line-height: 1; }
    .metric span { color: var(--muted); font-size: 12px; }
    .worker-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .worker-card {
      position: relative;
      overflow: hidden;
      border: 1px solid #d8e3ee;
      border-radius: 8px;
      background: #fff;
      padding: 13px;
      display: grid;
      gap: 9px;
      min-height: 170px;
    }
    .worker-card.running {
      border-color: rgba(20, 108, 148, 0.42);
      box-shadow: 0 12px 28px rgba(20, 108, 148, 0.10);
    }
    .worker-card.running::before {
      content: "";
      position: absolute;
      inset: 0 0 auto 0;
      height: 3px;
      background: linear-gradient(90deg, #146c94, #0f766e, #d97706, #146c94);
      background-size: 240% 100%;
      animation: monitor-flow 1.8s linear infinite;
    }
    .worker-card.failed { border-color: rgba(180, 35, 24, 0.45); }
    .worker-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
    }
    .worker-title { font-weight: 800; font-size: 14px; }
    .worker-status {
      border-radius: 999px;
      padding: 3px 8px;
      background: #eef2f7;
      color: #475467;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }
    .worker-status.running { background: #e6f4f1; color: var(--good); }
    .worker-status.failed { background: #fef3f2; color: var(--danger); }
    .worker-run {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .worker-action {
      font-size: 14px;
      line-height: 1.45;
      min-height: 40px;
    }
    .worker-meta {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .worker-meta div {
      border: 1px solid #eef2f7;
      border-radius: 7px;
      padding: 8px;
      background: #f8fafc;
      min-width: 0;
    }
    .worker-meta span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 3px;
    }
    .worker-meta strong {
      display: block;
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .activity-list {
      display: grid;
      gap: 8px;
    }
    .activity-row {
      border: 1px solid #e5ebf3;
      border-radius: 7px;
      padding: 9px 10px;
      background: #fff;
      display: grid;
      gap: 3px;
      font-size: 13px;
    }
    .activity-row strong { overflow-wrap: anywhere; }
    .activity-row span { color: var(--muted); font-size: 12px; }
    .mg400-strip {
      border: 1px solid #e5ebf3;
      border-radius: 8px;
      background: #fbfdff;
      padding: 11px 12px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .step-timeline {
      display: grid;
      gap: 8px;
      max-height: 360px;
      overflow: auto;
      padding-right: 2px;
    }
    .step-row {
      border: 1px solid #e5ebf3;
      border-radius: 7px;
      background: #fff;
      padding: 10px 11px;
      display: grid;
      gap: 5px;
    }
    .step-row.running {
      border-color: rgba(20, 108, 148, 0.42);
      background: #f6fbfd;
    }
    .step-row.failed {
      border-color: rgba(180, 35, 24, 0.42);
      background: #fffafa;
    }
    .step-row.final {
      border-color: rgba(15, 118, 110, 0.42);
      background: #f6fbfa;
    }
    .step-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .step-title {
      font-weight: 760;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .step-state {
      border-radius: 999px;
      padding: 2px 7px;
      background: #eef2f7;
      color: #475467;
      font-size: 11px;
      font-weight: 800;
      white-space: nowrap;
    }
    .step-row.running .step-state { background: #e8f4f8; color: var(--primary); }
    .step-row.failed .step-state { background: #fef3f2; color: var(--danger); }
    .step-row.final .step-state { background: #e6f4f1; color: var(--good); }
    .step-meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }
    .step-images {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      padding: 6px 0 2px;
    }
    .step-images img {
      max-width: 180px;
      max-height: 135px;
      object-fit: contain;
      border: 1px solid var(--line);
      border-radius: 6px;
      transition: transform 0.15s;
    }
    .step-images img:hover {
      transform: scale(1.04);
      border-color: var(--primary);
    }
    @keyframes monitor-flow {
      from { background-position: 0 0; }
      to { background-position: 240% 0; }
    }
    @media (max-width: 1050px) {
      main { grid-template-columns: 1fr; padding: 16px; }
      .topbar { padding: 14px 16px; }
    }
    @media (max-width: 720px) {
      .grid-2, .grid-4, .upload-grid { grid-template-columns: 1fr; }
      .monitor-overview, .worker-grid, .worker-meta { grid-template-columns: 1fr; }
      .jog { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .section-head { align-items: flex-start; flex-direction: column; }
      .kv div { grid-template-columns: 1fr; gap: 3px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div class="brand">
        <h1>Inputdemo MG400 Agent</h1>
        <div class="subtitle">调试任务输入、视觉材料、机械臂配置与运行结果</div>
      </div>
      <span id="modeBadge" class="badge mock">机械臂：读取中</span>
    </div>
  </header>

  <main>
    <div class="column">
      <section>
        <div class="section-head">
          <div class="section-title">
            <h2>任务输入</h2>
            <div class="section-note">输入工程师指令，选择文字或语音转写。</div>
          </div>
          <div class="segmented" role="tablist" aria-label="Prompt type">
            <button type="button" id="textMode" class="active">文字</button>
            <button type="button" id="voiceMode">语音</button>
          </div>
        </div>
        <form id="runForm" class="section-body stack">
          <div id="textPrompt">
            <label for="instruction">Instruction</label>
            <textarea id="instruction" placeholder="例如：我想确认输入电压 VCP 是否正常"></textarea>
            <p class="hint">这段文字会作为本次任务的 Instruction，并参与测试点识别。</p>
          </div>

          <div id="voicePrompt" class="hidden">
            <label for="voiceTranscript">语音转写文本</label>
            <textarea id="voiceTranscript" placeholder="点击开始听写后，说出工程师指令"></textarea>
            <div class="row">
              <button type="button" id="startVoice">开始听写</button>
              <button type="button" id="stopVoice" disabled>停止</button>
              <span id="voiceStatus" class="status">识别语言：中文普通话</span>
            </div>
            <p class="hint">浏览器会把语音实时转成文字，转写结果会同步作为本次 Instruction。</p>
          </div>

          <div class="subpanel">
            <h3>Case 信息</h3>
            <div class="grid-2">
              <div>
                <label for="caseId">Case ID</label>
                <input id="caseId" type="text" value="case-001">
              </div>
              <div>
                <label for="operator">Operator</label>
                <input id="operator" type="text" value="demo-user">
              </div>
            </div>
          </div>

          <div class="subpanel">
            <h3>图像与文档输入</h3>
            <div class="upload-grid">
              <div class="mini-stack">
                <div>
                  <label for="cameraImage">Camera_image (正面)</label>
                  <input id="cameraImage" type="file" accept="image/*">
                </div>
                <div class="row">
                  <button type="button" id="openCamera">打开摄像头</button>
                  <button type="button" id="checkCamera">检测设备</button>
                  <button type="button" id="captureCamera" disabled>拍照</button>
                  <button type="button" id="closeCamera" disabled>关闭摄像头</button>
                </div>
                <span id="cameraStatus" class="status"></span>
                <div class="camera-preview">
                  <video id="cameraVideo" class="hidden" autoplay playsinline></video>
                  <canvas id="cameraCanvas" class="hidden"></canvas>
                  <img id="cameraSnapshot" class="hidden" alt="Camera snapshot">
                </div>
                <p class="hint">相机拍摄的 PCBA 实物图（正面），用于 VLM 识别实物位置。</p>
                <div id="cameraImageList" class="file-list"></div>
              </div>

              <div class="mini-stack">
                <div>
                  <label for="cameraImageBack">Camera_image (背面)</label>
                  <input id="cameraImageBack" type="file" accept="image/*">
                </div>
                <p class="hint">可选上传。平台会先从原理图确定 TP，再根据位号图自动判断正面/背面；若为背面，将使用 PCB 外框和安装/定位孔进行映射。</p>
                <div id="cameraImageBackList" class="file-list"></div>
              </div>

              <div class="mini-stack">
                <div>
                  <label for="bitPdf">位号图 PDF</label>
                  <input id="bitPdf" type="file" accept="application/pdf,.pdf" multiple>
                  <p class="hint">位号图 / 装配图 PDF，会适配为 VLM agent 的 assembly_drawing_pdf。</p>
                </div>
                <div>
                  <label for="bitImage">位号图图片</label>
                  <input id="bitImage" type="file" accept="image/*" multiple>
                  <p class="hint">作为可选的 assembly_drawing 图片 fallback。</p>
                </div>
                <div>
                  <label for="schematicPdf">原理图 PDF</label>
                  <input id="schematicPdf" type="file" accept="application/pdf,.pdf" multiple>
                  <p class="hint">用于检索网络、TP 与参考证据。</p>
                </div>
                <div>
                  <label for="schematicImage">原理图图片</label>
                  <input id="schematicImage" type="file" accept="image/*" multiple>
                  <p class="hint">用于 VLM 视觉判读工程师意图。</p>
                </div>
              </div>
            </div>
            <div id="attachmentList" class="file-list"></div>
          </div>

          <div class="row">
            <button id="runButton" class="primary" type="submit">运行 Agent</button>
            <span id="runStatus" class="status"></span>
          </div>
        </form>
      </section>

      <section>
        <div class="section-head">
          <div class="section-title">
            <h2>运行监控</h2>
            <div class="section-note">实时显示两个 VLM 通道、当前动作、等待队列和 MG400 执行状态。</div>
          </div>
          <span id="monitorUpdated" class="status">等待状态...</span>
        </div>
        <div class="section-body stack">
          <div class="monitor-overview">
            <div class="metric">
              <strong id="activeVlmCount">0</strong>
              <span>活跃 VLM</span>
            </div>
            <div class="metric">
              <strong id="queuedRunCount">0</strong>
              <span>等待任务</span>
            </div>
            <div class="metric">
              <strong id="mg400MonitorState">idle</strong>
              <span>MG400</span>
            </div>
          </div>
          <div id="vlmWorkerGrid" class="worker-grid"></div>
          <div class="mg400-strip">
            <strong>MG400</strong>
            <span id="mg400MonitorAction" class="status">等待状态...</span>
          </div>
          <div>
            <h3>等待队列</h3>
            <div id="vlmQueueList" class="activity-list">
              <div class="activity-row"><span>暂无等待任务</span></div>
            </div>
          </div>
          <div>
            <h3>最近完成</h3>
            <div id="vlmRecentList" class="activity-list">
              <div class="activity-row"><span>暂无完成记录</span></div>
            </div>
          </div>
        </div>
      </section>
    </div>

    <div class="column">
      <section>
        <div class="section-head">
          <div class="section-title">
            <h2>MG400 连接配置</h2>
            <div class="section-note">设置机械臂通信、速度、负载和执行模式。</div>
          </div>
        </div>
        <form id="configForm" class="section-body stack">
          <div class="grid-2">
            <div>
              <label for="mode">执行模式</label>
              <select id="mode">
                <option value="simulation">Simulation / MuJoCo 仿真</option>
                <option value="mg400">MG400 / 真实下发</option>
              </select>
            </div>
            <div>
              <label for="ip">机械臂 IP</label>
              <input id="ip" type="text" placeholder="192.168.2.6">
            </div>
          </div>
          <div class="grid-4">
            <div>
              <label for="dashboardPort">Dashboard</label>
              <input id="dashboardPort" type="number">
            </div>
            <div>
              <label for="motionPort">Motion</label>
              <input id="motionPort" type="number">
            </div>
            <div>
              <label for="speed">速度 %</label>
              <input id="speed" type="number" min="1" max="100">
            </div>
            <div>
              <label for="load">负载 kg</label>
              <input id="load" type="number" step="0.1">
            </div>
          </div>
          <div class="grid-2">
            <div>
              <label for="motionCommand">运动方式</label>
              <select id="motionCommand">
                <option value="MovJ">MovJ</option>
                <option value="MovL">MovL</option>
              </select>
            </div>
            <div>
              <label for="timeoutMs">连接超时 ms</label>
              <input id="timeoutMs" type="number">
            </div>
          </div>
          <label class="row">
            <input id="autoEnable" type="checkbox" style="width:auto">
            自动清错，并在未使能时 EnableRobot
          </label>
          <div class="row">
            <button class="primary" type="submit">保存配置</button>
            <button id="testConnection" type="button">测试连接</button>
            <button id="readStatus" type="button">读取状态</button>
            <span id="configStatus" class="status"></span>
          </div>
          <p class="hint">选择 Simulation 后会自动启动 MuJoCo 并发送手动坐标；选择 MG400 后会下发到真实机械臂。</p>
        </form>
      </section>

      <section>
        <div class="section-head">
          <div class="section-title">
            <h2>本机以太网</h2>
            <div class="section-note">快速查看本机网卡 IP。</div>
          </div>
        </div>
        <div class="section-body stack">
          <div>
            <label for="adapterName">网卡名称</label>
            <input id="adapterName" type="text" value="以太网">
          </div>
          <div class="row">
            <button id="readEthernetInfo" type="button">查看 IP</button>
            <span id="ethernetStatus" class="status"></span>
          </div>
          <pre id="ethernetDetails" class="hidden"></pre>
        </div>
      </section>

      <section>
        <div class="section-head">
          <div class="section-title">
            <h2>Interactive 控制</h2>
            <div class="section-note">常用动作、坐标移动和 Jog 控制。</div>
          </div>
        </div>
        <div class="section-body stack">
          <div class="row">
            <button data-cmd="clearError">清错</button>
            <button data-cmd="enable">使能</button>
            <button data-cmd="disable">下使能</button>
            <button data-cmd="pause">暂停</button>
            <button data-cmd="continue">继续</button>
            <button data-cmd="reset">复位</button>
            <button data-cmd="pose">读位置</button>
          </div>
          <div class="subpanel">
            <h3>手动移动</h3>
            <div class="grid-4">
              <input id="moveX" type="number" placeholder="X">
              <input id="moveY" type="number" placeholder="Y">
              <input id="moveZ" type="number" placeholder="Z">
              <input id="moveR" type="number" placeholder="R">
            </div>
            <div class="row" style="margin-top:10px">
              <button id="moveButton" class="primary" type="button">移动到坐标</button>
              <span class="hint">安全范围：X/Y +/-450，Z +/-250，R +/-360</span>
            </div>
          </div>
          <div class="subpanel">
            <h3>Jog</h3>
            <div class="jog">
              <button data-jog="X+">X+</button><button data-jog="X-">X-</button>
              <button data-jog="Y+">Y+</button><button data-jog="Y-">Y-</button>
              <button data-jog="Z+">Z+</button><button data-jog="Z-">Z-</button>
              <button data-jog="R+">R+</button><button data-jog="R-">R-</button>
            </div>
            <div class="row" style="margin-top:10px">
              <button id="jogStop" class="primary" type="button">停止 Jog</button>
              <span id="interactiveStatus" class="status"></span>
            </div>
          </div>
        </div>
      </section>

      <section>
        <div class="section-head">
          <div class="section-title">
            <h2>运行结果</h2>
            <div class="section-note">展示 Agent 输出、MG400 指令和完整 JSON。</div>
          </div>
        </div>
        <div class="section-body stack">
          <div id="resultSummary" class="hint">提交后显示摘要。</div>
          <div class="row">
          </div>
          <div>
            <h3>VLM Step Timeline</h3>
            <div id="vlmTimeline" class="step-timeline">
              <div class="activity-row"><span>Waiting for VLM events</span></div>
            </div>
          </div>
          <pre id="eventLog" class="hidden"></pre>
          <pre id="result">等待操作...</pre>
        </div>
      </section>
    </div>
  </main>

  <script>
    const resultEl = document.querySelector("#result");
    const resultSummaryEl = document.querySelector("#resultSummary");
    const eventLogEl = document.querySelector("#eventLog");
    const vlmTimelineEl = document.querySelector("#vlmTimeline");
    const modeBadge = document.querySelector("#modeBadge");
    const runStatus = document.querySelector("#runStatus");
    const configStatus = document.querySelector("#configStatus");
    const ethernetStatus = document.querySelector("#ethernetStatus");
    const ethernetDetails = document.querySelector("#ethernetDetails");
    const interactiveStatus = document.querySelector("#interactiveStatus");
    const textMode = document.querySelector("#textMode");
    const voiceMode = document.querySelector("#voiceMode");
    const textPrompt = document.querySelector("#textPrompt");
    const voicePrompt = document.querySelector("#voicePrompt");
    const instructionEl = document.querySelector("#instruction");
    const voiceTranscriptEl = document.querySelector("#voiceTranscript");
    const startVoiceEl = document.querySelector("#startVoice");
    const stopVoiceEl = document.querySelector("#stopVoice");
    const voiceStatusEl = document.querySelector("#voiceStatus");
    const cameraImageEl = document.querySelector("#cameraImage");
    const cameraImageBackEl = document.querySelector("#cameraImageBack");
    const cameraImageBackListEl = document.querySelector("#cameraImageBackList");
    const bitPdfEl = document.querySelector("#bitPdf");
    const bitImageEl = document.querySelector("#bitImage");
    const schematicPdfEl = document.querySelector("#schematicPdf");
    const schematicImageEl = document.querySelector("#schematicImage");
    const openCameraEl = document.querySelector("#openCamera");
    const checkCameraEl = document.querySelector("#checkCamera");
    const captureCameraEl = document.querySelector("#captureCamera");
    const closeCameraEl = document.querySelector("#closeCamera");
    const cameraStatusEl = document.querySelector("#cameraStatus");
    const cameraVideoEl = document.querySelector("#cameraVideo");
    const cameraCanvasEl = document.querySelector("#cameraCanvas");
    const cameraSnapshotEl = document.querySelector("#cameraSnapshot");
    const cameraImageListEl = document.querySelector("#cameraImageList");
    const attachmentListEl = document.querySelector("#attachmentList");
    const vlmWorkerGridEl = document.querySelector("#vlmWorkerGrid");
    const vlmQueueListEl = document.querySelector("#vlmQueueList");
    const vlmRecentListEl = document.querySelector("#vlmRecentList");
    const activeVlmCountEl = document.querySelector("#activeVlmCount");
    const queuedRunCountEl = document.querySelector("#queuedRunCount");
    const mg400MonitorStateEl = document.querySelector("#mg400MonitorState");
    const mg400MonitorActionEl = document.querySelector("#mg400MonitorAction");
    const monitorUpdatedEl = document.querySelector("#monitorUpdated");
    let promptMode = "text";
    let finalTranscript = "";
    let cameraStream = null;
    let capturedCameraImage = null;
    let currentRun = null;
    let selectedMonitorRunId = null;
    let vlmStatusTimer = null;
    let currentRunStatusTimer = null;
    let currentRunEventsTimer = null;
    let currentRunEventsSource = null;
    let currentRunEventSeq = 0;
    let currentTimelineEvents = [];

    const fields = {
      mode: document.querySelector("#mode"),
      ip: document.querySelector("#ip"),
      dashboardPort: document.querySelector("#dashboardPort"),
      motionPort: document.querySelector("#motionPort"),
      speed: document.querySelector("#speed"),
      load: document.querySelector("#load"),
      motionCommand: document.querySelector("#motionCommand"),
      timeoutMs: document.querySelector("#timeoutMs"),
      autoEnable: document.querySelector("#autoEnable")
    };

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognizer = SpeechRecognition ? new SpeechRecognition() : null;

    function show(data) {
      resultEl.textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
    }

    function appendEventLog(event) {
      eventLogEl.classList.remove("hidden");
      const payload = event.payload || {};
      const display = formatEventDisplay(event);
      const parts = [
        "[" + new Date().toLocaleTimeString() + "]",
        display.title,
        display.message
      ];
      if (payload.artifacts?.length) parts.push("产物: " + payload.artifacts.join(" | "));
      const line = parts.filter(Boolean).join(" - ");
      eventLogEl.textContent += (eventLogEl.textContent ? "\n" : "") + line;
      eventLogEl.scrollTop = eventLogEl.scrollHeight;
    }

    function formatEventDisplay(event) {
      const payload = event.payload || {};
      if (payload.title || payload.message) {
        return { title: payload.title || event.type, message: payload.message || "" };
      }
      if (event.type === "agent.assistant") {
        const rawStep = payload.index ?? payload.step;
        const step = Number.isFinite(Number(rawStep)) ? Number(rawStep) + 1 : null;
        const tools = Number(payload.tool_call_count ?? payload.tool_calls?.length ?? 0);
        const elapsed = Number(payload.elapsed_sec || 0).toFixed(1);
        const content = String(payload.content || "").replace(/\s+/g, " ").slice(0, 120);
        return {
          title: step ? "VLM step " + step : "VLM assistant",
          message: tools + " tool call(s), " + elapsed + "s" + (content ? " - " + content : "")
        };
      }
      if (event.type === "tool.started") {
        const toolCall = payload.tool_call || {};
        const rawStep = payload.index ?? payload.step;
        const step = Number.isFinite(Number(rawStep)) ? Number(rawStep) + 1 : null;
        return {
          title: "Tool " + (toolCall.name || payload.name || "started"),
          message: (step ? "Step " + step + " - " : "") + "running"
        };
      }
      if (event.type === "tool.finished") {
        const toolCall = payload.tool_call || {};
        const toolResult = payload.tool_result || {};
        const rawStep = payload.index ?? payload.step;
        const step = Number.isFinite(Number(rawStep)) ? Number(rawStep) + 1 : null;
        const status = (toolResult.ok ?? payload.ok) === false ? "failed" : "ok";
        const final = payload.final || toolResult.is_final || payload.is_final ? ", final" : "";
        const text = String(toolResult.text || payload.text || "").replace(/\s+/g, " ").slice(0, 140);
        return {
          title: "Tool " + (toolCall.name || payload.name || "finished"),
          message: (step ? "Step " + step + " - " : "") + status + final + (text ? " - " + text : "")
        };
      }
      if (event.type === "agent.step") {
        const rawStep = payload.index ?? payload.step;
        const step = Number.isFinite(Number(rawStep)) ? Number(rawStep) + 1 : null;
        const toolCalls = payload.tool_calls || (payload.tool_call ? [payload.tool_call] : []);
        const names = toolCalls.map((item) => item.name).filter(Boolean).join(", ");
        const status = payload.tool_result?.ok === false ? "failed" : "done";
        return {
          title: step ? "VLM step " + step + " " + status : "VLM step " + status,
          message: names || ""
        };
      }
      if (event.type === "agent.waiting") {
        const rawStep = payload.index ?? payload.step;
        const step = Number.isFinite(Number(rawStep)) ? Number(rawStep) + 1 : null;
        return {
          title: step ? "VLM step " + step + " waiting" : "VLM waiting",
          message: "Waiting for model response..."
        };
      }
      if (event.type === "agent.run_dir") {
        return { title: "VLM workspace ready", message: payload.run_dir || payload.workspace || "" };
      }
      if (event.type === "agent.final") {
        return { title: "VLM final answer", message: payload.summary_path || payload.stopped_reason || "" };
      }
      if (event.type === "agent.failed") {
        return { title: "VLM failed", message: payload.error || "" };
      }
      if (event.type === "camera.ultra_close_captured") {
        return { title: "Ultra-close photo", message: payload.path || "" };
      }
      if (event.type === "camera.ultra_vlm_started") {
        return { title: "Ultra VLM refining " + (payload.tp_id || ""), message: payload.message || "" };
      }
      if (event.type === "camera.ultra_vlm_completed") {
        return { title: (payload.tp_id || "TP") + " ultra VLM done",
          message: payload.message + " | artifacts: " + Object.keys(payload.artifacts || {}).join(", ") };
      }
      if (event.type === "camera.ultra_vlm_failed") {
        return { title: "Ultra VLM failed", message: payload.error || payload.message || "" };
      }
      return { title: event.type, message: "" };
    }

    function watchRunStatus(runId) {
      stopRunWatchers();
      currentRunEventSeq = 0;
      currentTimelineEvents = [];
      renderVlmTimeline(currentTimelineEvents);
      eventLogEl.textContent = "";
      eventLogEl.classList.remove("hidden");
      appendEventLog({
        type: "agent.progress",
        payload: {
          title: "VLM run queued",
          message: "Waiting for VLM service events."
        }
      });
      watchRunEvents(runId);

      const refresh = async () => {
        try {
          const run = await jsonFetch("/api/runs/" + encodeURIComponent(runId));
          renderResult(run);
          if (run.report || run.error) {
            stopRunWatchers();
            setStatus(runStatus, run.error ? run.error : "完成", run.error ? "error" : "ok");
            return;
          }
          const state = run.state || run.vlmService?.status || "running";
          setStatus(runStatus, "VLM service running: " + state);
        } catch (error) {
          stopRunWatchers();
          setStatus(runStatus, error.message, "error");
        }
      };

      currentRunStatusTimer = setInterval(refresh, 5000);
      refresh();
    }

    function stopRunWatchers() {
      if (currentRunStatusTimer) clearInterval(currentRunStatusTimer);
      currentRunStatusTimer = null;
      if (currentRunEventsTimer) clearInterval(currentRunEventsTimer);
      currentRunEventsTimer = null;
      if (currentRunEventsSource) currentRunEventsSource.close();
      currentRunEventsSource = null;
    }

    function watchRunEvents(runId) {
      const url = "/api/runs/" + encodeURIComponent(runId) + "/events";
      if (window.EventSource) {
        currentRunEventsSource = new EventSource(url + "?since=0");
        currentRunEventsSource.onmessage = (message) => appendRunEventMessage(message);
        currentRunEventsSource.onerror = () => {
          if (currentRunEventsSource) currentRunEventsSource.close();
          currentRunEventsSource = null;
          startEventPolling(runId);
        };
        return;
      }
      startEventPolling(runId);
    }

    function startEventPolling(runId) {
      if (currentRunEventsTimer) return;
      const poll = async () => {
        try {
          const data = await jsonFetch(
            "/api/runs/" + encodeURIComponent(runId) + "/events?since=" + encodeURIComponent(currentRunEventSeq)
          );
          for (const event of data.events || []) appendRunEvent(event);
        } catch {
          if (currentRunEventsTimer) clearInterval(currentRunEventsTimer);
          currentRunEventsTimer = null;
        }
      };
      currentRunEventsTimer = setInterval(poll, 1000);
      poll();
    }

    function appendRunEventMessage(message) {
      try {
        appendRunEvent(JSON.parse(message.data));
      } catch {
        appendEventLog({
          type: "event.parse_failed",
          payload: { title: "Event parse failed", message: message.data }
        });
      }
    }

    function appendRunEvent(event) {
      currentRunEventSeq = Math.max(currentRunEventSeq, Number(event.seq) || currentRunEventSeq);
      currentTimelineEvents.push(event);
      renderVlmTimeline(currentTimelineEvents);
      appendEventLog(event);
    }

    function renderVlmTimeline(events) {
      const rows = buildVlmTimeline(events || []);
      if (!rows.length) {
        vlmTimelineEl.innerHTML = "<div class='activity-row'><span>Waiting for VLM events</span></div>";
        return;
      }
      vlmTimelineEl.innerHTML = rows.map(renderTimelineRow).join("");
      vlmTimelineEl.scrollTop = vlmTimelineEl.scrollHeight;
    }

    function buildVlmTimeline(events) {
      const rows = [];
      const byStep = new Map();
      const byTool = new Map();
      const stepsWithToolEvents = new Set();

      for (const event of events) {
        const payload = event.payload || {};
        if (event.type === "agent.queued") {
          rows.push(timelineLifecycleRow(event, "queued", "Run queued", payload.task_file || ""));
          continue;
        }
        if (event.type === "agent.started") {
          rows.push(timelineLifecycleRow(event, "running", "Agent started", payload.task_file || ""));
          continue;
        }
        if (event.type === "agent.run_dir") {
          rows.push(timelineLifecycleRow(event, "done", "Workspace ready", payload.run_dir || payload.workspace || ""));
          continue;
        }
        if (event.type === "agent.waiting") {
          const step = displayTimelineStep(payload);
          const key = "step-" + step;
          const row = byStep.get(key) || {
            key,
            step,
            order: timelineStepOrder(payload),
            status: "running",
            title: "Step " + step + " waiting for model",
            tool: "",
            message: "Waiting for model response",
            at: event.timestamp
          };
          row.status = "running";
          row.title = "Step " + step + " waiting for model";
          row.message = "Waiting for model response";
          row.at = row.at || event.timestamp;
          byStep.set(key, row);
          continue;
        }
        if (event.type === "tool.started") {
          const step = displayTimelineStep(payload);
          const toolCall = payload.tool_call || {};
          const toolName = toolCall.name || payload.name || "tool";
          const key = "tool-" + step + "-" + (toolCall.id || toolName || event.seq);
          stepsWithToolEvents.add(String(step));
          byTool.set(key, {
            key,
            step,
            order: timelineStepOrder(payload),
            status: "running",
            title: "Step " + step + " - " + toolName,
            tool: toolName,
            message: "Running" + (toolCall.arguments_preview ? ": " + String(toolCall.arguments_preview).slice(0, 180) : ""),
            at: event.timestamp
          });
          continue;
        }
        if (event.type === "tool.finished") {
          const step = displayTimelineStep(payload);
          const toolCall = payload.tool_call || {};
          const toolResult = payload.tool_result || {};
          const toolName = toolCall.name || payload.name || "tool";
          const key = "tool-" + step + "-" + (toolCall.id || toolName || event.seq);
          stepsWithToolEvents.add(String(step));
          const row = byTool.get(key) || {
            key,
            step,
            order: timelineStepOrder(payload),
            title: "Step " + step + " - " + toolName,
            tool: toolName,
            at: event.timestamp
          };
          const failed = toolResult.ok === false || toolResult.status === "failed";
          const final = Boolean(payload.final || toolResult.is_final);
          row.status = final ? "final" : (failed ? "failed" : "done");
          row.message = timelineToolResultMessage(payload);
          row.finishedAt = event.timestamp;
          byTool.set(key, row);
          continue;
        }
        if (event.type === "agent.step") {
          const step = displayTimelineStep(payload);
          if (stepsWithToolEvents.has(String(step))) continue;
          const key = "step-" + step;
          const tools = timelineToolNames(payload);
          const failed = timelineHasFailedTool(payload);
          const final = Boolean(payload.final);
          const row = byStep.get(key) || { key, step, order: timelineStepOrder(payload), at: event.timestamp };
          row.status = final ? "final" : (failed ? "failed" : "done");
          row.title = "Step " + step + (tools ? " - " + tools : "");
          row.tool = tools;
          row.message = timelineStepMessage(payload);
          row.at = row.at || event.timestamp;
          row.finishedAt = event.timestamp;
          byStep.set(key, row);
          continue;
        }
        if (event.type === "agent.final") {
          rows.push(timelineLifecycleRow(event, "final", "Final answer", payload.summary_path || payload.stopped_reason || ""));
          continue;
        }
        if (event.type === "agent.failed" || event.type === "node.failed") {
          rows.push(timelineLifecycleRow(event, "failed", "Run failed", payload.error || payload.stopped_reason || ""));
          continue;
        }
        // Camera / eye-in-hand pipeline events
        if (event.type === "camera.ultra_close_captured") {
          rows.push(timelineLifecycleRow(event, "running", "Ultra-close photo captured",
            payload.path || ""));
          continue;
        }
        if (event.type === "camera.ultra_vlm_started") {
          rows.push(timelineLifecycleRow(event, "running",
            "VLM refining " + (payload.tp_id || "TP") + " on ultra-close image",
            payload.message || ""));
          continue;
        }
        if (event.type === "camera.ultra_vlm_completed") {
          const imgs = payload.artifacts || {};
          const imgTags = Object.entries(imgs).map(([k, url]) =>
            "<span class='artifact-thumb'><a href='" + escapeAttr(url) + "' target='_blank'>" +
            "<img src='" + escapeAttr(url) + "' loading='lazy' style='max-width:160px;max-height:120px;border:1px solid var(--line);border-radius:4px;margin:2px;vertical-align:top' title='" + escapeAttr(k) + "'></a></span>"
          ).join("");
          rows.push({
            key: event.event_id || event.seq || "ultra-vlm-done",
            status: "done",
            title: (payload.tp_id || "TP") + " ultra VLM refined",
            message: (payload.message || "") + (imgTags ? "<br>" + imgTags : ""),
            imagesHtml: imgTags,
            order: 9700,
            at: event.timestamp
          });
          continue;
        }
        if (event.type === "camera.ultra_vlm_failed") {
          rows.push(timelineLifecycleRow(event, "failed",
            "Ultra VLM refinement failed",
            payload.error || payload.message || ""));
          continue;
        }
      }

      return [...rows, ...byStep.values(), ...byTool.values()]
        .sort((a, b) => {
          const orderDelta = Number(a.order || 0) - Number(b.order || 0);
          if (orderDelta) return orderDelta;
          return Number(a.at || 0) - Number(b.at || 0);
        });
    }

    function timelineLifecycleRow(event, status, title, message) {
      const lifecycleOrder = status === "queued" ? 1
        : status === "running" ? 2
          : status === "done" ? 3
            : status === "final" ? 10000
              : status === "failed" ? 10001
                : 9999;
      return {
        key: event.event_id || event.seq || title,
        status,
        title,
        message,
        order: lifecycleOrder,
        at: event.timestamp
      };
    }

    function renderTimelineRow(row) {
      const cls = row.status === "running" ? " running" : row.status === "failed" ? " failed" : row.status === "final" ? " final" : "";
      const when = row.at ? formatEventTime(row.at) : "";
      const message = [row.message, when].filter(Boolean).join(" | ");
      const imgSection = row.imagesHtml
        ? "<div class='step-images'>" + row.imagesHtml + "</div>"
        : "";
      return [
        "<div class='step-row" + cls + "'>",
        "<div class='step-head'>",
        "<div class='step-title'>" + escapeHtml(row.title || "-") + "</div>",
        "<div class='step-state'>" + escapeHtml(row.status || "-") + "</div>",
        "</div>",
        "<div class='step-meta'>" + escapeHtml(message || "-") + "</div>",
        imgSection,
        "</div>"
      ].join("");
    }

    function displayTimelineStep(payload) {
      const raw = payload.index ?? payload.step;
      const value = Number(raw);
      return Number.isFinite(value) ? value + 1 : "?";
    }

    function timelineStepOrder(payload) {
      const raw = payload.index ?? payload.step;
      const value = Number(raw);
      return Number.isFinite(value) ? 100 + value : 9998;
    }

    function timelineToolNames(payload) {
      const calls = payload.tool_calls || (payload.tool_call ? [payload.tool_call] : []);
      return calls.map((item) => item?.name).filter(Boolean).join(", ");
    }

    function timelineHasFailedTool(payload) {
      const results = payload.tool_results || (payload.tool_result ? [payload.tool_result] : []);
      return results.some((item) => item?.ok === false || item?.status === "failed");
    }

    function timelineStepMessage(payload) {
      const results = payload.tool_results || (payload.tool_result ? [payload.tool_result] : []);
      const first = results.find(Boolean);
      const text = first?.text || first?.feedback || first?.message || first?.error || "";
      if (text) return String(text).replace(/\s+/g, " ").slice(0, 220);
      const calls = payload.tool_calls || (payload.tool_call ? [payload.tool_call] : []);
      if (calls.length) return calls.length + " tool call(s)";
      return payload.final ? "Final tool completed" : "Step completed";
    }

    function timelineToolResultMessage(payload) {
      const result = payload.tool_result || {};
      const duration = Number.isFinite(Number(payload.duration_s || result.duration_s))
        ? Number(payload.duration_s || result.duration_s).toFixed(2) + "s"
        : "";
      const text = result.text || result.feedback || result.message || result.error || "";
      const body = text ? String(text).replace(/\s+/g, " ").slice(0, 220) : "Tool completed";
      return [duration, body].filter(Boolean).join(" - ");
    }

    function formatEventTime(value) {
      const numeric = Number(value);
      const millis = numeric > 100000000000 ? numeric : numeric * 1000;
      return new Date(millis).toLocaleTimeString();
    }

    function showEthernetDetails(data) {
      ethernetDetails.classList.remove("hidden");
      ethernetDetails.textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
    }

    function showProgressPanel(runId) {
      selectedMonitorRunId = runId;
      refreshVlmStatus();
    }

    function removeProgressPanel() {
      selectedMonitorRunId = null;
    }

    function startVlmStatusPolling() {
      if (vlmStatusTimer) clearInterval(vlmStatusTimer);
      vlmStatusTimer = setInterval(refreshVlmStatus, 1000);
      refreshVlmStatus();
    }

    async function refreshVlmStatus() {
      try {
        const data = await jsonFetch("/api/vlm/status");
        renderVlmStatus(data);
      } catch (error) {
        renderVlmStatus({
          ok: false,
          activeCount: 0,
          queueCount: 0,
          workers: [],
          queue: [],
          recent: [],
          mg400: {
            status: "unknown",
            action: "WebUI status service unavailable"
          },
          updatedAt: Date.now()
        });
        monitorUpdatedEl.textContent = "监控读取失败：" + error.message;
      }
    }

    function renderVlmStatus(data) {
      activeVlmCountEl.textContent = String(data.activeCount || 0);
      queuedRunCountEl.textContent = String(data.queueCount || 0);
      mg400MonitorStateEl.textContent = data.mg400?.status || "idle";
      mg400MonitorActionEl.textContent = data.mg400?.action || "MG400 空闲";
      monitorUpdatedEl.textContent = "更新 " + new Date(data.updatedAt || Date.now()).toLocaleTimeString();
      vlmWorkerGridEl.innerHTML = (data.workers || []).map(renderWorkerCard).join("");
      vlmQueueListEl.innerHTML = renderActivityList(data.queue || [], "暂无等待任务");
      vlmRecentListEl.innerHTML = renderActivityList(data.recent || [], "暂无完成记录");
    }

    function renderWorkerCard(worker) {
      const status = worker.status || "idle";
      const isRunning = status !== "idle";
      const selected = worker.runId && worker.runId === selectedMonitorRunId;
      const step = worker.step ? "Step " + worker.step + (worker.maxSteps ? " / Max " + worker.maxSteps : "") : "-";
      const lastTool = worker.lastTool || "-";
      const elapsed = isRunning ? formatDuration(worker.elapsedMs || 0) : "-";
      const runLine = worker.runId
        ? escapeHtml(worker.runId) + (worker.caseId ? " · " + escapeHtml(worker.caseId) : "")
        : "等待新任务";
      return [
        "<div class='worker-card " + (isRunning ? "running " : "") + (status === "failed" ? "failed " : "") + "'>",
        "<div class='worker-head'>",
        "<div class='worker-title'>" + escapeHtml(worker.id || "vlm") + (selected ? " · 当前" : "") + "</div>",
        "<div class='worker-status " + (isRunning ? "running" : "") + (status === "failed" ? " failed" : "") + "'>" + escapeHtml(status) + "</div>",
        "</div>",
        "<div class='worker-run'>" + runLine + "</div>",
        "<div class='worker-action'>" + escapeHtml(worker.action || (isRunning ? "运行中" : "等待新任务")) + "</div>",
        "<div class='worker-meta'>",
        "<div><span>进度</span><strong>" + escapeHtml(step) + "</strong></div>",
        "<div><span>工具</span><strong>" + escapeHtml(lastTool) + "</strong></div>",
        "<div><span>耗时</span><strong>" + escapeHtml(elapsed) + "</strong></div>",
        "</div>",
        "<div class='worker-run'>" + escapeHtml(worker.lastEventType || worker.phase || "") + "</div>",
        "</div>"
      ].join("");
    }

    function renderActivityList(items, emptyText) {
      if (!items.length) return "<div class='activity-row'><span>" + escapeHtml(emptyText) + "</span></div>";
      return items.map((item) => {
        const step = item.step ? " · Step " + item.step + (item.maxSteps ? " / Max " + item.maxSteps : "") : "";
        const tool = item.lastTool ? " · " + item.lastTool : "";
        return [
          "<div class='activity-row'>",
          "<strong>" + escapeHtml(item.runId || "-") + (item.caseId ? " · " + escapeHtml(item.caseId) : "") + "</strong>",
          "<span>" + escapeHtml((item.status || "-") + step + tool + " · " + formatDuration(item.elapsedMs || 0)) + "</span>",
          "<span>" + escapeHtml(item.action || item.lastEventType || "") + "</span>",
          "</div>"
        ].join("");
      }).join("");
    }

    function formatDuration(ms) {
      const total = Math.max(0, Math.floor(Number(ms || 0) / 1000));
      const minutes = Math.floor(total / 60);
      const seconds = total % 60;
      return String(minutes).padStart(2, "0") + ":" + String(seconds).padStart(2, "0");
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function escapeAttr(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;")
        .replace(/</g, "&lt;");
    }

    function setStatus(el, text, kind = "") {
      el.className = "status" + (kind ? " " + kind : "");
      el.textContent = text;
    }

    async function jsonFetch(url, options = {}) {
      const response = await fetch(url, options);
      const data = await response.json();
      if (!response.ok || data.ok === false) {
        const error = new Error(data.error || "请求失败");
        error.message = formatApiError(data);
        error.payload = data;
        throw error;
      }
      return data;
    }

    function formatApiError(data) {
      const details = data?.details || {};
      return [
        data?.error || "Request failed",
        details.category ? "category: " + details.category : "",
        details.action ? "action: " + details.action : "",
        details.workspace ? "workspace: " + details.workspace : "",
        details.runDir ? "runDir: " + details.runDir : "",
        details.summaryPath ? "summary: " + details.summaryPath : "",
        details.baseUrl ? "baseUrl: " + details.baseUrl : "",
        details.status ? "status: " + details.status : ""
      ].filter(Boolean).join("\n");
    }

    function setPromptMode(mode) {
      promptMode = mode;
      textMode.classList.toggle("active", mode === "text");
      voiceMode.classList.toggle("active", mode === "voice");
      textPrompt.classList.toggle("hidden", mode !== "text");
      voicePrompt.classList.toggle("hidden", mode !== "voice");
    }

    function fileSize(bytes) {
      if (!bytes) return "0 B";
      const units = ["B", "KB", "MB", "GB"];
      const power = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
      return (bytes / Math.pow(1024, power)).toFixed(power ? 1 : 0) + " " + units[power];
    }

    function mediaKind(file) {
      const type = String(file?.type || "").toLowerCase();
      const name = String(file?.name || "").toLowerCase();
      if (type.includes("pdf") || name.endsWith(".pdf")) return "PDF";
      if (type.startsWith("image/")) return "Image";
      return "File";
    }

    function renderFiles(target, files, kind) {
      target.innerHTML = "";
      Array.from(files).forEach((file) => {
        const row = document.createElement("div");
        row.className = "file-row";
        row.innerHTML = "<span>" + file.name + "</span><strong class=\"badge\">" + kind + " / " + mediaKind(file) + " / " + fileSize(file.size) + "</strong>";
        target.appendChild(row);
      });
    }

    function dataUrlSize(dataUrl) {
      const base64 = String(dataUrl).split(",")[1] || "";
      return Math.round(base64.length * 0.75);
    }

    function renderCameraImage() {
      cameraImageListEl.innerHTML = "";
      if (capturedCameraImage) {
        const row = document.createElement("div");
        row.className = "file-row";
        row.innerHTML = "<span>" + capturedCameraImage.name + "</span><strong class=\"badge\">Camera_image (正面) / " + fileSize(capturedCameraImage.size) + "</strong>";
        cameraImageListEl.appendChild(row);
        return;
      }
      renderFiles(cameraImageListEl, cameraImageEl.files, "Camera_image (正面)");
    }

    function renderCameraImageBack() {
      cameraImageBackListEl.innerHTML = "";
      renderFiles(cameraImageBackListEl, cameraImageBackEl.files, "Camera_image (背面)");
    }

    async function readFile(file) {
      if (!file) return null;
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve({
          name: file.name,
          type: file.type || "application/octet-stream",
          size: file.size,
          dataUrl: reader.result
        });
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      });
    }

    async function collect(files, kind) {
      const entries = [];
      for (const file of Array.from(files)) {
        entries.push({ kind, ...(await readFile(file)) });
      }
      return entries;
    }

    function renderAttachments() {
      attachmentListEl.innerHTML = "";
      const bitPdfRows = document.createElement("div");
      const bitImageRows = document.createElement("div");
      const schematicPdfRows = document.createElement("div");
      const schematicImageRows = document.createElement("div");
      bitPdfRows.className = "file-list";
      bitImageRows.className = "file-list";
      schematicPdfRows.className = "file-list";
      schematicImageRows.className = "file-list";
      attachmentListEl.append(bitPdfRows, bitImageRows, schematicPdfRows, schematicImageRows);
      renderFiles(bitPdfRows, bitPdfEl.files, "位号图 PDF");
      renderFiles(bitImageRows, bitImageEl.files, "位号图图片");
      renderFiles(schematicPdfRows, schematicPdfEl.files, "原理图 PDF");
      renderFiles(schematicImageRows, schematicImageEl.files, "原理图图片");
    }

    async function checkCameraDevices() {
      if (!navigator.mediaDevices?.enumerateDevices) {
        cameraStatusEl.textContent = "当前浏览器不支持设备检测";
        return [];
      }
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter((device) => device.kind === "videoinput");
      cameraStatusEl.textContent = cameras.length ? "检测到 " + cameras.length + " 个摄像头设备" : "未检测到摄像头设备";
      return cameras;
    }

    async function openCamera() {
      if (!navigator.mediaDevices?.getUserMedia) {
        cameraStatusEl.textContent = "当前浏览器不支持摄像头";
        return;
      }
      try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        cameraVideoEl.srcObject = cameraStream;
        cameraVideoEl.classList.remove("hidden");
        captureCameraEl.disabled = false;
        closeCameraEl.disabled = false;
        openCameraEl.disabled = true;
        cameraStatusEl.textContent = "摄像头已打开";
      } catch (error) {
        const cameras = await checkCameraDevices();
        cameraStatusEl.textContent = "摄像头打开失败：" + error.name + " / " + error.message + "；可见摄像头数：" + cameras.length;
      }
    }

    function closeCamera() {
      if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
      cameraStream = null;
      cameraVideoEl.srcObject = null;
      cameraVideoEl.classList.add("hidden");
      captureCameraEl.disabled = true;
      closeCameraEl.disabled = true;
      openCameraEl.disabled = false;
      cameraStatusEl.textContent = "";
    }

    function captureCamera() {
      if (!cameraStream || !cameraVideoEl.videoWidth || !cameraVideoEl.videoHeight) {
        cameraStatusEl.textContent = "摄像头画面还没有准备好";
        return;
      }
      cameraCanvasEl.width = cameraVideoEl.videoWidth;
      cameraCanvasEl.height = cameraVideoEl.videoHeight;
      const context = cameraCanvasEl.getContext("2d");
      context.drawImage(cameraVideoEl, 0, 0, cameraCanvasEl.width, cameraCanvasEl.height);
      const dataUrl = cameraCanvasEl.toDataURL("image/jpeg", 0.92);
      capturedCameraImage = {
        name: "camera-capture-" + new Date().toISOString().replace(/[:.]/g, "-") + ".jpg",
        type: "image/jpeg",
        size: dataUrlSize(dataUrl),
        dataUrl
      };
      cameraSnapshotEl.src = dataUrl;
      cameraSnapshotEl.classList.remove("hidden");
      cameraImageEl.value = "";
      renderCameraImage();
      cameraStatusEl.textContent = "已拍照";
    }

    function updateBadge(mode) {
      modeBadge.textContent = "机械臂：" + (mode === "mg400" ? "MG400 真实下发" : "MuJoCo 仿真");
      modeBadge.className = "badge" + (mode === "mg400" ? "" : " mock");
    }

    function collectConfig() {
      return {
        mode: fields.mode.value,
        ip: fields.ip.value.trim(),
        dashboardPort: Number(fields.dashboardPort.value),
        motionPort: Number(fields.motionPort.value),
        speed: Number(fields.speed.value),
        load: Number(fields.load.value),
        motionCommand: fields.motionCommand.value,
        timeoutMs: Number(fields.timeoutMs.value),
        autoEnable: fields.autoEnable.checked
      };
    }

    function fillConfig(config) {
      fields.mode.value = config.mode;
      fields.ip.value = config.ip;
      fields.dashboardPort.value = config.dashboardPort;
      fields.motionPort.value = config.motionPort;
      fields.speed.value = config.speed;
      fields.load.value = config.load;
      fields.motionCommand.value = config.motionCommand;
      fields.timeoutMs.value = config.timeoutMs;
      fields.autoEnable.checked = config.autoEnable;
      updateBadge(config.mode);
    }

    function renderResult(run) {
      currentRun = run;
      if (Array.isArray(run.nodeEvents) && run.nodeEvents.length >= currentTimelineEvents.length) {
        currentTimelineEvents = run.nodeEvents.slice();
        renderVlmTimeline(currentTimelineEvents);
      }
      const attachmentCount = run.input.modelAttachments?.length || 0;
      const imageRef = run.vlmObservation?.imageRef || "-";
      const pose = run.modelOutput?.mg400Pose || null;
      const poseText = pose ? "x=" + pose.x + ", y=" + pose.y + ", z=" + pose.z + ", r=" + pose.r : "-";
      const arm = run.execution.arm?.[0] || {};
      const vlmTaskFile = run.vlmAgentCase?.taskFile || "-";
      const vlmCommand = run.vlmAgentCase?.command || "-";
      const vlmRunDir = run.vlmObservation?.runDir || run.modelOutput?.runDir || "-";
      const vlmWorkspace = run.vlmObservation?.workspace || run.modelOutput?.workspace || "-";
      const vlmPrecheck = run.vlmObservation?.precheck || run.modelOutput?.precheck || null;
      const vlmPrecheckText = vlmPrecheck
        ? [
            "env=" + (vlmPrecheck.envFile || "-"),
            "model=" + (vlmPrecheck.model || "-"),
            vlmPrecheck.connectivity?.checked
              ? "connectivity=" + (vlmPrecheck.connectivity.ok === false ? "failed" : "ok")
              : "connectivity=skipped"
          ].join("; ")
        : "-";
      resultSummaryEl.innerHTML =
        "<dl class=\"kv\">" +
        "<div><dt>Run ID</dt><dd>" + run.runId + "</dd></div>" +
        "<div><dt>状态</dt><dd>" + run.state + "</dd></div>" +
        "<div><dt>Camera_image</dt><dd>" + imageRef + "</dd></div>" +
        "<div><dt>模型附件</dt><dd>" + attachmentCount + " 个</dd></div>" +
        "<div><dt>VLM task</dt><dd>" + vlmTaskFile + "</dd></div>" +
        "<div><dt>VLM workspace</dt><dd>" + vlmWorkspace + "</dd></div>" +
        "<div><dt>VLM runDir</dt><dd>" + vlmRunDir + "</dd></div>" +
        "<div><dt>VLM precheck</dt><dd>" + vlmPrecheckText + "</dd></div>" +
        "<div><dt>运行命令</dt><dd>" + vlmCommand + "</dd></div>" +
        "<div><dt>MG400 Pose</dt><dd>" + poseText + "</dd></div>" +
        "<div><dt>机械臂输出</dt><dd>" + (arm.status || "-") + " / " + (arm.tcpCommand || "-") + "</dd></div>" +
        "<div><dt>报告</dt><dd>" + (run.report?.summary || "-") + "</dd></div>" +
        "</dl>";
      show(run);
    }

    if (recognizer) {
      recognizer.lang = "zh-CN";
      recognizer.continuous = true;
      recognizer.interimResults = true;
      recognizer.onstart = () => {
        startVoiceEl.disabled = true;
        stopVoiceEl.disabled = false;
        voiceStatusEl.textContent = "正在听写...";
      };
      recognizer.onresult = (event) => {
        let interimTranscript = "";
        for (let index = event.resultIndex; index < event.results.length; index += 1) {
          const transcript = event.results[index][0].transcript;
          if (event.results[index].isFinal) finalTranscript += transcript.trim() + " ";
          else interimTranscript += transcript;
        }
        voiceTranscriptEl.value = (finalTranscript + interimTranscript).trim();
        instructionEl.value = voiceTranscriptEl.value;
      };
      recognizer.onerror = (event) => {
        voiceStatusEl.textContent = "听写失败：" + event.error;
      };
      recognizer.onend = () => {
        startVoiceEl.disabled = false;
        stopVoiceEl.disabled = true;
        voiceStatusEl.textContent = voiceTranscriptEl.value.trim() ? "听写已停止" : "识别语言：中文普通话";
      };
    } else {
      startVoiceEl.disabled = true;
      stopVoiceEl.disabled = true;
      voiceStatusEl.textContent = "当前浏览器不支持实时语音识别";
    }

    textMode.addEventListener("click", () => setPromptMode("text"));
    voiceMode.addEventListener("click", () => setPromptMode("voice"));
    voiceTranscriptEl.addEventListener("input", () => { instructionEl.value = voiceTranscriptEl.value; });
    startVoiceEl.addEventListener("click", () => {
      if (!recognizer) return;
      finalTranscript = voiceTranscriptEl.value.trim();
      finalTranscript = finalTranscript ? finalTranscript + " " : "";
      recognizer.start();
    });
    stopVoiceEl.addEventListener("click", () => { if (recognizer) recognizer.stop(); });
    cameraImageEl.addEventListener("change", () => {
      capturedCameraImage = null;
      cameraSnapshotEl.classList.add("hidden");
      renderCameraImage();
    });
    cameraImageBackEl.addEventListener("change", () => {
      renderCameraImageBack();
    });
    openCameraEl.addEventListener("click", openCamera);
    checkCameraEl.addEventListener("click", () => checkCameraDevices().catch((error) => {
      cameraStatusEl.textContent = "设备检测失败：" + error.message;
    }));
    captureCameraEl.addEventListener("click", captureCamera);
    closeCameraEl.addEventListener("click", closeCamera);
    bitPdfEl.addEventListener("change", renderAttachments);
    bitImageEl.addEventListener("change", renderAttachments);
    schematicPdfEl.addEventListener("change", renderAttachments);
    schematicImageEl.addEventListener("change", renderAttachments);

    document.querySelector("#runForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      try {
        setStatus(runStatus, "正在读取文件...");
        const textCommand = promptMode === "voice" ? voiceTranscriptEl.value.trim() : instructionEl.value.trim();
        if (!textCommand) throw new Error("请填写 Instruction。");
        if (bitPdfEl.files.length + bitImageEl.files.length === 0) throw new Error("请上传至少一个位号图 PDF 或位号图图片。");
        if (schematicPdfEl.files.length + schematicImageEl.files.length === 0) throw new Error("请上传至少一个原理图 PDF 或原理图图片。");
        const bit = [
          ...await collect(bitPdfEl.files, "bit_image"),
          ...await collect(bitImageEl.files, "bit_image")
        ];
        const schematic = [
          ...await collect(schematicPdfEl.files, "schematic_diagram"),
          ...await collect(schematicImageEl.files, "schematic_diagram")
        ];
        const payload = {
          Instruction: textCommand,
          Case_ID: document.querySelector("#caseId").value.trim(),
          Operator: document.querySelector("#operator").value.trim(),
          Camera_image: capturedCameraImage || await readFile(cameraImageEl.files[0]),
          Camera_image_back: await readFile(cameraImageBackEl.files[0]),
          Target_board_side: "auto",
          Bit_image: bit,
          Schematic_Diagram: schematic
        };
        setStatus(runStatus, "正在运行...");
        const data = await jsonFetch("/api/runs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        setStatus(runStatus, "已提交 · 云端推理中...", "ok");
        renderResult(data);
        if (data.serviceMode && data.runId) {
          showProgressPanel(data.runId);
          watchRunStatus(data.runId);
        }
      } catch (error) {
        setStatus(runStatus, error.message, "error");
        show(error.payload || error.message);
      }
    });

    document.querySelector("#configForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      try {
        setStatus(configStatus, "正在保存...");
        const config = await jsonFetch("/api/mg400/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(collectConfig())
        });
        fillConfig(config);
        setStatus(configStatus, "已保存", "ok");
        show(config);
      } catch (error) {
        setStatus(configStatus, error.message, "error");
      }
    });

    document.querySelector("#testConnection").addEventListener("click", async () => {
      try {
        setStatus(configStatus, "正在测试连接...");
        const data = await jsonFetch("/api/mg400/test", { method: "POST" });
        setStatus(configStatus, "连接成功", "ok");
        show(data);
      } catch (error) {
        setStatus(configStatus, error.message, "error");
        show(error.message);
      }
    });

    document.querySelector("#readStatus").addEventListener("click", async () => {
      try {
        setStatus(configStatus, "正在读取...");
        const data = await jsonFetch("/api/mg400/status");
        setStatus(configStatus, "状态已读取", "ok");
        show(data);
      } catch (error) {
        setStatus(configStatus, error.message, "error");
      }
    });

    document.querySelector("#readEthernetInfo").addEventListener("click", async () => {
      try {
        setStatus(ethernetStatus, "正在读取当前 IP...");
        const name = encodeURIComponent(document.querySelector("#adapterName").value.trim() || "以太网");
        const data = await jsonFetch("/api/ethernet/info?name=" + name);
        setStatus(ethernetStatus, "当前 IP 已读取", "ok");
        show(data);
        showEthernetDetails(data);
      } catch (error) {
        setStatus(ethernetStatus, error.message, "error");
      }
    });

    async function sendCommand(command) {
      const data = await jsonFetch("/api/mg400/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(command)
      });
      show(data);
      return data;
    }

    document.querySelectorAll("[data-cmd]").forEach((button) => {
      button.addEventListener("click", async () => {
        try {
          setStatus(interactiveStatus, "执行中...");
          await sendCommand({ name: button.dataset.cmd });
          setStatus(interactiveStatus, "完成", "ok");
        } catch (error) {
          setStatus(interactiveStatus, error.message, "error");
        }
      });
    });

    document.querySelector("#moveButton").addEventListener("click", async () => {
      try {
        setStatus(interactiveStatus, "移动中...");
        const readMoveNumber = (selector) => {
          const raw = document.querySelector(selector).value.trim();
          return raw === "" ? null : Number(raw);
        };
        await sendCommand({
          name: "move",
          pose: {
            x: readMoveNumber("#moveX"),
            y: readMoveNumber("#moveY"),
            z: readMoveNumber("#moveZ"),
            r: readMoveNumber("#moveR")
          },
          motionCommand: fields.motionCommand.value
        });
        setStatus(interactiveStatus, "移动命令完成", "ok");
      } catch (error) {
        setStatus(interactiveStatus, error.message, "error");
      }
    });

    document.querySelectorAll("[data-jog]").forEach((button) => {
      button.addEventListener("click", async () => {
        try {
          setStatus(interactiveStatus, "Jog " + button.dataset.jog);
          await sendCommand({ name: "jog", axis: button.dataset.jog });
        } catch (error) {
          setStatus(interactiveStatus, error.message, "error");
        }
      });
    });

    document.querySelector("#jogStop").addEventListener("click", async () => {
      try {
        await sendCommand({ name: "jogStop" });
        setStatus(interactiveStatus, "Jog 已停止", "ok");
      } catch (error) {
        setStatus(interactiveStatus, error.message, "error");
      }
    });

    jsonFetch("/api/mg400/config")
      .then(fillConfig)
      .catch((error) => {
        show(error.message);
        updateBadge("simulation");
      });

    startVlmStatusPolling();

  </script>
</body>
</html>`;
