# PCBA Debugging Bench — 架构与 Workflow（2026-08）

> 基于当前仓库真实代码整理：Inputdemo 编排、VLM Agent、J1 眼在手标定、ultra TP 精修、tip_offset。
> 共 **8** 张 Mermaid 图。旧版六月文档见 `真实架构与流程-基于当前代码.md`（部分内容已过时，以本文为准）。

---

## 图 1 — 整仓系统架构

```mermaid
flowchart TB
  subgraph UI["用户交互 :3000"]
    Browser["浏览器 Web UI<br/>webPage.js"]
  end

  subgraph Node["Inputdemo 编排层 :3000"]
    Server["server.js<br/>runs / SSE / states"]
    InputCore["inputCore + caseAdapter"]
    Mapper["vlmTargetExecutionMapper<br/>calibratedVisionTarget"]
    TipJS["tipOffset.js"]
    CamCtrl["eyeInHandCameraController.js"]
  end

  subgraph VLM["VLM 智能体"]
    NewVlm["new_vlm_agent / Debugging-agent-v2<br/>Part 0→B→A→C→D → step08"]
  end

  subgraph Calib["标定资产 calibration/"]
    CamYaml["camera_config.yaml<br/>T_ec + J1 model LOCKED"]
    TipJson["tip_offset.json"]
    CamCtrJson["camera_center_offset.json"]
    Xforms["eye_in_hand_xyz.py<br/>coordinate_transforms.py"]
    Ultra["vlm_refine_ultra.py"]
  end

  subgraph HW["硬件桥"]
    GW["robotGatewayServer.js :8010"]
    Bridge["mg400_bridge.py"]
    CamPy["eye_in_hand_camera.py<br/>Basler + PixelToWorld"]
    Scope["示波器 / MockEquipment"]
  end

  Browser -->|HTTP/SSE| Server
  Server --> InputCore
  InputCore -->|task.yaml + 图像| NewVlm
  NewVlm -->|TP pixel / step08| Mapper
  Mapper --> TipJS
  Mapper --> CamCtrl
  CamCtrl --> CamPy
  CamPy --> Xforms
  Xforms --> CamYaml
  Server -->|ultra refine| Ultra
  Ultra --> CamYaml
  TipJS --> TipJson
  CamCtrl --> CamCtrJson
  Server -->|MOVE / probe| GW
  GW --> Bridge
  Bridge --> MG400["MG400 机械臂"]
  Server --> Scope
```

---

## 图 2 — 端到端主 Workflow

```mermaid
sequenceDiagram
  actor Op as 工程师
  participant UI as Web UI
  participant S as server.js
  participant V as VLM Agent
  participant C as eye_in_hand_camera
  participant U as vlm_refine_ultra
  participant G as Robot Gateway
  participant R as MG400

  Op->>UI: 提交 Instruction + Case + 附件
  UI->>S: POST /api/runs
  S->>S: IDLE → PREPARING
  S->>V: 创建 case / 调用 VLM service 或 CLI
  V->>V: Part 0→B→A→C→D 定位 TP
  V-->>S: step08 / final_answer (pixel)
  S->>S: EXECUTING
  S->>C: global/close 捕获 + pixelToBase
  C-->>S: basePoint (table XY)
  S->>G: 移到 ultra 观察位 (Z≈-115)
  G->>R: MovJ
  S->>C: ultra_close 拍照
  S->>U: VLM text + OpenCV pad above text
  U-->>S: pad_pixel → 再投影 basePoint
  S->>S: tipTargetToTcpPose (tip_offset)
  S->>G: tip hover → 探针下降
  G->>R: 运动 + 阈值/接触
  S->>S: REPORTING → 报告 + SSE
  S-->>UI: 结果事件
```

---

## 图 3 — ① 标定总流程

```mermaid
flowchart LR
  A["1. 内参<br/>Basler K,D"] --> B["2. 外参 T_ec<br/>J1 end-frame Kabsch"]
  B --> C["3. tip_offset<br/>r, δ 切向模型"]
  C --> D["4. camera_center_offset<br/>光心 LOOK 切向"]
  D --> E["5. 锁定写入<br/>camera_config.yaml<br/>+ tip/cam json"]
  E --> F["运行时共用<br/>pixelToTable / tip / LOOK"]

  style E fill:#d4edda
  style F fill:#cce5ff
```

| 量 | 模型 | 锁定文件 |
|---|---|---|
| `T_end_to_camera` | `T_base_cam = Trans(TCP) @ Rz(J1) @ T_ec` | `calibration/camera_config.yaml` |
| `tip_offset` | `tip = TCP + r·(sin(J1+δ), −cos(J1+δ))` | `tip_offset.json` |
| `camera_center_offset` | 同形式，独立 r/δ | `camera_center_offset.json` |

三者独立，互不代入。

---

## 图 4 — ①a 外参 / J1 标定细节

```mermaid
flowchart TB
  T0["approach-tl / jog tip"] --> T1["touch-tl → P_TL<br/>tip_offset 反解 TCP"]
  T1 --> L["LOOK L1..Ln<br/>变 J1 ≥~15–20°"]
  L --> D["physical TL 检测<br/>Camera-title LEFT"]
  D --> P["PnP → tl_cam"]
  P --> K["Kabsch in J1 frame<br/>y = Rz(-J1)@(P_TL−TCP)<br/>= R_ec @ tl_cam + t_ec"]
  K --> V["validate: pixelToTable<br/>+ tip hover on TL"]
  V -->|yes| Lock["lock T_ec + sync VLM yaml"]
  V -->|no| Restore["restore bak / more LOOK"]

  subgraph Runtime["运行时"]
    RT["pixel_to_table(u,v,pose)<br/>Trans(TCP) @ Rz(J1) @ T_ec<br/>∩ table_z"]
  end
  Lock --> RT
```

主脚本：`calibrate_extrinsics_recal.py` / `calibrate_camera_j1_offset.py`；核心几何：`eye_in_hand_xyz.compose_base_to_camera`。

---

## 图 5 — ② VLM TP 定位流程

```mermaid
flowchart TB
  IN["inputs: 意图 + 原理图 + 位号 PDF + 板图<br/>workflow_doc + skills_doc"] --> P0["Part 0<br/>PDF 搜位号 / 栅格"]
  P0 --> PB["Part B<br/>原理图 → 信号/TP"]
  PB --> PA["Part A<br/>装配图最大 IC 等"]
  PA --> PC["Part C<br/>板图对齐准备"]
  PC --> PD["Part D<br/>case12 OpenCV IC 对齐<br/>或 case10 dual ROI"]
  PD --> S8["Step8 / finish<br/>board_tp_marked<br/>step08_final_tp.png"]
  S8 --> OUT["TP pixel + confidence<br/>→ Node mapper"]

  style S8 fill:#d4edda
```

规程：`new_vlm_agent/data/skills/STANDARD_WORKFLOW.md` + `SKILL.md`。

---

## 图 6 — ③ 眼在手捕获 + Ultra 精修

```mermaid
flowchart TB
  G0["global capture<br/>Z≈50 / safe"] --> Loc["板定位 / PCB center"]
  Loc --> Close["close capture<br/>camera_center_offset 移光心"]
  Close --> Coarse["close 图 VLM → 粗 TP pixel<br/>pixelToBase → basePoint"]
  Coarse --> UltraMove["TCP ≈ P − Δ_cam(J1)<br/>Z_ultra ≈ −115"]
  UltraMove --> Shot["ultra_close 拍照"]
  Shot --> VLM2["vlm_refine_ultra.py<br/>① 粗点投到 ultra 像素<br/>② ROI 内 VLM 找 TPn text<br/>③ OpenCV 对比度滤假圆<br/>④ 选文字上方略偏右 pad"]
  VLM2 --> Reproj["pad_pixel → pixelToTable<br/>更新 basePoint"]
  Reproj --> Tip["tip_offset → TCP hover<br/>再探针下降"]

  style VLM2 fill:#fff3cd
  style Tip fill:#d4edda
```

关键：`finalizeSplitServiceRun`（`server.js`）+ `calibration/vlm_refine_ultra.py`。

---

## 图 7 — ④ 运行时坐标变换

```mermaid
flowchart LR
  subgraph PixelPath["任意像素 → 桌面"]
    UV["(u,v)"] --> Ray["undistort → 相机射线"]
    Ray --> Tbc["T_base_cam =<br/>Trans(TCP) @ Rz(J1) @ T_ec"]
    Tbc --> Hit["∩ table_z → (X,Y,Z_table)"]
  end

  subgraph TipPath["尖点命令"]
    Ptip["目标 tip XY"] --> TipOff["Δ_tip(J1)<br/>r=27, δ=6.15°"]
    TipOff --> TCP1["TCP = P_tip − Δ_tip"]
  end

  subgraph CamPath["光心 LOOK"]
    Pcam["目标桌面点"] --> CamOff["Δ_cam(J1)<br/>r=34, δ=−164.7°"]
    CamOff --> TCP2["TCP = P − Δ_cam"]
  end

  Hit --> Ptip
  Hit --> Pcam
```

J1 来源：`pose.j1_deg` 优先，否则 `atan2(TCP_y, TCP_x)`。法兰角 `R`(J4) 不参与相机挂载。

---

## 图 8 — ⑤ Node 状态机与模块调用

```mermaid
stateDiagram-v2
  [*] --> IDLE
  IDLE --> PREPARING: POST /api/runs
  PREPARING --> EXECUTING: VLM 完成 / split 子 run 就绪
  EXECUTING --> REPORTING: 运动+量测结束或失败收尾
  REPORTING --> IDLE: 报告完成

  state PREPARING {
    [*] --> AdaptCase
    AdaptCase --> CallVLM
    CallVLM --> WaitStep08
  }

  state EXECUTING {
    [*] --> ProjectPixel
    ProjectPixel --> EyeInHand
    EyeInHand --> UltraRefine
    UltraRefine --> TipHover
    TipHover --> ProbeDescend
    ProbeDescend --> Measure
  }
```

```mermaid
flowchart TB
  S["server.js"] --> Store["runStore + SSE events"]
  S --> States["states.js<br/>IDLE/PREPARING/EXECUTING/REPORTING"]
  S --> Adapt["vlmAgentCaseAdapter"]
  S --> Runner["vlmAgentServiceRunner<br/>/ remote / CLI"]
  S --> Final["finalizeSplitServiceRun"]
  Final --> Vision["calibratedVisionTarget<br/>pixelToBase"]
  Final --> UltraPy["spawn vlm_refine_ultra.py"]
  Final --> Tip["tipOffset.js"]
  Final --> Reach["mg400Reachability"]
  Final --> GW["robotGatewayClient → :8010"]
  Final --> Eq["equipmentController"]
```

---

## 端口与启动（速查）

| 服务 | 命令 | 端口 |
|---|---|---|
| Web + API | `cd Inputdemo && npm start` | 3000 |
| Robot Gateway | `cd Inputdemo && npm run robot-gateway` | 8010 |
| VLM Service | `python -m agent.service`（视部署） | 8000（常见） |

标定验证示例：

```bash
python calibration/_find_tl_z110_now.py -120
python calibration/vlm_refine_ultra.py <ultra.jpg> <x,y,z> <x,y,z,r> TP9
```

---

## 相关代码入口

| 主题 | 路径 |
|---|---|
| 编排 | `Inputdemo/src/api/server.js` |
| J1 几何 | `calibration/eye_in_hand_xyz.py` |
| 像素投影 | `calibration/coordinate_transforms.py` |
| 锁定外参 | `calibration/camera_config.yaml` |
| Ultra pad | `calibration/vlm_refine_ultra.py` |
| Tip | `calibration/tip_offset.py` / `Inputdemo/src/domain/tipOffset.js` |
| VLM 规程 | `new_vlm_agent/data/skills/STANDARD_WORKFLOW.md` |
