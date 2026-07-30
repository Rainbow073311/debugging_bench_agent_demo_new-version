# Debugging Bench Agent

Inputdemo 是调试平台的 HTTP/API 与网页入口。生产默认路径已经接入真实 `Debugging-agent-v2` VLM agent，不再默认返回 mock VLM 或 mock large model 结果。

```text
Instruction + Case Metadata + Camera/Bit/Schematic Files
  -> LLM Input Core
  -> YAML Builder
  -> Debugging-agent-v2 task.yaml
  -> Real VLM agent / real model service
  -> Agent Plan
  -> MG400 or Simulation execution
  -> GUI/Report Result
```

## Runtime VLM

`npm start` / `POST /api/runs` 会使用：

- `src/adapters/realVlmAgentRunner.js`
- `src/adapters/realVlmAgentClient.js`
- `../Vlm agent/Debugging-agent-v2`

真实模型配置从 `Vlm agent/Debugging-agent-v2/.env` 读取，必须包含：

```env
VLM_BASE_URL=...
VLM_API_KEY=...
VLM_MODEL=...
```

也可以用环境变量覆盖：

- `VLM_AGENT_DIR`: 指向 `Debugging-agent-v2` 目录
- `VLM_ENV_FILE`: 指向要使用的 `.env`
- `VLM_AGENT_MAX_STEPS`: 单次 agent 最大步数，默认 `80`
- `VLM_AGENT_TIMEOUT_MS`: Node 等待真实 agent 的超时时间

如果这些配置缺失，接口会失败并提示配置真实 VLM 服务，而不会静默返回 mock 数据。

## Project Structure

```text
src/
  adapters/       外部能力接口实现；生产默认使用 realVlmAgent*，mock adapter 仅保留给测试
  agent/          Agent 编排与状态流转
  api/            HTTP API 服务与内置前端页面
  cli/            本地端到端演示入口
  domain/         数据结构、状态、错误类型
  utils/          通用工具
docs/
  api.md          API 契约说明
test/
  pipeline.test.js
```

## Run

```bash
npm start
```

默认监听 `http://localhost:3000`。打开 `/` 可以提交：

1. `Instruction`
2. `Case ID`
3. `Operator`
4. `Camera_image`
5. `Bit_image`
6. `Schematic_Diagram`

## Test

```bash
npm test
```

## Oscilloscope selection

RTO6 remains the default. Its existing configuration and bridge are preserved:

```env
OSCILLOSCOPE_DRIVER=rto6
```

To select the parallel Keysight DSOX1204G adapter:

```powershell
python -m pip install -r requirements-instruments.txt
$env:OSCILLOSCOPE_DRIVER="dsox1204g"
npm start
```

The DSOX1204G adapter reads `config/dsox1204g.json`, communicates over
VXI-11, records the current source's display-interval average voltage, captures
a PNG screen image, and creates a separate Excel report under
`outputs/dsox1204g_measurements`.

测试仍使用 mock adapter 作为离线夹具，避免单元测试消耗真实模型调用。
