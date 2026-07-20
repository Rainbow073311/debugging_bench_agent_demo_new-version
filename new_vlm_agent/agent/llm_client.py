"""OpenAI-compatible chat client with vision + tool-calling support.

The design goal is that swapping the backend VLM only requires editing
`.env` (`VLM_BASE_URL`, `VLM_API_KEY`, `VLM_MODEL`). Every mainstream
provider — OpenAI, DeepSeek, Qwen/DashScope, Zhipu GLM-4V, Doubao,
Moonshot, local vLLM — exposes an OpenAI-compatible Chat Completions
endpoint, so a single client is enough.

Two tool-calling modes are supported:

* Native `tools=[...]` + `tool_choice="auto"` (preferred, OpenAI-style).
* JSON-tag fallback for providers whose tool protocol is incomplete /
  buggy. The model is asked to emit `<tool_call>{"name":...}</tool_call>`
  which we parse on our side.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI

from .config import Config


# --------------------------------------------------------------------------- #
# Data types exchanged with the agent core
# --------------------------------------------------------------------------- #

@dataclass
class ToolInvocation:
    """A single tool call the model wants the agent runtime to execute."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantReply:
    """Normalised view of one assistant turn.

    `raw_message` is the dict we must append back into `messages` so the
    provider can resolve follow-up `tool` messages against the same
    `tool_call_id`s it emitted. We keep it opaque on purpose.
    """
    content: str
    tool_calls: list[ToolInvocation]
    raw_message: dict[str, Any]
    timing: dict[str, float] | None = None
    usage: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

_TOOL_TAG_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Tool-calling turns rarely need long completions; cap decode budget vs cfg default.
_TOOL_TURN_MAX_TOKENS = 512


def _is_qwen_hybrid_thinking_model(model: str) -> bool:
    """Qwen3.x hybrid models enable thinking by default on DashScope / vLLM."""
    m = (model or "").lower()
    return "qwen3" in m or m.startswith("qwen-plus")


def _materialize_stream(stream: Any) -> Any:
    """Fold an OpenAI-style SSE stream into a ChatCompletion-like response."""
    content: list[str] = []
    tcs: dict[int, dict[str, str]] = {}
    finish: str | None = None
    usage = None
    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        for choice in getattr(chunk, "choices", None) or []:
            finish = choice.finish_reason or finish
            delta = choice.delta
            if not delta:
                continue
            if delta.content:
                content.append(delta.content)
            for tc in delta.tool_calls or []:
                idx = int(getattr(tc, "index", 0) or 0)
                slot = tcs.setdefault(idx, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                fn = tc.function
                if fn and fn.name:
                    slot["name"] = fn.name
                if fn and fn.arguments:
                    slot["args"] += fn.arguments

    class _Fn:
        def __init__(self, name: str, arguments: str):
            self.name = name
            self.arguments = arguments

    class _TC:
        def __init__(self, tc_id: str, name: str, arguments: str):
            self.id = tc_id
            self.type = "function"
            self.function = _Fn(name, arguments)

    class _Msg:
        def __init__(self):
            self.content = "".join(content)
            self.tool_calls = [
                _TC(v["id"], v["name"], v["args"]) for _, v in sorted(tcs.items())
            ] or None
            self.reasoning_content = None

    class _Choice:
        def __init__(self):
            self.message = _Msg()
            self.finish_reason = finish

    class _Resp:
        def __init__(self):
            self.choices = [_Choice()]
            self.usage = usage

    return _Resp()


def _dedupe_finish_calls(invocations: list[ToolInvocation]) -> list[ToolInvocation]:
    """If the model emits multiple ``finish`` tags in one turn, keep the last one."""
    finishes = [i for i in invocations if i.name == "finish"]
    if len(finishes) <= 1:
        return invocations
    others = [i for i in invocations if i.name != "finish"]
    return others + [finishes[-1]]


def _json_load_with_auto_closers(raw: str) -> dict[str, Any] | None:
    """Best-effort parse for mildly truncated JSON object blocks.

    Handles the common case where the model omits one or more trailing
    `}` / `]` at the end of a `<tool_call>...</tool_call>` payload.
    """
    text = (raw or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    stack: list[str] = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            stack.append("}")
            continue
        if ch == "[":
            stack.append("]")
            continue
        if ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                return None
    if in_str:
        return None
    if not stack:
        return None
    repaired = text + "".join(reversed(stack))
    try:
        obj = json.loads(repaired)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _repair_common_tool_call_json(raw: str) -> str:
    """Repair common malformed fallback tool JSON emitted by some models.

    Example repaired:
      {"name":"read_text_file", {"path":"..."}}
      -> {"name":"read_text_file","arguments":{"path":"..."}}
    """
    txt = (raw or "").strip()
    if not txt:
        return txt
    # Missing `arguments` key after `name`/`tool`.
    txt = re.sub(
        r'(\{\s*"(?:name|tool)"\s*:\s*"[^"]+"\s*),\s*\{',
        r'\1, "arguments": {',
        txt,
        flags=re.DOTALL,
    )
    return txt


def _parse_tagged_tool_calls(content: str) -> list[ToolInvocation]:
    """Parse ``<tool_call>{...}</tool_call>`` blocks.

    The previous regex used ``\\{.*?\\}``, which stops at the *first* ``}`` and
    breaks nested JSON (e.g. ``finish`` with ``arguments.answer``).
    """
    invocations: list[ToolInvocation] = []
    for i, m in enumerate(_TOOL_TAG_BLOCK_RE.finditer(content)):
        raw = (m.group(1) or "").strip()
        if not raw:
            continue
        payload = _json_load_with_auto_closers(raw)
        if not payload:
            payload = _json_load_with_auto_closers(_repair_common_tool_call_json(raw))
        if not payload:
            continue
        name = payload.get("name") or payload.get("tool")
        args = payload.get("arguments") or payload.get("args")
        if args is None:
            # Graceful fallback for payloads like:
            # {"name":"read_text_file","path":"..."}
            args = {
                k: v for k, v in payload.items()
                if k not in {"name", "tool", "arguments", "args"}
            }
        if args is None:
            args = {}
        if not name:
            continue
        invocations.append(ToolInvocation(
            id=f"tag_{i}",
            name=name,
            arguments=args if isinstance(args, dict) else {"_raw": args},
        ))
    if not invocations:
        invocations = _parse_unclosed_tool_call_tag(content)
    return _dedupe_finish_calls(invocations)


def _parse_unclosed_tool_call_tag(content: str) -> list[ToolInvocation]:
    """Recover ``<tool_call>{...}`` when the model omits ``</tool_call>``."""
    m = re.search(r"<tool_call>\s*(\{.*\})\s*$", content or "", re.DOTALL)
    if not m:
        return []
    raw = (m.group(1) or "").strip()
    if not raw:
        return []
    payload = _json_load_with_auto_closers(raw)
    if not payload:
        payload = _json_load_with_auto_closers(_repair_common_tool_call_json(raw))
    if not payload:
        return []
    name = payload.get("name") or payload.get("tool")
    args = payload.get("arguments") or payload.get("args")
    if args is None:
        args = {
            k: v for k, v in payload.items()
            if k not in {"name", "tool", "arguments", "args"}
        }
    if args is None:
        args = {}
    if not name:
        return []
    return [ToolInvocation(
        id="tag_unclosed_0",
        name=name,
        arguments=args if isinstance(args, dict) else {"_raw": args},
    )]


class LLMClient:
    """Thin wrapper around `openai.OpenAI` with multimodal helpers."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.http_timeout_sec,
            max_retries=cfg.http_max_retries,
        )

    # ------------------------------------------------------------------ #
    # Message builders — kept here so tests and the agent share them.
    # ------------------------------------------------------------------ #

    @staticmethod
    def text_part(text: str) -> dict[str, Any]:
        return {"type": "text", "text": text}

    @staticmethod
    def image_part(data_url: str, detail: str = "high") -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": data_url, "detail": detail},
        }

    @staticmethod
    def user_message(parts: Iterable[dict[str, Any]] | str) -> dict[str, Any]:
        if isinstance(parts, str):
            return {"role": "user", "content": parts}
        return {"role": "user", "content": list(parts)}

    @staticmethod
    def system_message(text: str) -> dict[str, Any]:
        return {"role": "system", "content": text}

    @staticmethod
    def tool_result_message(tool_call_id: str,
                            content: str | list[dict[str, Any]]) -> dict[str, Any]:
        """Build a `role=tool` message. Some providers only accept strings,
        others accept multimodal parts; we default to string but allow a
        list for providers that tolerate images inside tool results.
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def chat(self, messages: list[dict[str, Any]],
             tools_schema: list[dict[str, Any]] | None = None) -> AssistantReply:
        """Call the underlying chat.completions endpoint once.

        `tools_schema` follows the OpenAI tool schema
        (`[{"type": "function", "function": {"name":..., "parameters":...}}, ...]`).
        """
        chat_t0 = time.time()
        max_tokens = self.cfg.max_tokens
        if tools_schema:
            max_tokens = min(max_tokens, _TOOL_TURN_MAX_TOKENS)
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.cfg.use_native_tools and tools_schema:
            kwargs["tools"] = tools_schema
            kwargs["tool_choice"] = "auto"

        # --- thinking / reasoning controls ---------------------------- #
        # 1) DeepSeek V4: extra_body={"thinking":{"type":"enabled"}}
        # 2) Intern-S1 family (LMDeploy Chat API): extra_body.thinking_mode=<bool>
        # 3) Qwen3 hybrid (DashScope / vLLM): enable_thinking defaults ON — disable for low-latency tool turns.
        # 4) Optional reasoning_effort for providers that support it.
        model_lower = self.cfg.model.lower()
        extra_body: dict[str, Any] = {}

        if self.cfg.enable_thinking:
            extra_body["thinking"] = {"type": "enabled"}
        elif _is_qwen_hybrid_thinking_model(model_lower):
            extra_body["enable_thinking"] = False
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        if model_lower.startswith("intern-s1") and self.cfg.thinking_mode is not None:
            extra_body["thinking_mode"] = bool(self.cfg.thinking_mode)

        if extra_body:
            kwargs["extra_body"] = extra_body

        # Intern-S1 docs emphasise `thinking_mode`; avoid sending an
        # unsupported `reasoning_effort` there unless explicitly needed.
        if self.cfg.reasoning_effort and not model_lower.startswith("intern-s1"):
            kwargs["reasoning_effort"] = self.cfg.reasoning_effort

        prep_dt = time.time() - chat_t0
        resp = None
        attempts = max(1, int(self.cfg.connect_retries))
        retry_sleep_s = 0.0
        api_call_s = 0.0
        for attempt in range(attempts):
            try:
                api_t0 = time.time()
                raw = self._client.chat.completions.create(**kwargs)
                if hasattr(raw, "choices"):
                    resp = raw
                else:
                    resp = _materialize_stream(raw)
                api_call_s += time.time() - api_t0
                break
            except BadRequestError as e:
                api_call_s += time.time() - api_t0
                body = str(e)
                if "unknown variant `image_url`, expected `text`" in body:
                    raise RuntimeError(
                        "Current model/provider rejected inline image blocks "
                        "(`image_url`). This usually means the selected model is "
                        "text-only. For DeepSeek, switch to a vision-capable model "
                        "(e.g. a `...vl...` / `...vision...` model) or keep the "
                        "current model and rely on tool-driven image processing only."
                    ) from e
                if "fc related" in body.lower() or "-20009" in body:
                    raise RuntimeError(
                        "This OpenAI-compatible endpoint rejected native function-calling "
                        "(`tools` / `tool_choice`). Intern chat gateways sometimes return "
                        "code -20009 / 'Failed to parse fc related info…' when the model "
                        "or route does not accept the advertised tool schema.\n\n"
                        "Try:\n"
                        "  • Set VLM_USE_NATIVE_TOOLS=false in .env, or run:\n"
                        "      python -m agent run <task.yaml> --no-native-tools\n"
                        "    (uses <tool_call>…</tool_call> in the assistant text instead.)\n"
                        "  • Or switch to a model/endpoint on that platform that explicitly "
                        "supports OpenAI-style tools for your account."
                    ) from e
                raise
            except (APIConnectionError, APITimeoutError):
                api_call_s += time.time() - api_t0
                if attempt + 1 >= attempts:
                    raise
                sleep_s = min(4.0 * (2 ** attempt), 60.0)
                retry_sleep_s += sleep_s
                time.sleep(sleep_s)
        if resp is None:  # pragma: no cover
            raise RuntimeError("LLMClient.chat: failed without response")

        parse_t0 = time.time()
        usage = getattr(resp, "usage", None)
        completion_tokens = 0
        if usage is not None:
            completion_tokens = int(
                getattr(usage, "completion_tokens", 0)
                or (
                    usage.get("completion_tokens", 0)
                    if isinstance(usage, dict)
                    else 0
                )
                or 0
            )
        usage_payload: dict[str, Any] = {}
        if usage is not None:
            if hasattr(usage, "model_dump"):
                try:
                    usage_payload = usage.model_dump()
                except Exception:  # noqa: BLE001
                    usage_payload = {}
            elif isinstance(usage, dict):
                usage_payload = dict(usage)
            else:
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if hasattr(usage, key):
                        try:
                            usage_payload[key] = int(getattr(usage, key))
                        except Exception:  # noqa: BLE001
                            pass
        # Coarse estimate only: decode phase scales with completion tokens.
        # This is NOT provider-ground-truth latency breakdown.
        est_tok_per_s = 45.0
        est_decode_s = min(float(api_call_s), float(completion_tokens) / est_tok_per_s)
        est_prefill_s = max(0.0, float(api_call_s) - est_decode_s)

        def _timing_payload(parse_started_at: float) -> dict[str, float]:
            return {
                "prep_s": round(prep_dt, 4),
                "api_call_s": round(api_call_s, 4),
                "retry_sleep_s": round(retry_sleep_s, 4),
                "parse_s": round(time.time() - parse_started_at, 4),
                "llm_total_s": round(time.time() - chat_t0, 4),
                "completion_tokens": float(completion_tokens),
                "est_prefill_s": round(est_prefill_s, 4),
                "est_decode_s": round(est_decode_s, 4),
            }

        choice = resp.choices[0]
        msg = getattr(choice, "message", None)

        # Some OpenAI-compatible providers occasionally return a choice
        # whose `message` is null (or malformed) when multimodal/tool
        # turns are mixed. Do not crash; degrade gracefully.
        if msg is None:
            fallback_content = ""
            if hasattr(choice, "text") and isinstance(choice.text, str):
                fallback_content = choice.text
            elif hasattr(choice, "model_dump"):
                try:
                    dumped = choice.model_dump()
                    maybe_text = dumped.get("text")
                    if isinstance(maybe_text, str):
                        fallback_content = maybe_text
                except Exception:  # noqa: BLE001
                    pass
            fallback_content = (fallback_content or "").strip()
            if not fallback_content:
                fallback_content = (
                    "[provider-warning] assistant message is null; "
                    "continue with tools or call finish."
                )
            return AssistantReply(
                content=fallback_content,
                tool_calls=[],
                raw_message={"role": "assistant", "content": fallback_content},
                timing=_timing_payload(parse_t0),
                usage=usage_payload or None,
            )

        content = msg.content or ""
        tool_calls: list[ToolInvocation] = []
        raw_message: dict[str, Any] = {"role": "assistant", "content": content}

        # DeepSeek's docs state: when the assistant emitted tool calls,
        # subsequent turns MUST re-send `reasoning_content` or the API
        # returns 400. We therefore attach it to raw_message; the agent
        # loop appends raw_message verbatim to `messages`, so it will
        # round-trip to the provider unchanged. Providers that do not
        # recognise the field typically ignore extra JSON keys.
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            raw_message["reasoning_content"] = reasoning

        # Native tool_calls path ---------------------------------------- #
        native_calls = getattr(msg, "tool_calls", None) or []
        if native_calls:
            raw_tc: list[dict[str, Any]] = []
            for tc in native_calls:
                args_raw = tc.function.arguments or "{}"
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {"_raw": args_raw}
                tool_calls.append(ToolInvocation(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
                raw_tc.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": args_raw if isinstance(args_raw, str) else json.dumps(args_raw),
                    },
                })
            raw_message["tool_calls"] = raw_tc

        # JSON-tag fallback --------------------------------------------- #
        # Only parse tags when the provider did NOT emit native tool_calls,
        # otherwise we'd double-execute the same call.
        elif content:
            tool_calls = _parse_tagged_tool_calls(content)

        return AssistantReply(
            content=content,
            tool_calls=tool_calls,
            raw_message=raw_message,
            timing=_timing_payload(parse_t0),
            usage=usage_payload or None,
        )
