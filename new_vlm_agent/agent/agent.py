"""Agent core: the plan → tool → observe loop.

Design goals:

* Keep the loop provider-agnostic. Anything VLM-specific sits inside
  `LLMClient`; anything task-specific sits inside `prompts.py`. The
  loop here just orchestrates message building, tool dispatch and
  trace logging.
* When a tool produces an image (e.g. a crop or an annotated capture),
  we attach it as an image part in the NEXT user turn, because most
  OpenAI-compatible providers reject multimodal content inside
  `role=tool` messages.
* Every run writes a JSONL trace under `workspace/runs/<timestamp>/`
  so training / evaluation scripts can replay conversations offline.
* Optional ``phase_isolated_context`` (default on with hierarchical mode): each
  plan phase group (part0/partB/partA/partD) resets ``messages[]`` for the same
  VLM API call — handoff is artifact paths + compact inputs + full planner-step.
"""

# Efficiency Change Log (training_platform)
# 2026-06-04 | phase_isolated_context: per-phase fresh messages, JSON plan unchanged,
#   single VLM API; handoff = artifact paths + key facts + full next_action_hint.

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from rich.console import Console
from rich.panel import Panel

from .builtin_tools import build_default_registry, set_runtime_context
from .config import Config
from .llm_client import AssistantReply, LLMClient, ToolInvocation
from .post_run_reflect import run_post_run_reflection
from .prompts import (
    CLI_WORKFLOW_MODE_VLM_TEST_APPEND_ZH,
    FALLBACK_TOOL_PROTOCOL,
    SYSTEM_PROMPT_TP_LOCATE,
)
from .tools import ToolRegistry, ToolResult, normalize_finish_arguments
from .utils import encode_image_data_url, truncate


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class AgentStep:
    index: int
    assistant_content: str
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    timing: dict[str, float] = field(default_factory=dict)
    tool_timing: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRun:
    task_question: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: Any = None
    stopped_reason: str = "unfinished"
    run_dir: Path | None = None
    # Set when the run aborts on an uncaught exception (still persisted).
    last_error: str | None = None
    part_timing: dict[str, float] = field(default_factory=dict)


@dataclass
class WorkflowPlanStep:
    step_id: str
    title: str
    objective: str
    done_any_artifacts: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    next_action_hint: str = ""


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class Agent:
    def __init__(self,
                 cfg: Config,
                 registry: ToolRegistry | None = None,
                 console: Console | None = None,
                 system_prompt: str | None = None,
                 event_sink: Any = None) -> None:
        self.cfg = cfg
        self.cfg.ensure_workspace()
        self.registry = registry or build_default_registry(cfg.workspace_dir)
        self.client = LLMClient(cfg)
        self.console = console or Console()
        self._event_sink = event_sink
        self.system_prompt = system_prompt or SYSTEM_PROMPT_TP_LOCATE
        self._initial_task_compacted = False
        self._run_inputs: dict[str, Any] = {}
        self._part0_mark_tp_calls = 0

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink(event_type, payload)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, question: str,
            inputs: dict[str, Any] | None = None,
            run_name: str | None = None) -> AgentRun:
        """Execute the agent loop for a single task.

        `inputs` is a free-form dict. Image paths are listed as text only in
        the first user turn; the agent attaches images via ``view_image`` or
        tool-produced artifacts (Lynn workflow), not as raw multi-MB init blobs.
        """
        wf_mode = str(getattr(self.cfg, "workflow_mode", "default") or "default").strip()
        if not wf_mode:
            wf_mode = "default"
        if wf_mode == "vlm_test":
            os.environ["VLM_AGENT_WORKFLOW_MODE"] = "vlm_test"
        else:
            os.environ.pop("VLM_AGENT_WORKFLOW_MODE", None)

        run_dir = self._prepare_run_dir(run_name)
        q_eff = self._effective_task_question(question)
        self._run_inputs = dict(inputs or {})
        self._part0_mark_tp_calls = 0
        step3_premarked = self._is_step3_premarked_case(inputs or {})
        self.console.print(Panel.fit(
            f"[bold]model[/bold] = {self.cfg.model}\n"
            f"[bold]workspace[/bold] = {self.cfg.workspace_dir}\n"
            f"[bold]workflow_mode[/bold] = {self.cfg.workflow_mode}\n"
            f"[bold]phase_isolated_context[/bold] = {self._phase_isolation_enabled()}\n"
            f"[bold]run_dir[/bold] = {run_dir}\n"
            f"[bold]tools[/bold] = {', '.join(self.registry.names())}",
            title="VLM Agent starting",
        ))

        messages = self._initial_messages(q_eff, inputs or {})
        set_runtime_context(
            project_root=Path.cwd(),
            workspace=self.cfg.workspace_dir,
            input_paths={k: v for k, v in (inputs or {}).items() if isinstance(v, str)},
            workflow_mode=getattr(self.cfg, "workflow_mode", "default"),
            run_started_at=time.time(),
        )
        self._log_jsonl(run_dir, "messages.init.jsonl", messages)

        result = AgentRun(task_question=q_eff, run_dir=run_dir)
        tools_schema = self.registry.openai_schema() if self.cfg.use_native_tools else None
        run_exception: BaseException | None = None
        self._run_started_at = time.time()
        if step3_premarked:
            self._bootstrap_step3_premarked_artifacts(inputs or {})
        phase_steps: dict[str, list[AgentStep]] = {}
        current_phase: str | None = None
        recent_tool_signatures: list[str] = []
        plan_steps: list[WorkflowPlanStep] = []
        plan_idx = 0
        last_plan_group: str | None = None
        phase_isolated = self._phase_isolation_enabled()
        if bool(getattr(self.cfg, "hierarchical_agent_mode", True)):
            plan_steps = self._build_workflow_plan(inputs or {})
            plan_idx = self._advance_plan_index(plan_steps, 0)
            try:
                (run_dir / "planner_plan.json").write_text(
                    json.dumps(
                        [
                            {
                                "step_id": s.step_id,
                                "title": s.title,
                                "objective": s.objective,
                                "done_any_artifacts": s.done_any_artifacts,
                            }
                            for s in plan_steps
                        ],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass
            self._emit("agent.run_dir", {
                "run_dir": str(run_dir),
                "task_flow": [
                    {"step_id": s.step_id, "title": s.title}
                    for s in plan_steps
                ],
            })

        step_budget_warned = False
        try:
            for step_idx in range(self.cfg.max_steps):
                step_t0 = time.time()
                self._compact_context_for_step(messages, step_idx)
                if not step_budget_warned and step_idx >= self.cfg.max_steps - 3:
                    remaining = self.cfg.max_steps - step_idx
                    messages.append(self.client.user_message(
                        f"[step-budget] Only {remaining} step(s) left "
                        f"(max_steps={self.cfg.max_steps}). Call `finish` now."
                    ))
                    step_budget_warned = True
                if plan_steps:
                    plan_idx = self._advance_plan_index(plan_steps, plan_idx)
                    if phase_isolated:
                        if plan_idx >= len(plan_steps):
                            cur_group = "done"
                        else:
                            cur_group = self._plan_phase_group(plan_steps[plan_idx].step_id)
                        need_reset = False
                        if last_plan_group is None:
                            if plan_idx > 0:
                                need_reset = True
                        elif cur_group != last_plan_group and cur_group != "done":
                            need_reset = True
                        if need_reset:
                            self._reset_messages_for_new_phase(
                                messages,
                                q_eff,
                                inputs or {},
                                plan_steps,
                                plan_idx,
                                run_dir,
                            )
                        last_plan_group = cur_group
                    self._upsert_planner_step_message(messages, plan_steps, plan_idx)
                step_tools_schema = (
                    self._phase_tools_schema(plan_steps, plan_idx)
                    if self.cfg.use_native_tools
                    else None
                )
                step_input_messages = self._snapshot_messages(messages)
                current_plan = (
                    plan_steps[plan_idx]
                    if plan_steps and 0 <= plan_idx < len(plan_steps)
                    else None
                )
                self._emit("agent.waiting", {
                    "index": step_idx,
                    "step": step_idx + 1,
                    "step_id": current_plan.step_id if current_plan else None,
                    "step_title": current_plan.title if current_plan else None,
                    "task_flow": [
                        {"step_id": s.step_id, "title": s.title}
                        for s in plan_steps
                    ] if plan_steps else [],
                })
                t0 = time.time()
                reply = self.client.chat(messages, tools_schema=step_tools_schema or tools_schema)
                llm_dt = time.time() - t0
                llm_breakdown = dict(reply.timing or {})
                llm_usage = dict(reply.usage or {})

                self._render_assistant(step_idx, reply, llm_dt)
                messages.append(reply.raw_message)

                if not reply.tool_calls:
                    step_timing = {
                        "llm_s": round(llm_dt, 4),
                        "tools_s": 0.0,
                        "images_attach_s": 0.0,
                        "step_total_s": round(time.time() - step_t0, 4),
                    }
                    if llm_breakdown:
                        step_timing["llm_prep_s"] = round(
                            float(llm_breakdown.get("prep_s", 0.0)), 4
                        )
                        step_timing["llm_api_call_s"] = round(
                            float(llm_breakdown.get("api_call_s", 0.0)), 4
                        )
                        step_timing["llm_retry_sleep_s"] = round(
                            float(llm_breakdown.get("retry_sleep_s", 0.0)), 4
                        )
                        step_timing["llm_parse_s"] = round(
                            float(llm_breakdown.get("parse_s", 0.0)), 4
                        )
                        step_timing["llm_completion_tokens"] = round(
                            float(llm_breakdown.get("completion_tokens", 0.0)), 4
                        )
                        step_timing["llm_est_prefill_s"] = round(
                            float(llm_breakdown.get("est_prefill_s", 0.0)), 4
                        )
                        step_timing["llm_est_decode_s"] = round(
                            float(llm_breakdown.get("est_decode_s", 0.0)), 4
                        )
                    result.steps.append(AgentStep(
                        index=step_idx,
                        assistant_content=reply.content,
                        tool_calls=[],
                        tool_results=[],
                        timing=step_timing,
                        tool_timing=[],
                        usage=llm_usage,
                    ))
                    self._append_step_record(
                        run_dir,
                        {
                            "index": step_idx,
                            "input_messages": step_input_messages,
                            "assistant_raw_message": reply.raw_message,
                            "assistant_content": reply.content,
                            "assistant_tool_calls": [],
                            "tool_results": [],
                            "timing": step_timing,
                            "usage": llm_usage,
                        },
                    )
                    self._emit("agent.step", {
                        "index": step_idx,
                        "step": step_idx + 1,
                        "step_id": current_plan.step_id if current_plan else None,
                        "step_title": current_plan.title if current_plan else None,
                        "tool_calls": [],
                        "timing": step_timing,
                    })
                    self._render_step_timing(step_idx, step_timing, [])
                    # Some models occasionally emit an empty assistant turn.
                    # Nudge once to either continue with tools or terminate
                    # cleanly via `finish`, instead of silently stopping.
                    if step_idx < self.cfg.max_steps - 1:
                        messages.append(self.client.user_message(
                            "You emitted no tool call. Continue by calling the "
                            "next required tool, or call `finish` now with a "
                            "structured answer (or needs_user_help=true if "
                            "evidence is insufficient)."
                        ))
                        continue
                    result.stopped_reason = "assistant-stopped-without-tool-call"
                    break

                tool_call_payload: list[dict[str, Any]] = []
                tool_result_payload: list[dict[str, Any]] = []
                tool_timing_payload: list[dict[str, Any]] = []
                attached_images: list[str] = []
                final_answer: Any = None
                tools_t0 = time.time()

                for call in reply.tool_calls:
                    block_reason = (
                        self._is_call_blocked_by_plan(call, plan_steps, plan_idx)
                        if plan_steps
                        else None
                    )
                    repeated_block = self._append_repetition_guard(
                        messages, call, recent_tool_signatures
                    )
                    if block_reason or repeated_block:
                        result_obj = ToolResult(
                            text=block_reason or (
                                f"[loop-guard] Repeated `{call.name}` call blocked. "
                                "Continue with the next required workflow action."
                            ),
                            ok=False,
                            is_final=False,
                            final_data=None,
                        )
                        tool_dt = 0.0
                        self._render_tool(call, result_obj)
                        context_tool_text = self._compress_tool_result_for_context(
                            call.name,
                            result_obj.text,
                        )
                        if self.cfg.use_native_tools:
                            messages.append(self.client.tool_result_message(
                                tool_call_id=call.id,
                                content=context_tool_text,
                            ))
                        else:
                            messages.append(self.client.user_message(
                                f"[tool-result name={call.name} id={call.id}]\n"
                                f"{context_tool_text}"
                            ))
                        tool_call_payload.append({
                            "id": call.id,
                            "name": call.name,
                            "arguments": call.arguments,
                        })
                        tool_result_payload.append({
                            "id": call.id,
                            "ok": result_obj.ok,
                            "duration_s": round(tool_dt, 4),
                            "text": truncate(result_obj.text, 2000),
                            "images": [],
                            "is_final": False,
                        })
                        tool_timing_payload.append({
                            "id": call.id,
                            "name": call.name,
                            "ok": result_obj.ok,
                            "duration_s": round(tool_dt, 4),
                        })
                        continue
                    exec_arguments: Any = self._route_outline_tool_arguments(
                        call.name, call.arguments
                    )
                    if call.name == "finish" and isinstance(call.arguments, dict):
                        exec_arguments = normalize_finish_arguments(call.arguments)
                    self._emit("tool.started", {
                        "index": step_idx,
                        "step": step_idx + 1,
                        "tool_call": {"name": call.name},
                        "name": call.name,
                    })
                    tool_t0 = time.time()
                    result_obj = self.registry.run(call.name, exec_arguments)
                    tool_dt = time.time() - tool_t0
                    if call.name == "record_board_side_decision" and result_obj.ok:
                        resolved_side = self._board_side_decision()
                        if resolved_side in {"front", "back"}:
                            # `auto` is a one-time router. Replace the mixed
                            # discovery plan as soon as the physical side is known.
                            plan_steps = self._workflow_plan_for_resolved_side(
                                resolved_side
                            )
                            plan_idx = self._advance_plan_index(plan_steps, 0)
                            last_plan_group = None
                            messages.append(self.client.user_message(
                                "[side-route-lock]\n"
                                f"Physical board side is locked to `{resolved_side}`. "
                                "The opposite-side workflow and tools are forbidden "
                                "for the remainder of this run."
                            ))
                    if call.name == "mark_tp_on_assembly_from_pdf_hit":
                        self._part0_mark_tp_calls += 1
                    finish_contract_errors: list[str] = []
                    # Hard guardrails: if model tries to finish without required
                    # skill artifacts/evidence, reject and ask it to continue.
                    if call.name == "finish" and result_obj.is_final:
                        finish_contract_errors = self._validate_skill_contract()
                        finish_contract_errors.extend(
                            self._validate_finish_answer(
                                exec_arguments.get("answer")
                                if isinstance(exec_arguments, dict)
                                else None,
                            )
                        )
                        if finish_contract_errors:
                            result_obj = ToolResult(
                                text=(
                                    "[contract-error] finish was rejected because required "
                                    "skill artifacts are missing:\n- "
                                    + "\n- ".join(finish_contract_errors)
                                    + "\nPlease continue tool calls to produce the missing "
                                      "evidence, then call finish again."
                                ),
                                ok=False,
                                is_final=False,
                                final_data=None,
                            )
                    self._render_tool(call, result_obj)
                    self._maybe_emit_python_error_hint(messages, call, result_obj)
                    mark_trigger_call = call
                    mark_trigger_result = result_obj
                    if self._should_auto_assembly_search_after_signal(
                        call, result_obj, plan_steps, plan_idx
                    ):
                        search_args = self._build_assembly_search_args_from_signal()
                        search_call = ToolInvocation(
                            id=f"{call.id}-auto-asm-search",
                            name="search_pdf_text",
                            arguments=search_args,
                        )
                        search_t0 = time.time()
                        search_result = self.registry.run(
                            "search_pdf_text", search_args
                        )
                        search_dt = time.time() - search_t0
                        self._append_tool_exchange(
                            messages=messages,
                            call=search_call,
                            result_obj=search_result,
                            tool_dt=search_dt,
                            tool_call_payload=tool_call_payload,
                            tool_result_payload=tool_result_payload,
                            tool_timing_payload=tool_timing_payload,
                            result_tag="auto-chained",
                        )
                        if search_result.ok:
                            messages.append(self.client.user_message(
                                "[auto-chain] Part0: assembly `search_pdf_text` ran after "
                                "`case10_signal_to_tp.json`. Do not repeat assembly search."
                            ))
                            mark_trigger_call = search_call
                            mark_trigger_result = search_result
                    if self._should_auto_mark_tp_after_search(
                        mark_trigger_call, mark_trigger_result, plan_steps, plan_idx
                    ):
                        mark_args = self._build_mark_tp_args_from_search(mark_trigger_call)
                        mark_call = ToolInvocation(
                            id=f"{call.id}-auto-mark",
                            name="mark_tp_on_assembly_from_pdf_hit",
                            arguments=mark_args,
                        )
                        mark_t0 = time.time()
                        mark_result = self.registry.run(
                            "mark_tp_on_assembly_from_pdf_hit", mark_args
                        )
                        self._part0_mark_tp_calls += 1
                        mark_dt = time.time() - mark_t0
                        self._append_tool_exchange(
                            messages=messages,
                            call=mark_call,
                            result_obj=mark_result,
                            tool_dt=mark_dt,
                            tool_call_payload=tool_call_payload,
                            tool_result_payload=tool_result_payload,
                            tool_timing_payload=tool_timing_payload,
                            result_tag="auto-chained",
                        )
                        if mark_result.ok:
                            messages.append(self.client.user_message(
                                "[auto-chain] Part0: `mark_tp_on_assembly_from_pdf_hit` ran "
                                "immediately after `search_pdf_text`. Do not re-read search JSON "
                                "or copy the assembly PNG manually; continue to PartB."
                            ))
                        else:
                            messages.append(self.client.user_message(
                                "[auto-chain] Automatic TP marking failed. Call "
                                "`mark_tp_on_assembly_from_pdf_hit` once with the search JSON path."
                            ))
                    context_tool_text = self._compress_tool_result_for_context(
                        call.name,
                        result_obj.text,
                    )
                    if self.cfg.use_native_tools:
                        messages.append(self.client.tool_result_message(
                            tool_call_id=call.id,
                            content=context_tool_text,
                        ))
                    else:
                        # Fallback: non-native providers often reject role=tool,
                        # so we feed the result back as a user message.
                        messages.append(self.client.user_message(
                            f"[tool-result name={call.name} id={call.id}]\n"
                            f"{context_tool_text}"
                        ))
                    if call.name == "finish" and finish_contract_errors:
                        messages.append(self.client.user_message(
                            self._build_finish_retry_hint(finish_contract_errors)
                        ))
                    tool_call_payload.append({
                        "id": call.id,
                        "name": call.name,
                        "arguments": exec_arguments
                        if isinstance(exec_arguments, dict)
                        else call.arguments,
                    })
                    tool_result_payload.append({
                        "id": call.id,
                        "ok": result_obj.ok,
                        "duration_s": round(tool_dt, 4),
                        "text": truncate(result_obj.text, 2000),
                        "images": result_obj.images,
                        "is_final": result_obj.is_final,
                    })
                    tool_timing_payload.append({
                        "id": call.id,
                        "name": call.name,
                        "ok": result_obj.ok,
                        "duration_s": round(tool_dt, 4),
                    })
                    attached_images.extend(result_obj.images)
                    if result_obj.is_final and final_answer is None:
                        final_answer = result_obj.final_data
                    if (
                        result_obj.ok
                        and call.name == "emit_step08_from_case12_aligned"
                        and plan_steps
                        and self._is_plan_step_done(plan_steps[-1])
                    ):
                        messages.append(self.client.user_message(
                            self._build_finish_now_planner_message()
                        ))
                    if self._should_auto_emit_step08_after_align(
                        call, result_obj, plan_steps, plan_idx
                    ):
                        emit_args: dict[str, Any] = {}
                        emit_call = ToolInvocation(
                            id=f"{call.id}-auto-emit-step08",
                            name="emit_step08_from_case12_aligned",
                            arguments=emit_args,
                        )
                        emit_t0 = time.time()
                        emit_result = self.registry.run(
                            "emit_step08_from_case12_aligned", emit_args
                        )
                        emit_dt = time.time() - emit_t0
                        self._append_tool_exchange(
                            messages=messages,
                            call=emit_call,
                            result_obj=emit_result,
                            tool_dt=emit_dt,
                            tool_call_payload=tool_call_payload,
                            tool_result_payload=tool_result_payload,
                            tool_timing_payload=tool_timing_payload,
                            result_tag="auto-chained",
                        )
                        if emit_result.ok:
                            messages.append(self.client.user_message(
                                "[auto-chain] PartD: `emit_step08_from_case12_aligned` ran after "
                                "alignment. Read pixel from `debug/step08_result.json` and call "
                                "`finish` now."
                            ))
                            if plan_steps and self._is_plan_step_done(plan_steps[-1]):
                                messages.append(self.client.user_message(
                                    self._build_finish_now_planner_message()
                                ))

                tools_dt = time.time() - tools_t0

                # If any tool produced images, push them in a follow-up user
                # message so the VLM can look at them next turn.
                attach_t0 = time.time()
                if attached_images:
                    parts: list[dict[str, Any]] = [
                        self.client.text_part(
                            "Here are the image(s) produced by the previous tool call(s). "
                            "Inspect them and continue.\n"
                            "Attached file paths (for `view_image` if older attachments are "
                            "removed from context to save size):\n"
                            + "\n".join(f"- {p}" for p in attached_images)
                        )
                    ]
                    if self._supports_inline_images():
                        for img_path in attached_images:
                            try:
                                parts.append(self.client.image_part(
                                    encode_image_data_url(img_path)
                                ))
                            except Exception as e:  # noqa: BLE001
                                parts.append(self.client.text_part(
                                    f"[image-attach-error] {img_path}: {e}"
                                ))
                    else:
                        joined = "\n".join(f"- {p}" for p in attached_images)
                        parts.append(self.client.text_part(
                            "Inline image blocks are not supported by current "
                            f"model/provider. Generated image files:\n{joined}\n"
                            "Use these paths with tools in subsequent steps."
                        ))
                    messages.append(self.client.user_message(parts))
                attach_dt = time.time() - attach_t0

                step_timing = {
                    "llm_s": round(llm_dt, 4),
                    "tools_s": round(tools_dt, 4),
                    "images_attach_s": round(attach_dt, 4),
                    "step_total_s": round(time.time() - step_t0, 4),
                }
                if llm_breakdown:
                    step_timing["llm_prep_s"] = round(
                        float(llm_breakdown.get("prep_s", 0.0)), 4
                    )
                    step_timing["llm_api_call_s"] = round(
                        float(llm_breakdown.get("api_call_s", 0.0)), 4
                    )
                    step_timing["llm_retry_sleep_s"] = round(
                        float(llm_breakdown.get("retry_sleep_s", 0.0)), 4
                    )
                    step_timing["llm_parse_s"] = round(
                        float(llm_breakdown.get("parse_s", 0.0)), 4
                    )
                    step_timing["llm_completion_tokens"] = round(
                        float(llm_breakdown.get("completion_tokens", 0.0)), 4
                    )
                    step_timing["llm_est_prefill_s"] = round(
                        float(llm_breakdown.get("est_prefill_s", 0.0)), 4
                    )
                    step_timing["llm_est_decode_s"] = round(
                        float(llm_breakdown.get("est_decode_s", 0.0)), 4
                    )
                result.steps.append(AgentStep(
                    index=step_idx,
                    assistant_content=reply.content,
                    tool_calls=tool_call_payload,
                    tool_results=tool_result_payload,
                    timing=step_timing,
                    tool_timing=tool_timing_payload,
                    usage=llm_usage,
                ))
                self._append_step_record(
                    run_dir,
                    {
                        "index": step_idx,
                        "input_messages": step_input_messages,
                        "assistant_raw_message": reply.raw_message,
                        "assistant_content": reply.content,
                        "assistant_tool_calls": tool_call_payload,
                        "tool_results": tool_result_payload,
                        "timing": step_timing,
                        "usage": llm_usage,
                    },
                )
                tool_names = [tc.get("name") for tc in tool_call_payload if tc.get("name")]
                self._emit("agent.step", {
                    "index": step_idx,
                    "step": step_idx + 1,
                    "step_id": current_plan.step_id if current_plan else None,
                    "step_title": current_plan.title if current_plan else None,
                    "tool_calls": tool_names,
                    "timing": step_timing,
                })
                self._render_step_timing(step_idx, step_timing, tool_timing_payload)
                latest = result.steps[-1]
                current_phase = self._maybe_emit_phase_summary(
                    messages=messages,
                    phase_steps=phase_steps,
                    current_phase=current_phase,
                    latest_step=latest,
                )

                if final_answer is not None:
                    result.final_answer = final_answer
                    result.stopped_reason = "finish-tool-called"
                    break
            else:
                result.stopped_reason = "max-steps-reached"
                if self._force_submit_at_max_steps(result):
                    result.stopped_reason = "max-steps-reached-forced-submit"
        except BaseException as e:
            run_exception = e
            result.stopped_reason = f"exception:{type(e).__name__}"
            result.last_error = str(e)[:8000]
        finally:
            result.part_timing = self._compute_part_timing(result)
            self._persist_run(run_dir, result, messages)
            title = "VLM Agent done"
            if run_exception is not None:
                title = "VLM Agent stopped (error logged)"
            part_lines = self._render_part_timing_lines(result.part_timing)
            done_body = (
                f"[bold]stopped[/bold] = {result.stopped_reason}\n"
                f"[bold]steps[/bold]   = {len(result.steps)}\n"
                + (f"{part_lines}\n" if part_lines else "")
                + f"[bold]answer[/bold]  = "
                + f"{json.dumps(result.final_answer, ensure_ascii=False) if result.final_answer else '(none)'}"
            )
            self.console.print(Panel.fit(
                done_body,
                title=title,
            ))
            if getattr(self.cfg, "post_run_reflect", False):
                run_post_run_reflection(
                    self.client,
                    self.cfg,
                    run_dir,
                    result,
                    self._infer_step_part,
                    console=self.console,
                )

        if run_exception is not None:
            raise run_exception
        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    _IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}

    @staticmethod
    def _user_message_has_inline_image(content: Any) -> bool:
        if not isinstance(content, list):
            return False
        return any(
            isinstance(p, dict) and p.get("type") == "image_url"
            for p in content
        )

    @staticmethod
    def _inline_image_url_byte_estimate(messages: list[dict[str, Any]]) -> int:
        """Rough size of data-URL image payloads (dominates JSON request body)."""
        total = 0
        for m in messages:
            c = m.get("content")
            if not isinstance(c, list):
                continue
            for p in c:
                if not isinstance(p, dict) or p.get("type") != "image_url":
                    continue
                u = p.get("image_url")
                url = u.get("url") if isinstance(u, dict) else u
                if isinstance(url, str):
                    total += len(url.encode("utf-8"))
        return total

    @staticmethod
    def _strip_image_urls_from_user_message(msg: dict[str, Any]) -> None:
        """Remove image_url parts; keep text so paths / tool output remain."""
        content = msg.get("content")
        if not isinstance(content, list):
            return
        removed = 0
        new_parts: list[Any] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "image_url":
                removed += 1
                continue
            new_parts.append(p)
        if removed == 0:
            return
        note = (
            f"\n\n[context compaction: removed {removed} inline image(s) to stay under "
            "API / gateway size limits; use `view_image` on paths listed in this turn "
            "or in tool text output.]"
        )
        appended = False
        for p in new_parts:
            if isinstance(p, dict) and p.get("type") == "text":
                t = p.get("text")
                p["text"] = (t if isinstance(t, str) else "") + note
                appended = True
                break
        if not appended:
            new_parts.insert(0, {"type": "text", "text": note.strip()})
        msg["content"] = new_parts

    def _compact_old_inline_images(self, messages: list[dict[str, Any]]) -> None:
        """Remove inline images only when size exceeds max; strip earliest first.

        Protected: the last ``context_image_keep_last`` user turns that still
        carry images (minimum 1 when max_bytes > 0). If still over budget,
        strip again while protecting only the newest image-bearing turn.
        """
        max_b = int(self.cfg.context_image_max_bytes)
        if max_b <= 0:
            return
        if self._inline_image_url_byte_estimate(messages) <= max_b:
            return

        keep = int(self.cfg.context_image_keep_last)
        if keep <= 0:
            keep = 1

        image_user_indices: list[int] = []
        for i, m in enumerate(messages):
            if m.get("role") != "user":
                continue
            if self._user_message_has_inline_image(m.get("content")):
                image_user_indices.append(i)

        def _strip_from_oldest(
            protected: set[int],
        ) -> None:
            for idx in image_user_indices:
                if self._inline_image_url_byte_estimate(messages) <= max_b:
                    return
                if idx in protected:
                    continue
                if not self._user_message_has_inline_image(
                    messages[idx].get("content")
                ):
                    continue
                self._strip_image_urls_from_user_message(messages[idx])

        if image_user_indices:
            protected_k = set(image_user_indices[-keep:])
            _strip_from_oldest(protected_k)

        if (
            self._inline_image_url_byte_estimate(messages) > max_b
            and len(image_user_indices) > 1
        ):
            protected_last = set(image_user_indices[-1:])
            _strip_from_oldest(protected_last)

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            out: list[str] = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    t = p.get("text")
                    if isinstance(t, str):
                        out.append(t)
            return "\n".join(out)
        return ""

    def _set_message_text(self, msg: dict[str, Any], text: str) -> None:
        msg["content"] = [self.client.text_part(text)]

    def _build_task_summary_text(self, task_text: str) -> str:
        lines = [ln.rstrip() for ln in task_text.splitlines() if ln.strip()]
        out: list[str] = [
            "[task-summary]",
            "Context was compacted to reduce repeated prompt tokens.",
        ]
        kept = 0
        for ln in lines:
            keep = False
            if ln.startswith("## Task") or ln.startswith("## Provided inputs"):
                keep = True
            if ln.lstrip().startswith("- [image]") or ln.lstrip().startswith("- [file]"):
                keep = True
            low = ln.lower()
            if any(k in low for k in ("mandatory", "must", "finish", "step", "debug/")):
                keep = True
            if keep:
                out.append(ln)
                kept += 1
            if kept >= 40:
                break
        summary = "\n".join(out)
        return truncate(summary, 2200)

    def _compact_message_window(self, messages: list[dict[str, Any]]) -> None:
        keep_recent = max(4, int(getattr(self.cfg, "context_keep_recent_turns", 10)))
        if len(messages) <= keep_recent + 2:
            return
        tail_indices = set(range(max(0, len(messages) - keep_recent), len(messages)))
        keep_indices: set[int] = set()
        for i, m in enumerate(messages):
            if i == 0:
                keep_indices.add(i)
                continue
            txt = self._message_text(m.get("content"))
            if txt.startswith("[phase-summary]") or txt.startswith("[global-summary]") or txt.startswith("[finish-retry-diff]"):
                keep_indices.add(i)
        keep_indices.update(tail_indices)
        compacted = [messages[i] for i in range(len(messages)) if i in keep_indices]
        messages[:] = compacted

    def _compact_context_for_step(self, messages: list[dict[str, Any]], step_idx: int) -> None:
        self._compact_old_inline_images(messages)
        drop_after = int(getattr(self.cfg, "context_drop_initial_after_step", 0))
        if (
            drop_after > 0
            and step_idx >= drop_after
            and not self._initial_task_compacted
            and len(messages) >= 2
            and messages[1].get("role") == "user"
        ):
            task_text = self._message_text(messages[1].get("content"))
            if task_text:
                self._set_message_text(messages[1], self._build_task_summary_text(task_text))
                self._initial_task_compacted = True
        self._compact_message_window(messages)

    def _build_workflow_plan(self, inputs: dict[str, Any]) -> list[WorkflowPlanStep]:
        requested_side = str(inputs.get("target_board_side", "")).strip().lower()
        if requested_side in {"front", "top"}:
            return self._workflow_plan_for_resolved_side("front")
        if requested_side in {"back", "bottom", "bot"}:
            return self._workflow_plan_for_resolved_side("back")
        if requested_side == "auto":
            return self._auto_side_workflow_plan()

        # Prefer explicit machine-readable plan when provided.
        candidate_paths: list[Path] = []
        in_task = inputs.get("workflow_plan_file")
        if isinstance(in_task, str) and in_task.strip():
            candidate_paths.append(Path(in_task.strip()))
        for cp in candidate_paths:
            try:
                if cp.is_file():
                    loaded = self._load_workflow_plan_file(cp)
                    if loaded:
                        return loaded
            except Exception:
                continue
        has_front = (
            isinstance(inputs.get("front_board_photo"), str)
            and bool(str(inputs.get("front_board_photo")).strip())
        )
        has_back = (
            isinstance(inputs.get("back_board_photo"), str)
            and bool(str(inputs.get("back_board_photo")).strip())
        )
        if has_front and not has_back:
            return self._workflow_plan_for_resolved_side("front")
        if has_back and not has_front:
            return self._workflow_plan_for_resolved_side("back")
        if has_front and has_back:
            return self._auto_side_workflow_plan()
        return self._auto_side_workflow_plan()

    @staticmethod
    def _load_workflow_plan_file(plan_path: Path) -> list[WorkflowPlanStep]:
        obj = json.loads(plan_path.read_text(encoding="utf-8"))
        steps_raw = obj.get("steps") if isinstance(obj, dict) else None
        if not isinstance(steps_raw, list):
            return []
        out: list[WorkflowPlanStep] = []
        for it in steps_raw:
            if not isinstance(it, dict):
                continue
            sid = str(it.get("step_id", "")).strip()
            title = str(it.get("title", "")).strip()
            objective = str(it.get("objective", "")).strip()
            if not sid or not title:
                continue
            done = it.get("done_any_artifacts") or []
            allowed = it.get("allowed_tools") or []
            out.append(
                WorkflowPlanStep(
                    step_id=sid,
                    title=title,
                    objective=objective,
                    done_any_artifacts=[str(x) for x in done if isinstance(x, str)],
                    allowed_tools=[str(x) for x in allowed if isinstance(x, str)],
                    next_action_hint=str(it.get("next_action_hint", "")).strip(),
                )
            )
        return out

    @staticmethod
    def _is_step3_premarked_case(inputs: dict[str, Any]) -> bool:
        flag = inputs.get("start_from_step3")
        if flag is True or str(flag).lower() in {"true", "1", "yes"}:
            return True
        if str(inputs.get("_start_from_step3", "")).lower() == "true":
            return True
        loc = inputs.get("front_locator_marked")
        brd = inputs.get("front_board_marked")
        if not (isinstance(loc, str) and Path(loc).is_file()):
            return False
        if not (isinstance(brd, str) and Path(brd).is_file()):
            return False
        if inputs.get("front_board_photo") or inputs.get("assembly_drawing_pdf"):
            return False
        return True

    def _bootstrap_step3_premarked_artifacts(self, inputs: dict[str, Any]) -> None:
        """Copy pre-marked Step2 PNGs into workspace/debug for Part D-only runs."""
        ws = self.cfg.workspace_dir.resolve()
        dbg = ws / "debug"
        dbg.mkdir(parents=True, exist_ok=True)
        now = time.time()
        pairs = (
            ("front_locator_marked", "step02_locator_front_anchor.png"),
            ("front_board_marked", "step02_board_front_anchor.png"),
        )
        for in_key, out_name in pairs:
            src = inputs.get(in_key)
            if not isinstance(src, str) or not src.strip():
                continue
            src_p = Path(src)
            if not src_p.is_file():
                continue
            dest = dbg / out_name
            shutil.copyfile(src_p, dest)
            os.utime(dest, (now, now))
        loc = dbg / "step02_locator_front_anchor.png"
        brd = dbg / "step02_board_front_anchor.png"
        if loc.is_file():
            dest = dbg / "case10_assembly_largest_ic_box.png"
            shutil.copyfile(loc, dest)
            os.utime(dest, (now, now))
        if brd.is_file():
            dest = dbg / "case10_largest_ic_box.png"
            shutil.copyfile(brd, dest)
            os.utime(dest, (now, now))

    @staticmethod
    def _step3_premarked_workflow_plan() -> list[WorkflowPlanStep]:
        return [
            WorkflowPlanStep(
                step_id="partd_case12_align_and_finish",
                title="PartD case12 align + finish (pre-marked step02)",
                objective=(
                    "Step02 anchor PNGs are already in debug/. Run case12 build+align, "
                    "emit step08, then finish."
                ),
                done_any_artifacts=[
                    "debug/case12_step02_locator_graph.json",
                    "debug/case12_board_points_aligned.json",
                    "debug/step08_result.json",
                    "debug/step08_final_tp.png",
                ],
                allowed_tools=[
                    "case12_build_and_align_from_step02_anchors",
                    "emit_step08_from_case12_aligned",
                    "finish",
                ],
                next_action_hint=(
                    "Call `case12_build_and_align_from_step02_anchors` (default step02 paths), "
                    "then `emit_step08_from_case12_aligned`, then `finish` with pixel from "
                    "step08_result.json."
                ),
            ),
        ]

    @staticmethod
    def _expected_step08_pixel_from_aligned(
        tp: list[Any],
        board_anchor: Path | None,
    ) -> tuple[float, float] | None:
        if not (isinstance(tp, list) and len(tp) == 2):
            return None
        try:
            tx = float(tp[0])
            ty = float(tp[1])
        except (TypeError, ValueError):
            return None
        if board_anchor is not None and board_anchor.is_file():
            try:
                from case12_step02_graph import clamp_board_pixel

                with Image.open(board_anchor) as im:
                    w, h = im.size
                cx, cy = clamp_board_pixel(tx, ty, w, h)
                return float(cx), float(cy)
            except Exception:  # noqa: BLE001
                pass
        return tx, ty

    @staticmethod
    def _default_workflow_plan() -> list[WorkflowPlanStep]:
        return [
            WorkflowPlanStep(
                step_id="part0_signal_to_tp",
                title="Part0-A infer target TP",
                objective="Write debug/case10_signal_to_tp.json from user question + schematic evidence.",
                done_any_artifacts=["debug/case10_signal_to_tp.json"],
                allowed_tools=[
                    "read_text_file", "view_image", "search_pdf_text", "save_text_file",
                    "pdf_page_to_image",
                ],
            ),
            WorkflowPlanStep(
                step_id="part0_pdf_search_and_mark",
                title="Part0-B/C search TP in assembly PDF and green-circle it",
                objective="Produce target TP search JSON and green-circled assembly image.",
                done_any_artifacts=[
                    "debug/case10_target_tp_pdf_search.json",
                    "debug/case10_assembly_drawing.png",
                    "debug/case10_assembly_drawing_tp_marked.png",
                ],
                allowed_tools=[
                    "search_pdf_text",
                    "mark_tp_on_assembly_from_pdf_hit",
                    "pdf_page_to_image",
                    "view_image",
                    "run_python",
                ],
            ),
            WorkflowPlanStep(
                step_id="partb_locator_largest_ic",
                title="PartB assembly largest IC",
                objective="Create case10_assembly_largest_ic_box.png and case10_assembly_largest_ic.json.",
                done_any_artifacts=[
                    "debug/case10_assembly_vlm_hints.json",
                    "debug/case10_assembly_largest_ic_box.png",
                    "debug/case10_assembly_largest_ic.json",
                ],
                allowed_tools=["view_image", "save_text_file", "run_python", "detect_largest_ic_on_assembly_from_vlm_hint", "annotate_image", "read_text_file"],
            ),
            WorkflowPlanStep(
                step_id="parta_board_largest_ic",
                title="PartA board largest IC",
                objective="Full-board OpenCV detects largest QFP on PCB (no VLM hints); writes landscape + red box.",
                done_any_artifacts=[
                    "debug/case10_largest_ic_box.png",
                    "debug/case10_largest_ic.json",
                ],
                allowed_tools=["run_python", "detect_largest_ic_on_board_full", "detect_largest_ic_on_board_from_vlm_hint", "view_image", "save_text_file", "annotate_image", "read_text_file"],
            ),
            WorkflowPlanStep(
                step_id="partd_case12_align_and_finish",
                title="PartD case12 align + finish",
                objective="Create case12 aligned outputs and final step08 artifacts, then finish.",
                done_any_artifacts=[
                    "debug/case12_step02_locator_graph.json",
                    "debug/case12_board_points_aligned.json",
                    "debug/step08_result.json",
                    "debug/step08_final_tp.png",
                ],
                allowed_tools=["case12_build_and_align_from_step02_anchors", "emit_step08_from_case12_aligned", "run_python", "annotate_image", "save_text_file", "finish", "view_image", "read_text_file"],
            ),
        ]

    @staticmethod
    def _outline_hole_workflow_plan(side: str) -> list[WorkflowPlanStep]:
        """Use the same IC-free outline/hole registration on the selected board side."""
        normalized = str(side or "").strip().lower()
        if normalized not in {"front", "back"}:
            raise ValueError(f"Unsupported outline/hole board side: {side!r}")
        photo_key = f"{normalized}_board_photo"
        locator_label = "front/Top" if normalized == "front" else "back/Bottom"
        return [
            WorkflowPlanStep(
                step_id="part0_signal_to_tp",
                title="Part0-A infer target TP",
                objective="Write debug/case10_signal_to_tp.json from user question + schematic evidence.",
                done_any_artifacts=["debug/case10_signal_to_tp.json"],
                allowed_tools=["read_text_file", "view_image", "search_pdf_text", "save_text_file", "pdf_page_to_image"],
            ),
            WorkflowPlanStep(
                step_id="part0_pdf_search_and_mark",
                title=f"Part0-B/C mark TP on the {locator_label} locator page",
                objective=(
                    f"Search only the {locator_label} assembly page and produce the green-marked "
                    f"locator image for the requested {normalized}-side TP."
                ),
                done_any_artifacts=[
                    "debug/case10_target_tp_pdf_search.json",
                    "debug/case10_assembly_drawing.png",
                    "debug/case10_assembly_drawing_tp_marked.png",
                ],
                allowed_tools=["search_pdf_text", "mark_tp_on_assembly_from_pdf_hit", "pdf_page_to_image", "view_image", "run_python"],
            ),
            WorkflowPlanStep(
                step_id="partback_vlm_landmarks",
                title=f"Optional {normalized}-side PCB-edge opening validation",
                objective=(
                    f"Compare the marked locator with INPUT_PATHS.{photo_key}. Inspect only PCB-edge "
                    "mounting/tooling holes or cutouts. Record zero matches when ambiguous, or at "
                    "least two high-confidence pairs. These landmarks validate the fixed PCB frame; "
                    "they must not rotate, mirror, or replace its homography."
                ),
                done_any_artifacts=[
                    "debug/back_02_edge_hole_candidates.json",
                    "debug/back_02_vlm_edge_hole_candidate_sheet.png",
                    "debug/back_03_vlm_edge_hole_review.json",
                ],
                allowed_tools=[
                    "prepare_back_board_landmark_candidates",
                    "view_image",
                    "record_back_landmark_review",
                ],
                next_action_hint=(
                    "Call prepare_back_board_landmark_candidates with "
                    f"back_board_path=INPUT_PATHS.{photo_key}. Record only reliable PCB-edge "
                    "mechanical-opening pairs, or matches=[] so registration keeps the fixed outline."
                ),
            ),
            WorkflowPlanStep(
                step_id="partback_board_registration",
                title=f"{normalized.capitalize()} board outline and hole registration",
                objective=(
                    "Do not detect or use any IC anchor. Register the green-marked locator to "
                    f"INPUT_PATHS.{photo_key} using the PCB outline and circular "
                    "mounting/tooling holes."
                ),
                done_any_artifacts=["debug/back_board_registration.json", "debug/back_board_registration_overlay.png"],
                allowed_tools=["view_image", "register_back_board_from_outline_and_holes", "finish"],
                next_action_hint=(
                    "Call register_back_board_from_outline_and_holes with "
                    f"back_board_path=INPUT_PATHS.{photo_key}. If automatic CV fails, view both images "
                    "and retry exactly once with rough normalized PCB ROIs. If that also fails, finish with "
                    "needs_user_help=true instead of retrying."
                ),
            ),
            WorkflowPlanStep(
                step_id="partback_finish",
                title=f"{normalized.capitalize()} board final marker",
                objective=(
                    "Create standard Step08 artifacts from the outline/hole registration on "
                    f"INPUT_PATHS.{photo_key}; camera_view must be {normalized}."
                ),
                done_any_artifacts=["debug/step03_mapping.json", "debug/step08_final_tp.png", "debug/step08_result.json"],
                allowed_tools=["emit_step08_from_back_board_registration", "finish"],
                next_action_hint=(
                    "Call emit_step08_from_back_board_registration with "
                    f"back_board_path=INPUT_PATHS.{photo_key} and camera_view={normalized}, "
                    "then finish using debug/step08_result.json."
                ),
            ),
        ]

    @staticmethod
    def _back_board_workflow_plan() -> list[WorkflowPlanStep]:
        """Compatibility wrapper for the original back-side workflow."""
        return Agent._outline_hole_workflow_plan("back")

    @staticmethod
    def _auto_side_workflow_plan() -> list[WorkflowPlanStep]:
        """Infer the physical side, then route to the same outline/hole algorithm."""
        plan = Agent._outline_hole_workflow_plan("front")[:2]
        side_step = WorkflowPlanStep(
            step_id="partside_locator_decision",
            title="Infer physical board side from locator",
            objective=(
                "View debug/case10_assembly_drawing_tp_marked.png and decide whether the marked TP "
                "belongs to the front or back locator side. The user must not choose the side."
            ),
            done_any_artifacts=["debug/board_side_decision.json"],
            allowed_tools=["view_image", "record_board_side_decision"],
            next_action_hint=(
                "View the marked locator page. Use its TOP/BOTTOM, FRONT/BACK, page title, mirror, "
                "silkscreen and component-layout evidence, then call record_board_side_decision."
            ),
        )
        plan.append(side_step)
        return plan

    @staticmethod
    def _workflow_plan_for_resolved_side(side: str) -> list[WorkflowPlanStep]:
        """Return a physically isolated plan after the one-time auto decision."""
        normalized = str(side or "").strip().lower()
        if normalized in {"front", "back"}:
            return Agent._outline_hole_workflow_plan(normalized)
        raise ValueError(f"Unsupported resolved board side: {side!r}")

    def _artifact_exists(self, rel: str) -> bool:
        ws = self.cfg.workspace_dir.resolve()
        since = float(getattr(self, "_run_started_at", 0.0))
        for p in (ws / rel, ws / "workspace" / rel):
            if not p.exists():
                continue
            try:
                st = p.stat()
                # copy2 may preserve mtime from source; ctime captures creation/update on Windows.
                latest_fs_ts = max(float(st.st_mtime), float(st.st_ctime))
                if latest_fs_ts + 1e-3 >= since:
                    return True
            except OSError:
                continue
        return False

    def _assembly_search_hit_count(self) -> int | None:
        """Return hit_count from case10_target_tp_pdf_search.json, or None if missing."""
        ws = self.cfg.workspace_dir.resolve()
        for p in (ws / "debug/case10_target_tp_pdf_search.json", ws / "workspace/debug/case10_target_tp_pdf_search.json"):
            if not p.is_file():
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj.get("hit_count"), int):
                    return int(obj["hit_count"])
                hits = obj.get("hits")
                if isinstance(hits, list):
                    return len(hits)
            except Exception:
                continue
        return None

    def _artifact_is_landscape(self, rel: str) -> bool | None:
        ws = self.cfg.workspace_dir.resolve()
        for p in (ws / rel, ws / "workspace" / rel):
            if not p.exists():
                continue
            try:
                with Image.open(p) as im:
                    w, h = im.size
                return bool(w >= h)
            except Exception:  # noqa: BLE001
                return None
        return None

    def _board_side_decision(self) -> str | None:
        ws = self.cfg.workspace_dir.resolve()
        for path in (ws / "debug/board_side_decision.json", ws / "workspace/debug/board_side_decision.json"):
            if not path.is_file():
                continue
            try:
                side = str(json.loads(path.read_text(encoding="utf-8")).get("side", "")).strip().lower()
                if side in {"front", "back"}:
                    return side
            except Exception:
                continue
        return None

    def _resolved_board_side(self) -> str | None:
        """Return the auto decision, or the explicit task-side lock."""
        decided = self._board_side_decision()
        if decided in {"front", "back"}:
            return decided
        requested = str(self._run_inputs.get("target_board_side") or "").strip().lower()
        aliases = {"top": "front", "bottom": "back", "bot": "back"}
        requested = aliases.get(requested, requested)
        return requested if requested in {"front", "back"} else None

    def _route_outline_tool_arguments(self, tool: str, arguments: Any) -> Any:
        """Force outline/hole tools onto the physical photo selected by the task."""
        if tool not in {
            "register_back_board_from_outline_and_holes",
            "emit_step08_from_back_board_registration",
        }:
            return arguments
        side = self._resolved_board_side()
        if side not in {"front", "back"}:
            return arguments
        routed = dict(arguments) if isinstance(arguments, dict) else {}
        routed["back_board_path"] = f"INPUT_PATHS.{side}_board_photo"
        if tool == "emit_step08_from_back_board_registration":
            routed["camera_view"] = side
        return routed

    def _is_plan_step_done(self, step: WorkflowPlanStep) -> bool:
        # Once the locator has proven the TP is on the back, the two largest-IC
        # phases are intentionally bypassed. The back has no stable IC anchor.
        if self._resolved_board_side() == "back" and step.step_id in {
            "partb_locator_largest_ic",
            "parta_board_largest_ic",
        }:
            return True
        return bool(step.done_any_artifacts) and all(
            self._artifact_exists(p) for p in step.done_any_artifacts
        )

    def _advance_plan_index(self, plan_steps: list[WorkflowPlanStep], idx: int) -> int:
        cur = max(0, idx)
        while cur < len(plan_steps) and self._is_plan_step_done(plan_steps[cur]):
            cur += 1
        return cur

    def _phase_isolation_enabled(self) -> bool:
        return (
            bool(getattr(self.cfg, "phase_isolated_context", True))
            and bool(getattr(self.cfg, "hierarchical_agent_mode", True))
        )

    @staticmethod
    def _plan_phase_group(step_id: str) -> str:
        sid = (step_id or "").lower()
        if sid.startswith("part0"):
            return "part0"
        if sid.startswith("partback"):
            return "partBack"
        if sid.startswith("partb"):
            return "partB"
        if sid.startswith("parta"):
            return "partA"
        if sid.startswith("partd"):
            return "partD"
        return step_id or "unknown"

    def _collect_handoff_artifact_paths(
        self,
        plan_steps: list[WorkflowPlanStep],
        until_idx: int,
    ) -> list[str]:
        from .core.handoff_manager import filter_handoff_artifact_paths

        paths: list[str] = []
        seen: set[str] = set()
        ws = self.cfg.workspace_dir
        for step in plan_steps[: max(0, until_idx)]:
            for rel in step.done_any_artifacts:
                norm = rel.replace("\\", "/")
                if norm in seen:
                    continue
                seen.add(norm)
                if (ws / Path(norm)).is_file():
                    paths.append(norm)
        return filter_handoff_artifact_paths(paths)

    def _handoff_key_facts(self) -> str:
        facts: list[str] = []
        ws = self.cfg.workspace_dir
        sig_path = ws / "debug" / "case10_signal_to_tp.json"
        if sig_path.is_file():
            try:
                obj = json.loads(sig_path.read_text(encoding="utf-8"))
                tp = obj.get("tp_id_or_ref")
                if isinstance(tp, str) and tp.strip():
                    facts.append(f"tp_id_or_ref={tp.strip()}")
            except Exception:  # noqa: BLE001
                pass
        if facts:
            return "\n".join(f"- {f}" for f in facts)
        return ""

    def _build_phase_isolated_messages(
        self,
        question: str,
        inputs: dict[str, Any],
        plan_steps: list[WorkflowPlanStep],
        plan_idx: int,
    ) -> list[dict[str, Any]]:
        """Fresh worker context for a new plan phase group (same VLM, empty history)."""
        sys_text = self.system_prompt
        if not self.cfg.use_native_tools:
            sys_text += "\n\n" + FALLBACK_TOOL_PROTOCOL.replace(
                "{tool_list}", self.registry.describe_for_prompt()
            )
        messages: list[dict[str, Any]] = [self.client.system_message(sys_text)]

        task_lines = [
            ln.strip()
            for ln in question.splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        task_one = truncate(task_lines[0] if task_lines else "TP localization task", 320)

        handoff_paths = self._collect_handoff_artifact_paths(plan_steps, plan_idx)
        cur_step = plan_steps[plan_idx].step_id if plan_idx < len(plan_steps) else "done"
        lines: list[str] = [
            "[phase-handoff]",
            f"New isolated worker session for plan phase group (step_id={cur_step}).",
            f"Task goal (one line): {task_one}",
            "",
            "Completed artifacts from earlier phases (paths only; use tools to read/view):",
        ]
        if handoff_paths:
            lines.extend(f"- {p}" for p in handoff_paths)
        else:
            lines.append("- (none)")

        facts = self._handoff_key_facts()
        if facts:
            lines.extend(["", "Key facts already established:", facts])

        lines.extend(["", "INPUT_PATHS (unchanged):"])
        skip_keys = {"workflow_plan_file", "workflow_doc", "skills_doc"}
        for key, value in (inputs or {}).items():
            if key in skip_keys or not isinstance(value, str):
                continue
            lines.append(f"- {key} = {value}")

        lines.extend([
            "",
            "Rules:",
            "- Follow ONLY the next [planner-step] message (full next_action_hint preserved).",
            "- Do NOT redo completed phases; do NOT call list_files to explore.",
            "- Open artifacts with read_text_file / view_image when the current phase requires it.",
        ])
        messages.append(self.client.user_message("\n".join(lines)))
        return messages

    def _reset_messages_for_new_phase(
        self,
        messages: list[dict[str, Any]],
        question: str,
        inputs: dict[str, Any],
        plan_steps: list[WorkflowPlanStep],
        plan_idx: int,
        run_dir: Path | None = None,
    ) -> None:
        new_msgs = self._build_phase_isolated_messages(
            question, inputs, plan_steps, plan_idx
        )
        messages.clear()
        messages.extend(new_msgs)
        self._initial_task_compacted = True
        if run_dir is None:
            return
        try:
            group = (
                self._plan_phase_group(plan_steps[plan_idx].step_id)
                if plan_idx < len(plan_steps)
                else "done"
            )
            rec = {
                "plan_idx": plan_idx,
                "phase_group": group,
                "step_id": (
                    plan_steps[plan_idx].step_id if plan_idx < len(plan_steps) else "done"
                ),
                "handoff_artifacts": self._collect_handoff_artifact_paths(
                    plan_steps, plan_idx
                ),
            }
            with (run_dir / "phase_context_resets.jsonl").open(
                "a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def _read_step08_pixel(self) -> list[int] | None:
        step08 = self.cfg.workspace_dir / "debug" / "step08_result.json"
        if not step08.is_file():
            return None
        try:
            obj = json.loads(step08.read_text(encoding="utf-8"))
            px = obj.get("pixel")
            if isinstance(px, list) and len(px) == 2:
                return [int(px[0]), int(px[1])]
        except Exception:  # noqa: BLE001
            pass
        return None

    def _validate_resolved_side_artifacts(self) -> list[str]:
        """Reject artifacts produced by the opposite physical-side workflow."""
        side = self._resolved_board_side()
        if side not in {"front", "back"}:
            return []
        errors: list[str] = []
        ws = self.cfg.workspace_dir.resolve()
        step08 = self._path_first_existing([
            ws / "debug/step08_result.json",
            ws / "workspace/debug/step08_result.json",
        ])
        mapping = self._path_first_existing([
            ws / "debug/step03_mapping.json",
            ws / "workspace/debug/step03_mapping.json",
        ])
        camera_view: str | None = None
        mapping_method: str | None = None
        if step08 is not None:
            try:
                value = json.loads(step08.read_text(encoding="utf-8")).get("camera_view")
                camera_view = str(value).strip().lower() if value is not None else None
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Invalid step08 side metadata: {exc}")
        if mapping is not None:
            try:
                value = json.loads(mapping.read_text(encoding="utf-8")).get("mapping_method")
                mapping_method = str(value).strip() if value is not None else None
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Invalid mapping side metadata: {exc}")

        if camera_view and camera_view != side:
            errors.append(
                f"step08 camera_view={camera_view!r} conflicts with locked side={side!r}."
            )
        expected_mapping = f"{side}_board_outline_holes"
        if mapping_method and mapping_method != expected_mapping:
            errors.append(
                f"{side.capitalize()} side requires mapping_method={expected_mapping!r}; "
                f"got {mapping_method!r}."
            )
        return errors

    def _force_submit_at_max_steps(self, result: Any) -> bool:
        if result.final_answer is not None:
            return False
        pixel = self._read_step08_pixel()
        has_step8_png = self._artifact_exists("debug/step08_final_tp.png")
        has_step8_json = self._artifact_exists("debug/step08_result.json")
        side_errors = self._validate_resolved_side_artifacts()
        complete = bool(pixel and has_step8_png and has_step8_json and not side_errors)
        answer: dict[str, Any] = {
            "needs_user_help": not complete,
            "forced_submit": True,
            "max_steps": self.cfg.max_steps,
        }
        locked_side = self._resolved_board_side()
        if locked_side in {"front", "back"}:
            answer["camera_view"] = locked_side
        if pixel is not None and not side_errors:
            answer["pixel"] = pixel
        answer["user_message"] = (
            f"Reached max_steps={self.cfg.max_steps}; submitting best-effort result."
            + (" Side consistency failed: " + "; ".join(side_errors) if side_errors else "")
        )
        result.final_answer = answer
        self.console.print(Panel.fit(
            f"Step budget exhausted — forced submit (max_steps={self.cfg.max_steps})",
            title="max-steps cap",
        ))
        return True

    def _build_finish_now_planner_message(self, prefix: str = "Step08 is complete.") -> str:
        px = self._read_step08_pixel()
        tp = self._handoff_key_facts()
        lines = [
            "[planner-step]",
            f"{prefix} REQUIRED next action: call `finish` ONLY — no other tools.",
            "Use answer fields:",
            "- tp_id from debug/case10_signal_to_tp.json (or handoff below)",
            f"- pixel from debug/step08_result.json"
            + (f" = {px}" if px else " (read once if needed)"),
            "- needs_user_help=false if step08_final_tp.png exists",
            "",
            "Do NOT call view_image, read_text_file, annotate_image, or re-run alignment.",
        ]
        if tp:
            lines.extend(["", "Handoff:", tp])
        return "\n".join(lines)

    def _upsert_planner_step_message(
        self,
        messages: list[dict[str, Any]],
        plan_steps: list[WorkflowPlanStep],
        plan_idx: int,
    ) -> None:
        kept: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") != "user":
                kept.append(m)
                continue
            txt = self._message_text(m.get("content"))
            if txt.startswith("[planner-step]"):
                continue
            kept.append(m)
        messages[:] = kept

        completed = [s.step_id for s in plan_steps[:plan_idx]]
        if plan_idx >= len(plan_steps):
            messages.append(self.client.user_message(
                self._build_finish_now_planner_message(
                    prefix="All planned phases are done."
                )
            ))
            return

        cur = plan_steps[plan_idx]
        needed = ", ".join(cur.done_any_artifacts) if cur.done_any_artifacts else "(none)"
        next_action = cur.next_action_hint or "Proceed with current phase required artifact generation."
        allow_tools = ", ".join(cur.allowed_tools) if cur.allowed_tools else "(not specified)"
        msg = (
            "[planner-step]\n"
            f"current_step_id: {cur.step_id}\n"
            f"title: {cur.title}\n"
            f"objective: {cur.objective}\n"
            f"done_when_artifacts_exist: {needed}\n"
            f"allowed_tools: {allow_tools}\n"
            f"completed_steps: {', '.join(completed) if completed else '(none)'}\n"
            f"next_action_hint: {next_action}\n"
            "Do not redo completed steps."
        )
        messages.append(self.client.user_message(msg))

    def _phase_tools_schema(
        self,
        plan_steps: list[WorkflowPlanStep],
        plan_idx: int,
    ) -> list[dict[str, Any]]:
        """Dynamically narrow visible tools to the current plan phase."""
        # No plan -> expose full schema (legacy behavior).
        if not plan_steps:
            return self.registry.openai_schema()

        # Plan already done: keep only tools needed for final consistency + finish.
        if plan_idx >= len(plan_steps):
            # Step8 is already emitted at this point; force direct finish.
            final_allow = ("finish",)
            out: list[dict[str, Any]] = []
            for name in final_allow:
                t = self.registry.get(name)
                if t is not None:
                    out.append(t.to_openai_schema())
            return out or self.registry.openai_schema()

        current = plan_steps[plan_idx]
        allowed = list(current.allowed_tools or [])
        side = self._resolved_board_side()
        if not allowed:
            return self.registry.openai_schema()

        out: list[dict[str, Any]] = []
        for name in allowed:
            t = self.registry.get(name)
            if t is not None:
                out.append(t.to_openai_schema())
        return out or self.registry.openai_schema()

    def _is_call_blocked_by_plan(
        self,
        call: ToolInvocation,
        plan_steps: list[WorkflowPlanStep],
        plan_idx: int,
    ) -> str | None:
        tool = call.name
        side = self._resolved_board_side()
        ic_anchor_tools = {
            "case12_build_and_align_from_step02_anchors",
            "emit_step08_from_case12_aligned",
            "detect_largest_ic_on_assembly_from_vlm_hint",
            "detect_largest_ic_on_board_full",
            "detect_largest_ic_on_board_from_vlm_hint",
        }
        if tool in ic_anchor_tools:
            return (
                f"[side-guard] side={side or 'unresolved'}: IC-anchor tools are retained in code but disabled. "
                "Use the PCB outline and mounting/tooling-hole registration on the "
                "physical photo selected by target_board_side."
            )

        # The physical-side lock is a run-wide safety invariant.  Keep it
        # above the completed-plan return: a model can still emit a stale or
        # hallucinated tool call after the planner reaches its final index.
        if plan_idx >= len(plan_steps):
            return None
        current = plan_steps[plan_idx]
        current_id = current.step_id

        # Compact workflow: Part0 quickstart is already in ## Task.
        if (
            self._run_inputs.get("_compact_workflow") == "true"
            and tool == "read_text_file"
            and isinstance(call.arguments, dict)
        ):
            p = str(call.arguments.get("path", "")).replace("\\", "/").lower()
            if "standard_workflow" in p or p.endswith("/skill.md") or "/skills/skill.md" in p:
                return (
                    "[plan-guard] ## Task already includes Part0 quickstart. "
                    "Do not read STANDARD_WORKFLOW.md or SKILL.md — call "
                    "`search_pdf_text` on INPUT_PATHS['schematic_pdf'] now."
                )

        if current_id.startswith("part0") and tool == "run_python" and isinstance(call.arguments, dict):
            code = str(call.arguments.get("code", ""))
            allow_step0c = (
                current_id == "part0_pdf_search_and_mark"
                and self._artifact_exists("debug/case10_assembly_drawing.png")
                and (self._assembly_search_hit_count() or 0) == 0
            )
            if not allow_step0c:
                if re.search(
                    r"(import\s+fitz|from\s+pymupdf|import\s+pypdf|PyPDF|pdfminer|fitz\.open|"
                    r"pypdf not available|pdf_page_to_image is not defined|"
                    r"No suitable PDF text extraction)",
                    code,
                    re.I,
                ) or current_id == "part0_signal_to_tp":
                    return (
                        "[plan-guard] Part0 PDF work must use built-in tools "
                        "`search_pdf_text`, `pdf_page_to_image`, and "
                        "`mark_tp_on_assembly_from_pdf_hit` — not run_python PDF probes."
                    )

        if current_id == "part0_pdf_search_and_mark":
            has_drawing = self._artifact_exists("debug/case10_assembly_drawing.png")
            has_marked = self._artifact_exists("debug/case10_assembly_drawing_tp_marked.png")
            has_search = self._artifact_exists("debug/case10_target_tp_pdf_search.json")
            hit_count = self._assembly_search_hit_count() if has_search else None
            mark_tp_hint = (
                "Call `mark_tp_on_assembly_from_pdf_hit` directly "
                "(optionally `assembly_png_path=INPUT_PATHS['assembly_drawing_page1_png']`). "
                "It materializes `debug/case10_assembly_drawing.png` and writes "
                "`debug/case10_assembly_drawing_tp_marked.png`."
            )
            zero_hit_visual = (
                "Assembly PDF text search returned 0 hits (graphic-only TP label). "
                "`view_image` on `debug/case10_assembly_drawing.png`, locate tp_id visually, "
                "then `run_python` Step0C to draw the green circle."
            )
            if has_search and not has_marked and hit_count == 0:
                if tool == "search_pdf_text":
                    return (
                        "[plan-guard] Assembly search JSON already records 0 text hits. "
                        f"Do not repeat search_pdf_text. {zero_hit_visual}"
                    )
                if tool == "mark_tp_on_assembly_from_pdf_hit" and has_drawing:
                    return (
                        "[plan-guard] mark_tp already rasterized the assembly page. "
                        f"{zero_hit_visual}"
                    )
            elif has_search and not has_marked and (hit_count or 0) > 0:
                if (
                    tool == "mark_tp_on_assembly_from_pdf_hit"
                    and getattr(self, "_part0_mark_tp_calls", 0) >= 1
                ):
                    return (
                        "[plan-guard] `mark_tp_on_assembly_from_pdf_hit` already ran once. "
                        "Check `debug/case10_assembly_drawing_tp_marked.png`; if missing, "
                        "use `annotate_image` once on `debug/case10_assembly_drawing.png`."
                    )
                if tool == "search_pdf_text":
                    return (
                        "[plan-guard] Assembly search JSON already has hits. "
                        f"Do not repeat search_pdf_text. {mark_tp_hint}"
                    )
                if tool in {"read_text_file", "list_files"}:
                    return (
                        "[plan-guard] Assembly search is done and TP is not marked yet. "
                        f"Skip read/list. {mark_tp_hint}"
                    )
            elif has_search and not has_marked and hit_count is None:
                if (
                    tool == "mark_tp_on_assembly_from_pdf_hit"
                    and getattr(self, "_part0_mark_tp_calls", 0) >= 1
                ):
                    return (
                        "[plan-guard] `mark_tp_on_assembly_from_pdf_hit` already ran once. "
                        "Check `debug/case10_assembly_drawing_tp_marked.png`."
                    )
                if tool in {"read_text_file", "list_files"}:
                    return (
                        "[plan-guard] `debug/case10_target_tp_pdf_search.json` exists and TP is not "
                        f"marked yet. Skip read/list. {mark_tp_hint}"
                    )
            if has_drawing and not has_marked and tool in {"read_text_file", "list_files"}:
                return (
                    "[plan-guard] `case10_assembly_drawing.png` is ready but TP is not marked yet. "
                    f"Do not loop on file reads/listing. {zero_hit_visual if hit_count == 0 else mark_tp_hint}"
                )
            if tool == "run_python" and isinstance(call.arguments, dict):
                code = str(call.arguments.get("code", ""))
                if re.search(r"\bhits\s*\[\s*['\"][^'\"]+['\"]\s*\]", code):
                    return (
                        "[plan-guard] Detected risky list indexing pattern in Python code: "
                        "`hits['...']`. In this step, `hits` is a LIST.\n"
                        "Use safe pattern:\n"
                        "- `hits = search_data.get('hits', [])`\n"
                        "- `if not hits: raise RuntimeError(...)`\n"
                        "- `hit = hits[0]`\n"
                        "- then access `hit['rect_pdf']`, `hit['page']`, etc."
                    )
                if "get(\"hits\", {})" in code or "get('hits', {})" in code:
                    return (
                        "[plan-guard] `search_data.get('hits', {})` is risky here: default type must be LIST, not dict. "
                        "Use `search_data.get('hits', [])`."
                    )

        if current_id == "part0_signal_to_tp" and tool == "list_files":
            return (
                "[plan-guard] `list_files` is disabled in Part0-A. "
                "Use INPUT_PATHS['schematic_pdf'] with `search_pdf_text` directly."
            )

        if current_id == "part0_pdf_search_and_mark" and tool == "list_files":
            return (
                "[plan-guard] `list_files` is disabled in Part0-B/C. "
                "Use INPUT_PATHS['assembly_drawing_pdf'] with `search_pdf_text`, then "
                "`mark_tp_on_assembly_from_pdf_hit`."
            )

        if current_id == "parta_board_largest_ic" and tool == "list_files":
            fb = self._run_inputs.get("front_board_photo")
            fb_hint = str(fb).strip() if isinstance(fb, str) and str(fb).strip() else "INPUT_PATHS['front_board_photo']"
            return (
                "[plan-guard] `list_files` is disabled in PartA to avoid loops. "
                "Use board source directly from `INPUT_PATHS['front_board_photo']` "
                f"(current resolved hint: `{fb_hint}`), then call `view_image` on that board "
                "source and write `debug/case10_vlm_hints.json`."
            )

        # PartD hard-stop: once Step8 artifacts exist, only `finish` is allowed.
        if current_id == "partd_case12_align_and_finish":
            has_step8_png = self._artifact_exists("debug/step08_final_tp.png")
            has_step8_json = self._artifact_exists("debug/step08_result.json")
            if has_step8_png and has_step8_json and tool != "finish":
                return (
                    "[plan-guard] PartD final artifacts are ready "
                    "(`step08_final_tp.png` + `step08_result.json`). "
                    "No further tool calls are needed; call `finish` now."
                )

        allowed = set(current.allowed_tools or [])
        if allowed and tool not in allowed:
            if (
                current_id == "part0_pdf_search_and_mark"
                and self._artifact_exists("debug/case10_target_tp_pdf_search.json")
                and not self._artifact_exists("debug/case10_assembly_drawing_tp_marked.png")
                and (self._assembly_search_hit_count() or 0) > 0
                and tool in {"read_text_file", "list_files", "run_python"}
            ):
                return (
                    "[plan-guard] After `search_pdf_text`, the only required next tool is "
                    "`mark_tp_on_assembly_from_pdf_hit` (assembly copy/marking is built in)."
                )
            order_labels: list[str] = []
            for s in plan_steps:
                label = (
                    s.step_id.replace("part0_signal_to_tp", "part0")
                    .replace("part0_pdf_search_and_mark", "part0")
                    .replace("partb_locator_largest_ic", "partB")
                    .replace("parta_board_largest_ic", "partA")
                    .replace("partd_case12_align_and_finish", "partD")
                )
                if not order_labels or order_labels[-1] != label:
                    order_labels.append(label)
            order_hint = " -> ".join(order_labels)
            return (
                f"[plan-guard] Tool `{tool}` is not allowed in phase `{current_id}`. "
                f"Follow STANDARD_WORKFLOW order: {order_hint}."
            )

        # No PDF TP search beyond Part0.
        if tool == "search_pdf_text" and not current_id.startswith("part0"):
            return (
                "[plan-guard] `search_pdf_text` is Part0-only in this workflow. "
                "Do not go back to TP PDF search after Part0."
            )
        if tool == "pdf_draw_circle_then_rasterize":
            if current_id.startswith("part0"):
                return (
                    "[plan-guard] `pdf_draw_circle_then_rasterize` is not the primary Part0 path. "
                    "After `search_pdf_text` gets rect_pdf, call "
                    "`mark_tp_on_assembly_from_pdf_hit` to write "
                    "`debug/case10_assembly_drawing_tp_marked.png`."
                )
            return (
                "[plan-guard] `pdf_draw_circle_then_rasterize` is Part0-only and not allowed "
                "in the current phase."
            )
        if current_id == "part0_pdf_search_and_mark" and tool == "pdf_page_to_image":
            asm_page1_png = self._run_inputs.get("assembly_drawing_page1_png")
            if isinstance(asm_page1_png, str) and asm_page1_png.strip():
                return (
                    "[plan-guard] Part0 for this case should use `assembly_drawing_page1_png` "
                    "via `mark_tp_on_assembly_from_pdf_hit` (it copies to "
                    "`debug/case10_assembly_drawing.png` automatically). Do not use "
                    "`pdf_page_to_image` or manual copy `run_python`."
                )
            args = call.arguments if isinstance(call.arguments, dict) else {}
            out_path = str(args.get("out_path", "")).replace("\\", "/").lower()
            if out_path and not out_path.endswith("debug/case10_assembly_drawing.png"):
                return (
                    "[plan-guard] In Part0-B/C, `pdf_page_to_image` must write exactly "
                    "`debug/case10_assembly_drawing.png` (do not invent alternate filenames)."
                )
        if current_id == "partb_locator_largest_ic" and tool == "annotate_image":
            args = call.arguments if isinstance(call.arguments, dict) else {}
            src = str(args.get("path", "")).replace("\\", "/").lower()
            out = str(args.get("out_path", "")).replace("\\", "/").lower()
            points = args.get("points", [])
            if not src.endswith("debug/case10_assembly_drawing_tp_marked.png"):
                return (
                    "[plan-guard] PartB final annotate must use source "
                    "`debug/case10_assembly_drawing_tp_marked.png` "
                    "(keep green TP circle + add red largest-IC box)."
                )
            if not out.endswith("debug/case10_assembly_largest_ic_box.png"):
                return (
                    "[plan-guard] PartB final annotate output must be "
                    "`debug/case10_assembly_largest_ic_box.png`."
                )
            if isinstance(points, list):
                for pt in points:
                    if not isinstance(pt, dict):
                        continue
                    bb = pt.get("bbox")
                    if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                        continue
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bb]
                    except Exception:  # noqa: BLE001
                        continue
                    bw, bh = (x2 - x1), (y2 - y1)
                    if bw < 40 or bh < 40 or (bw * bh) < 2500:
                        return (
                            "[plan-guard] PartB final IC bbox looks too small for largest package "
                            f"({bw}x{bh}). Expand work_roi (recommend >=20% margin around VLM ROI), "
                            "rerun OpenCV in expanded ROI, then annotate again."
                        )
                    break
        if current_id == "partb_locator_largest_ic" and tool == "run_python" and isinstance(call.arguments, dict):
            has_hints = self._artifact_exists("debug/case10_assembly_vlm_hints.json")
            has_ic_json = self._artifact_exists("debug/case10_assembly_largest_ic.json")
            if has_hints and not has_ic_json:
                return (
                    "[plan-guard] PartB should use deterministic tool "
                    "`detect_largest_ic_on_assembly_from_vlm_hint` after VLM hints are ready. "
                    "Call that tool now (work_margin_ratio=0.2) instead of ad-hoc run_python."
                )
            code = str(call.arguments.get("code", ""))
            low_code = code.lower()
            if "cv2.canny" in low_code:
                return (
                    "[plan-guard] PartB should reuse the previously successful baseline, not Canny-edge flow. "
                    "Use grayscale + THRESH_BINARY_INV (~180) + MORPH_RECT 3x3 with dilate=1 "
                    "inside work_roi, then contour area/aspect filtering."
                )
            if "touch" in low_code and "border" in low_code:
                return (
                    "[plan-guard] PartB edge-touch hard rejection caused regressions. "
                    "Do not use strict border-touch elimination; keep area/aspect-based filtering first."
                )
        if current_id == "partb_locator_largest_ic" and tool == "save_text_file":
            args = call.arguments if isinstance(call.arguments, dict) else {}
            p = str(args.get("path", "")).replace("\\", "/").lower()
            if p.endswith("debug/case10_assembly_spatial_description.md"):
                return (
                    "[plan-guard] PartB no longer requires `case10_assembly_spatial_description.md`. "
                    "Write `debug/case10_assembly_vlm_hints.json` instead and proceed to OpenCV debug + annotate."
                )
            if p.endswith("debug/case10_assembly_vlm_hints.json"):
                raw = args.get("content", "")
                txt = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                if "\"vlm_roi\"" in txt or "\"image_wh\"" in txt:
                    return (
                        "[plan-guard] `debug/case10_assembly_vlm_hints.json` must be "
                        "`approx_bbox_norm`-only for geometry. "
                        "Do not write `vlm_roi` or `image_wh`."
                    )
                if "approx_bbox_norm" not in txt:
                    return (
                        "[plan-guard] PartB sequence: after "
                        "`view_image(debug/case10_assembly_drawing_tp_marked.png)`, "
                        "write `debug/case10_assembly_vlm_hints.json` with required keys:\n"
                        "- `approx_bbox_norm` = [x1n,y1n,x2n,y2n] in [0,1]\n"
                        "- at least one of `region_hint` / `relative_to_tp` / `visual_cues`\n"
                        "Then call `detect_largest_ic_on_assembly_from_vlm_hint`."
                    )
                try:
                    obj = json.loads(txt)
                except Exception:
                    return (
                        "[plan-guard] `debug/case10_assembly_vlm_hints.json` must be valid JSON "
                        "and include semantic coarse hints (not bbox-only)."
                    )
                bbox = obj.get("approx_bbox_norm")
                if (
                    not isinstance(bbox, list)
                    or len(bbox) != 4
                    or any(not isinstance(v, (int, float)) for v in bbox)
                ):
                    return (
                        "[plan-guard] `approx_bbox_norm` must be numeric [x1n, y1n, x2n, y2n]."
                    )
                x1n, y1n, x2n, y2n = [float(v) for v in bbox]
                if not (0.0 <= x1n <= 1.0 and 0.0 <= y1n <= 1.0 and 0.0 <= x2n <= 1.0 and 0.0 <= y2n <= 1.0):
                    return (
                        "[plan-guard] `approx_bbox_norm` values must be within [0,1]."
                    )
                if x2n <= x1n or y2n <= y1n:
                    return (
                        "[plan-guard] `approx_bbox_norm` must satisfy x2>x1 and y2>y1."
                    )
                if not any(k in obj for k in ("region_hint", "relative_to_tp", "visual_cues")):
                    return (
                        "[plan-guard] `debug/case10_assembly_vlm_hints.json` must include at least one "
                        "semantic coarse hint key: `region_hint` / `relative_to_tp` / `visual_cues` "
                        "(similar to case10_vlm_roi_hints.md style), not bbox-only."
                    )
        if current_id == "partb_locator_largest_ic" and tool == "read_text_file":
            has_hints = self._artifact_exists("debug/case10_assembly_vlm_hints.json")
            has_ic_json = self._artifact_exists("debug/case10_assembly_largest_ic.json")
            has_box = self._artifact_exists("debug/case10_assembly_largest_ic_box.png")
            if has_hints and not has_ic_json:
                return (
                    "[plan-guard] PartB already has VLM hints JSON. "
                    "Do not loop on `read_text_file`; call "
                    "`detect_largest_ic_on_assembly_from_vlm_hint` now (work_margin_ratio=0.2)."
                )
            if has_ic_json and not has_box:
                return (
                    "[plan-guard] PartB OpenCV JSON is ready. "
                    "Do not loop on `read_text_file`; ensure "
                    "`debug/case10_assembly_largest_ic_box.png` exists (detect tool writes it)."
                )
        if current_id == "parta_board_largest_ic" and tool == "save_text_file":
            args = call.arguments if isinstance(call.arguments, dict) else {}
            p = str(args.get("path", "")).replace("\\", "/").lower()
            if p.endswith("debug/case10_spatial_description.md"):
                return (
                    "[plan-guard] PartA no longer uses `case10_spatial_description.md`. "
                    "Write `debug/case10_vlm_hints.json` (or `debug/case10_board_vlm_hints.json`) "
                    "with ROI hints after viewing board image."
                )
            if p.endswith("debug/case10_board_vlm_hints.json") or p.endswith("debug/case10_vlm_hints.json"):
                raw = args.get("content", "")
                txt = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                if "\"vlm_roi\"" in txt or "\"image_wh\"" in txt:
                    return (
                        "[plan-guard] board hints JSON must be `approx_bbox_norm`-only for geometry. "
                        "Do not write `vlm_roi` or `image_wh`."
                    )
                if "approx_bbox_norm" not in txt:
                    return (
                        "[plan-guard] board hints JSON must include `approx_bbox_norm` "
                        "from VLM board inspection."
                    )
                try:
                    obj = json.loads(txt)
                except Exception:
                    return (
                        "[plan-guard] board hints JSON must be valid JSON "
                        "and include semantic coarse hints."
                    )
                bbox = obj.get("approx_bbox_norm")
                if (
                    not isinstance(bbox, list)
                    or len(bbox) != 4
                    or any(not isinstance(v, (int, float)) for v in bbox)
                ):
                    return (
                        "[plan-guard] board `approx_bbox_norm` must be numeric [x1n, y1n, x2n, y2n]."
                    )
                x1n, y1n, x2n, y2n = [float(v) for v in bbox]
                if not (0.0 <= x1n <= 1.0 and 0.0 <= y1n <= 1.0 and 0.0 <= x2n <= 1.0 and 0.0 <= y2n <= 1.0):
                    return (
                        "[plan-guard] board `approx_bbox_norm` values must be within [0,1]."
                    )
                if x2n <= x1n or y2n <= y1n:
                    return (
                        "[plan-guard] board `approx_bbox_norm` must satisfy x2>x1 and y2>y1."
                    )
                if not any(k in obj for k in ("region_hint", "visual_cues", "reference_text")):
                    return (
                        "[plan-guard] board hints JSON must include at least one "
                        "semantic coarse hint key: `region_hint` / `visual_cues` / `reference_text`."
                    )
        if current_id == "parta_board_largest_ic" and tool == "read_text_file":
            has_hints = (
                self._artifact_exists("debug/case10_vlm_hints.json")
                or self._artifact_exists("debug/case10_board_vlm_hints.json")
            )
            has_ic_json = self._artifact_exists("debug/case10_largest_ic.json")
            has_box = self._artifact_exists("debug/case10_largest_ic_box.png")
            if has_hints and not has_ic_json:
                return (
                    "[plan-guard] PartA already has board VLM hints JSON. "
                    "Do not loop on `read_text_file`; call "
                    "`detect_largest_ic_on_board_from_vlm_hint` now (work_margin_ratio=0.2)."
                )
            if has_ic_json and not has_box:
                return (
                    "[plan-guard] PartA OpenCV JSON is ready. "
                    "Do not loop on `read_text_file`; ensure "
                    "`debug/case10_largest_ic_box.png` exists (detect tool writes it)."
                )
        if current_id == "parta_board_largest_ic" and tool == "run_python":
            has_hints = (
                self._artifact_exists("debug/case10_vlm_hints.json")
                or self._artifact_exists("debug/case10_board_vlm_hints.json")
            )
            has_ic_json = self._artifact_exists("debug/case10_largest_ic.json")
            if has_hints and not has_ic_json:
                return (
                    "[plan-guard] PartA should use deterministic tool "
                    "`detect_largest_ic_on_board_from_vlm_hint` after VLM hints are ready. "
                    "Call that tool now (work_margin_ratio=0.2) instead of ad-hoc run_python."
                )
        if current_id == "parta_board_largest_ic" and tool == "run_python" and isinstance(call.arguments, dict):
            code = str(call.arguments.get("code", ""))
            low_code = code.lower()
            # Enforce stable board OpenCV baseline used by successful historical runs.
            if "cv2.threshold" in low_code and re.search(r"threshold\s*\([^)]*,\s*(1[6-9]\d|2\d\d)\s*,", low_code):
                return (
                    "[plan-guard] PartA board OpenCV threshold looks too high and likely suppresses "
                    "the dark IC region. Reuse legacy-stable path: THRESH_BINARY_INV around 60 "
                    "or `opencv_ballpark` HSV-dark mode, then morphology and contour filters."
                )
            if "cv2.canny" in low_code:
                return (
                    "[plan-guard] PartA board IC detection should reuse the stable baseline from "
                    "successful runs (gray + binary_inv threshold + dilate/erode), not Canny-edge path."
                )
            rotate_patterns = (
                "cv2.rotate(",
                "np.rot90(",
                ".transpose(",
                ".rotate(",
                "ImageOps.exif_transpose(",
            )
            if any(pat in code for pat in rotate_patterns):
                return (
                    "[plan-guard] PartA should keep the physical board frame unchanged. "
                    "Do not rotate/flip board images in this workflow; use "
                    "`INPUT_PATHS['front_board_photo']` directly for VLM hints + deterministic IC tool."
                )
        if current_id == "parta_board_largest_ic" and tool == "annotate_image":
            args = call.arguments if isinstance(call.arguments, dict) else {}
            src = str(args.get("path", "")).replace("\\", "/").lower()
            out = str(args.get("out_path", "")).replace("\\", "/").lower()
            points = args.get("points", [])
            fb = self._run_inputs.get("front_board_photo")
            fb_norm = str(fb).replace("\\", "/").lower() if isinstance(fb, str) else ""
            src_ok = (
                "case10_board_landscape.png" in src
                or "front_board_photo" in src
                or (fb_norm and src == fb_norm)
            )
            if not src_ok:
                return (
                    "[plan-guard] PartA annotate source must be the board image itself "
                    "(prefer `INPUT_PATHS['front_board_photo']`, compatibility: "
                    "`debug/case10_board_landscape.png`)."
                )
            if not out.endswith("debug/case10_largest_ic_box.png"):
                return (
                    "[plan-guard] PartA final annotate output must be "
                    "`debug/case10_largest_ic_box.png`."
                )
            if isinstance(points, list):
                for pt in points:
                    if not isinstance(pt, dict):
                        continue
                    bb = pt.get("bbox")
                    if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                        continue
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bb]
                    except Exception:  # noqa: BLE001
                        continue
                    bw, bh = (x2 - x1), (y2 - y1)
                    if bw < 40 or bh < 40 or (bw * bh) < 2500:
                        return (
                            "[plan-guard] PartA final IC bbox looks too small for largest package "
                            f"({bw}x{bh}). Expand board work_roi (recommend >=20% margin around VLM ROI), "
                            "rerun OpenCV in expanded ROI, then annotate again."
                        )
                    break
        if current_id in {"parta_board_largest_ic", "partd_case12_align_and_finish"} and tool == "run_python":
            has_locator = self._artifact_exists("debug/case10_assembly_largest_ic_box.png")
            has_board = self._artifact_exists("debug/case10_largest_ic_box.png")
            if has_locator and has_board:
                return (
                    "[plan-guard] Do not use ad-hoc run_python copy scripts here. "
                    "Call `case12_build_and_align_from_step02_anchors` directly with:\n"
                    "- `locator_anchor_path=debug/case10_assembly_largest_ic_box.png`\n"
                    "- `board_anchor_path=debug/case10_largest_ic_box.png`"
                )
        # Do not finish before PartD.
        if current_id == "partd_case12_align_and_finish" and tool == "read_text_file":
            has_graph = self._artifact_exists("debug/case12_step02_locator_graph.json")
            has_aligned = self._artifact_exists("debug/case12_board_points_aligned.json")
            has_step8_png = self._artifact_exists("debug/step08_final_tp.png")
            has_step8_json = self._artifact_exists("debug/step08_result.json")
            if has_graph and not has_aligned:
                return (
                    "[plan-guard] PartD direct path: graph is ready. "
                    "Run align now to produce `debug/case12_board_points_aligned.json` "
                    "(do not loop on reads)."
                )
            if has_aligned and not (has_step8_png and has_step8_json):
                return (
                    "[plan-guard] PartD direct path: aligned JSON is ready. "
                    "Call `emit_step08_from_case12_aligned` now to generate "
                    "`debug/step08_final_tp.png` + `debug/step08_result.json` "
                    "(and update `debug/step03_mapping.json`), then finish."
                )
            if has_step8_png and has_step8_json:
                return (
                    "[plan-guard] PartD final artifacts are ready "
                    "(`step08_final_tp.png` + `step08_result.json`). "
                    "Do not loop on `read_text_file`; call `finish` now."
                )
        if current_id == "partd_case12_align_and_finish" and tool == "view_image":
            has_step8_png = self._artifact_exists("debug/step08_final_tp.png")
            has_step8_json = self._artifact_exists("debug/step08_result.json")
            if has_step8_png and has_step8_json:
                return (
                    "[plan-guard] PartD final artifacts are already ready "
                    "(`step08_final_tp.png` + `step08_result.json`). "
                    "Skip extra `view_image` verification and call `finish` now."
                )
        if current_id == "partd_case12_align_and_finish" and tool == "save_text_file":
            args = call.arguments if isinstance(call.arguments, dict) else {}
            p = str(args.get("path", "")).replace("\\", "/").lower()
            blocked = (
                "debug/case12_step02_locator_graph.json",
                "debug/case12_step02_locator_graph.png",
                "debug/case12_board_points_aligned.json",
                "debug/case12_board_approx_overlay_opencv.png",
            )
            if any(p.endswith(x) for x in blocked):
                return (
                    "[plan-guard] PartD case12 artifacts must be generated by "
                    "`case12_step02_graph` functions, not manual `save_text_file`.\n"
                    "Use run_python:\n"
                    "1) `from case12_step02_graph import run_build_step02_locator_graph`\n"
                    "2) `run_build_step02_locator_graph(Path(os.environ['WORKSPACE']))`\n"
                    "3) `from case12_step02_graph import run_align_locator_graph_to_board_ic_bbox`\n"
                    "4) `run_align_locator_graph_to_board_ic_bbox(Path(os.environ['WORKSPACE']))`"
                )
        if current_id == "partd_case12_align_and_finish" and tool == "finish":
            ws = self.cfg.workspace_dir.resolve()
            map_p = self._path_first_existing(
                [ws / "debug" / "step03_mapping.json", ws / "workspace" / "debug" / "step03_mapping.json"]
            )
            mm: str | None = None
            if map_p is not None:
                try:
                    obj = json.loads(map_p.read_text(encoding="utf-8"))
                    mv = obj.get("mapping_method")
                    if isinstance(mv, str):
                        mm = mv.strip() or None
                except Exception:  # noqa: BLE001
                    mm = None
            if mm == "back_board_outline_holes":
                return None
            if mm not in {"case12_step02_opencv_ic_align", "case12_step02_vlm_ic_align"}:
                return (
                    "[plan-guard] Before `finish`, set `debug/step03_mapping.json` "
                    "`mapping_method` to `case12_step02_opencv_ic_align` (or "
                    "`case12_step02_vlm_ic_align` for vlm_test). This prevents fallback "
                    "to legacy dual-ROI Step4/5 contract checks."
                )
            aligned_p = self._path_first_existing(
                [
                    ws / "debug" / "case12_board_points_aligned.json",
                    ws / "workspace" / "debug" / "case12_board_points_aligned.json",
                ]
            )
            if aligned_p is not None:
                try:
                    ao = json.loads(aligned_p.read_text(encoding="utf-8"))
                    if not isinstance(ao.get("source"), str):
                        return (
                            "[plan-guard] `case12_board_points_aligned.json` is missing `source`. "
                            "Regenerate it via case12 align flow before finish."
                        )
                except Exception:  # noqa: BLE001
                    return (
                        "[plan-guard] `case12_board_points_aligned.json` is invalid JSON. "
                        "Regenerate alignment output before finish."
                    )
            graph_p = self._path_first_existing(
                [
                    ws / "debug" / "case12_step02_locator_graph.json",
                    ws / "workspace" / "debug" / "case12_step02_locator_graph.json",
                ]
            )
            if graph_p is not None:
                try:
                    go = json.loads(graph_p.read_text(encoding="utf-8"))
                    if go.get("schema_version") != 3:
                        return (
                            "[plan-guard] `case12_step02_locator_graph.json` must be schema_version=3 "
                            "from `run_build_step02_locator_graph`."
                        )
                    must_keys = ("tp_center", "largest_ic", "references", "graph_edges", "pairwise_roi")
                    miss = [k for k in must_keys if k not in go]
                    if miss:
                        return (
                            "[plan-guard] `case12_step02_locator_graph.json` is missing keys: "
                            + ", ".join(miss)
                            + ". Regenerate via `run_build_step02_locator_graph`."
                        )
                except Exception:  # noqa: BLE001
                    return (
                        "[plan-guard] Invalid `case12_step02_locator_graph.json`. "
                        "Regenerate with `run_build_step02_locator_graph`."
                    )
        if current_id == "partd_case12_align_and_finish" and tool == "run_python" and isinstance(call.arguments, dict):
            has_s02_loc = self._artifact_exists("debug/step02_locator_front_anchor.png")
            has_s02_board = self._artifact_exists("debug/step02_board_front_anchor.png")
            has_graph = self._artifact_exists("debug/case12_step02_locator_graph.json")
            if has_s02_loc and has_s02_board and not has_graph:
                return (
                    "[plan-guard] PartD should call `case12_build_and_align_from_step02_anchors` first. "
                    "It takes the two step02 anchor images and runs official case12 graph build+align "
                    "without `workspace/workspace` path issues."
                )
            has_aligned = self._artifact_exists("debug/case12_board_points_aligned.json")
            has_step8 = self._artifact_exists("debug/step08_final_tp.png") and self._artifact_exists("debug/step08_result.json")
            if has_aligned and not has_step8:
                return (
                    "[plan-guard] Aligned JSON is already ready in PartD. "
                    "Do not use ad-hoc run_python now; call `emit_step08_from_case12_aligned`."
                )
            code = str(call.arguments.get("code", ""))
            low = code.lower()
            if "cv2.matchtemplate(" in low:
                return (
                    "[plan-guard] PartD should reuse the built-in matching tool instead of ad-hoc "
                    "`cv2.matchTemplate` scripts. Use `match_green_tp_roi_to_board` with project "
                    "defaults (`search_max_dim=1200`, `min_match_score=0.2`, `scale_steps=26`) "
                    "or proceed with case12 graph align tools."
                )
            case12_files = (
                "case12_step02_locator_graph.json",
                "case12_step02_locator_graph.png",
                "case12_board_points_aligned.json",
                "case12_board_approx_overlay_opencv.png",
            )
            if any(x in low for x in case12_files):
                if (
                    "run_build_step02_locator_graph" not in code
                    and "run_align_locator_graph_to_board_ic_bbox" not in code
                    and "run_align_locator_graph_to_board_ic_bbox_vlm" not in code
                ):
                    return (
                        "[plan-guard] PartD case12 files are being written by ad-hoc Python. "
                        "Use official functions from `case12_step02_graph.py`:\n"
                        "- `run_build_step02_locator_graph(...)`\n"
                        "- `run_align_locator_graph_to_board_ic_bbox(...)` (or `_vlm` variant)."
                    )
        if current_id == "partd_case12_align_and_finish" and tool == "case12_build_and_align_from_step02_anchors":
            has_aligned = self._artifact_exists("debug/case12_board_points_aligned.json")
            has_step8 = self._artifact_exists("debug/step08_final_tp.png") and self._artifact_exists("debug/step08_result.json")
            if has_aligned and has_step8:
                return (
                    "[plan-guard] PartD is complete (`case12_board_points_aligned.json` + step08). "
                    "Call `finish` now; do not rebuild alignment."
                )
            if has_aligned and not has_step8:
                return (
                    "[plan-guard] PartD alignment already completed. "
                    "Call `emit_step08_from_case12_aligned` now instead of rebuilding graph/alignment."
                )
        if (
            self._is_step3_premarked_case(self._run_inputs)
            and current_id == "partd_case12_align_and_finish"
        ):
            has_aligned = self._artifact_exists("debug/case12_board_points_aligned.json")
            has_step8 = self._artifact_exists("debug/step08_final_tp.png") and self._artifact_exists(
                "debug/step08_result.json"
            )
            if tool in {
                "read_text_file",
                "list_files",
                "search_pdf_text",
                "view_image",
                "run_python",
                "annotate_image",
                "save_text_file",
                "mark_tp_on_assembly_from_pdf_hit",
                "pdf_page_to_image",
            }:
                if has_step8:
                    return (
                        "[plan-guard] Step3+ fast path: step08 artifacts exist. "
                        "Call `finish` only."
                    )
                if has_aligned:
                    return (
                        "[plan-guard] Step3+ fast path: alignment is done. "
                        "Call `emit_step08_from_case12_aligned` then `finish` — "
                        "no read/view/python."
                    )
                return (
                    "[plan-guard] Step3+ fast path: step02 anchors are preloaded. "
                    "Call `case12_build_and_align_from_step02_anchors` once, then "
                    "`emit_step08_from_case12_aligned`, then `finish`."
                )
            if tool == "emit_step08_from_case12_aligned" and has_step8:
                return (
                    "[plan-guard] `step08_final_tp.png` and `step08_result.json` already exist. "
                    "Call `finish` with pixel from step08_result.json."
                )
        if tool == "run_step3_mapping":
            if current_id in {"parta_board_largest_ic", "partd_case12_align_and_finish"}:
                return (
                    "[plan-guard] This STANDARD_WORKFLOW path should not use `run_step3_mapping`. "
                    "After step02 anchors are ready, call "
                    "`case12_build_and_align_from_step02_anchors` directly."
                )
        if current_id == "partd_case12_align_and_finish" and tool in {"annotate_image", "save_text_file"}:
            has_aligned = self._artifact_exists("debug/case12_board_points_aligned.json")
            has_step8 = self._artifact_exists("debug/step08_final_tp.png") and self._artifact_exists("debug/step08_result.json")
            if has_aligned and not has_step8:
                return (
                    "[plan-guard] PartD finalization should use dedicated tool "
                    "`emit_step08_from_case12_aligned` instead of manual annotate/save."
                )
        if tool == "finish" and current_id not in {
            "partd_case12_align_and_finish",
            "partback_board_registration",
            "partback_finish",
        }:
            return (
                "[plan-guard] `finish` is only allowed in the final mapping phase "
                "after prior phases complete."
            )
        return None

    @staticmethod
    def _tool_call_signature(call: ToolInvocation) -> str:
        if call.name == "read_text_file" and isinstance(call.arguments, dict):
            p = str(call.arguments.get("path", "")).replace("\\", "/").lower()
            return f"read_text_file|{p}"
        if call.name == "search_pdf_text" and isinstance(call.arguments, dict):
            pdf = str(call.arguments.get("pdf_path", "")).replace("\\", "/").lower()
            q = str(call.arguments.get("query", "")).strip().lower()
            return f"search_pdf_text|{pdf}|{q}"
        if call.name == "list_files" and isinstance(call.arguments, dict):
            p = str(call.arguments.get("path", "")).replace("\\", "/").lower()
            return f"list_files|{p}"
        if call.name == "view_image" and isinstance(call.arguments, dict):
            p = str(call.arguments.get("path", "")).replace("\\", "/").lower()
            return f"view_image|{p}"
        try:
            args = json.dumps(call.arguments or {}, ensure_ascii=False, sort_keys=True)
        except Exception:  # noqa: BLE001
            args = str(call.arguments)
        return f"{call.name}|{args}"

    def _append_repetition_guard(
        self,
        messages: list[dict[str, Any]],
        call: ToolInvocation,
        recent_signatures: list[str],
    ) -> bool:
        sig = self._tool_call_signature(call)
        recent_signatures.append(sig)
        if len(recent_signatures) > 6:
            del recent_signatures[:-6]
        if call.name == "view_image":
            if len(recent_signatures) >= 2 and recent_signatures[-1] == recent_signatures[-2]:
                messages.append(self.client.user_message(
                    "[loop-guard]\n"
                    "You called `view_image` on the same path consecutively. "
                    "Do not re-open the same image repeatedly; proceed with the next workflow action."
                ))
                return True
            return False
        if len(recent_signatures) < 3:
            return False
        if not (recent_signatures[-1] == recent_signatures[-2] == recent_signatures[-3]):
            return False
        if call.name not in {
            "list_files", "read_text_file", "search_pdf_text",
            "register_back_board_from_outline_and_holes",
            "emit_step08_from_back_board_registration",
        }:
            return False
        messages.append(self.client.user_message(
            "[loop-guard]\n"
            f"You have called the same `{call.name}` command 3 times consecutively. "
            "Do not repeat it again. Proceed to the next concrete workflow step."
        ))
        return True

    @staticmethod
    def _extract_artifact_paths(text: str) -> list[str]:
        pat = re.compile(
            r"([A-Za-z]:\\[^\\\n\r\t\"']+\.(?:png|jpg|jpeg|json|md|pdf|txt)|"
            r"(?:debug|artifacts|progress)[/\\][^\\\n\r\t\"']+\.(?:png|jpg|jpeg|json|md|txt))",
            re.IGNORECASE,
        )
        found: list[str] = []
        for m in pat.finditer(text):
            p = m.group(1)
            if p not in found:
                found.append(p)
            if len(found) >= 6:
                break
        return found

    def _compress_tool_result_for_context(self, tool_name: str, raw_text: str) -> str:
        max_chars = max(180, int(getattr(self.cfg, "context_tool_text_max_chars", 700)))
        text = (raw_text or "").strip()
        if not text:
            return f"[tool-summary] {tool_name}: (empty output)"
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        keep_lines: list[str] = []
        for ln in lines:
            low = ln.lower()
            if low.startswith("[python-error]") or "[contract-error]" in low:
                keep_lines.append(ln)
                continue
            if any(k in low for k in ("wrote ", "saved ", "listing ", "final_bbox", "pixel", "missing", "error", "warning")):
                keep_lines.append(ln)
            if len(keep_lines) >= 6:
                break
        if not keep_lines:
            keep_lines = lines[:4]
        paths = self._extract_artifact_paths(text)
        out = [f"[tool-summary] {tool_name}"]
        out.extend(f"- {ln}" for ln in keep_lines)
        if paths:
            out.append("- artifacts:")
            out.extend(f"  - {p}" for p in paths)
        joined = "\n".join(out)
        return truncate(joined, max_chars)

    @staticmethod
    def _build_finish_retry_hint(errors: list[str]) -> str:
        missing = [e.strip() for e in errors[:4]]
        body = "\n".join(f"- {e}" for e in missing)
        return (
            "[finish-retry-diff]\n"
            "Previous finish rejected. Only fix these gaps, then retry finish:\n"
            f"{body}\n"
            "Do not restate full reasoning; continue from current artifacts."
        )

    def _build_assembly_search_args_from_signal(self) -> dict[str, Any]:
        ws = self.cfg.workspace_dir
        sig_p = ws / "debug" / "case10_signal_to_tp.json"
        tp_query = ""
        locator_page: int | None = None
        signal_side = ""
        if sig_p.is_file():
            try:
                sig_obj = json.loads(sig_p.read_text(encoding="utf-8"))
                tp_query = str(sig_obj.get("tp_id_or_ref") or "").strip()
                for key in ("locator_page", "assembly_page"):
                    value = sig_obj.get(key)
                    if isinstance(value, int) and value >= 1:
                        locator_page = value
                        break
                signal_side = str(sig_obj.get("board_side") or "").strip().lower()
                target_points = sig_obj.get("target_points")
                if isinstance(target_points, list):
                    selected_point: dict[str, Any] | None = None
                    for point in target_points:
                        if not isinstance(point, dict):
                            continue
                        point_tp = str(
                            point.get("tp_id")
                            or point.get("tp_id_or_ref")
                            or point.get("test_point")
                            or ""
                        ).strip()
                        if not tp_query and point_tp:
                            tp_query = point_tp
                        if point_tp and tp_query and point_tp.upper() == tp_query.upper():
                            selected_point = point
                            break
                        if selected_point is None:
                            selected_point = point
                    if selected_point is not None:
                        if locator_page is None:
                            for key in ("locator_page", "assembly_page"):
                                value = selected_point.get(key)
                                if isinstance(value, int) and value >= 1:
                                    locator_page = value
                                    break
                        if not signal_side:
                            signal_side = str(
                                selected_point.get("board_side") or ""
                            ).strip().lower()
            except Exception:  # noqa: BLE001
                pass
        asm_pdf = (
            self._run_inputs.get("assembly_drawing_pdf")
            or self._run_inputs.get("locator_pdf")
            or self._run_inputs.get("assembly_drawing_pdf_path")
        )
        requested_side = str(
            self._run_inputs.get("target_board_side") or signal_side or ""
        ).strip().lower()
        page_reason = ""
        if requested_side in {"back", "bottom"}:
            # TI-style two-page assembly drawings used by this workflow place
            # TOP on page 1 and BOTTOM on page 2. Never silently fall back to
            # the opposite face when the requested TP appears on both pages.
            locator_page = 2
            page_reason = (
                "authoritative target_board_side=back requires BOTTOM assembly page"
            )
        elif requested_side in {"front", "top"}:
            locator_page = 1
            page_reason = (
                "authoritative target_board_side=front requires TOP assembly page"
            )
        elif locator_page is not None:
            page_reason = "explicit locator_page/assembly_page from case10_signal_to_tp.json"
        args = {
            "pdf_path": str(asm_pdf or ""),
            "query": tp_query,
            "out_json_path": "debug/case10_target_tp_pdf_search.json",
        }
        if locator_page is not None:
            args["page_filter"] = [locator_page]
            args["page_filter_reason"] = page_reason
        return args

    def _should_auto_emit_step08_after_align(
        self,
        call: ToolInvocation,
        result_obj: ToolResult,
        plan_steps: list[WorkflowPlanStep],
        plan_idx: int,
    ) -> bool:
        if not result_obj.ok or call.name != "case12_build_and_align_from_step02_anchors":
            return False
        if plan_idx >= len(plan_steps):
            return False
        if plan_steps[plan_idx].step_id != "partd_case12_align_and_finish":
            return False
        if not self._artifact_exists("debug/case12_board_points_aligned.json"):
            return False
        return not self._artifact_exists("debug/step08_result.json")

    def _should_auto_assembly_search_after_signal(
        self,
        call: ToolInvocation,
        result_obj: ToolResult,
        plan_steps: list[WorkflowPlanStep],
        plan_idx: int,
    ) -> bool:
        if not result_obj.ok or call.name != "save_text_file":
            return False
        if plan_idx >= len(plan_steps):
            return False
        if plan_steps[plan_idx].step_id != "part0_signal_to_tp":
            return False
        args = call.arguments if isinstance(call.arguments, dict) else {}
        path = str(args.get("path", "")).replace("\\", "/").lower()
        if not path.endswith("debug/case10_signal_to_tp.json"):
            return False
        if self._artifact_exists("debug/case10_target_tp_pdf_search.json"):
            return False
        asm_pdf = (
            self._run_inputs.get("assembly_drawing_pdf")
            or self._run_inputs.get("locator_pdf")
        )
        return bool(asm_pdf)

    def _build_mark_tp_args_from_search(self, call: ToolInvocation) -> dict[str, Any]:
        args = call.arguments if isinstance(call.arguments, dict) else {}
        mark_args: dict[str, Any] = {
            "search_json_path": str(
                args.get("out_json_path") or "debug/case10_target_tp_pdf_search.json"
            ),
        }
        asm = self._run_inputs.get("assembly_drawing_page1_png")
        if isinstance(asm, str) and asm.strip():
            mark_args["assembly_png_path"] = asm
        pdf = self._run_inputs.get("assembly_drawing_pdf")
        if isinstance(pdf, str) and pdf.strip():
            mark_args["assembly_pdf_path"] = pdf
        return mark_args

    def _should_auto_mark_tp_after_search(
        self,
        call: ToolInvocation,
        result_obj: ToolResult,
        plan_steps: list[WorkflowPlanStep],
        plan_idx: int,
    ) -> bool:
        if not result_obj.ok or call.name != "search_pdf_text":
            return False
        if plan_idx >= len(plan_steps):
            return False
        step_id = plan_steps[plan_idx].step_id
        if not step_id.startswith("part0"):
            return False
        if self._artifact_exists("debug/case10_assembly_drawing_tp_marked.png"):
            return False
        hit_count = self._assembly_search_hit_count()
        return bool(hit_count and hit_count > 0)

    def _append_tool_exchange(
        self,
        *,
        messages: list[dict[str, Any]],
        call: ToolInvocation,
        result_obj: ToolResult,
        tool_dt: float,
        tool_call_payload: list[dict[str, Any]],
        tool_result_payload: list[dict[str, Any]],
        tool_timing_payload: list[dict[str, Any]],
        result_tag: str = "",
    ) -> None:
        self._render_tool(call, result_obj)
        context_tool_text = self._compress_tool_result_for_context(call.name, result_obj.text)
        tag_suffix = f" {result_tag}".rstrip()
        if self.cfg.use_native_tools:
            messages.append(self.client.tool_result_message(
                tool_call_id=call.id,
                content=context_tool_text,
            ))
        else:
            messages.append(self.client.user_message(
                f"[tool-result name={call.name} id={call.id}{tag_suffix}]\n"
                f"{context_tool_text}"
            ))
        exec_arguments = call.arguments if isinstance(call.arguments, dict) else call.arguments
        tool_call_payload.append({
            "id": call.id,
            "name": call.name,
            "arguments": exec_arguments,
        })
        tool_result_payload.append({
            "id": call.id,
            "ok": result_obj.ok,
            "duration_s": round(tool_dt, 4),
            "text": truncate(result_obj.text, 2000),
            "images": result_obj.images,
            "is_final": result_obj.is_final,
        })
        tool_timing_payload.append({
            "id": call.id,
            "name": call.name,
            "ok": result_obj.ok,
            "duration_s": round(tool_dt, 4),
        })

    def _maybe_emit_python_error_hint(
        self,
        messages: list[dict[str, Any]],
        call: ToolInvocation,
        result_obj: ToolResult,
    ) -> None:
        if call.name != "run_python" or bool(result_obj.ok):
            return
        txt = str(result_obj.text or "")
        if "list indices must be integers or slices, not str" in txt:
            messages.append(self.client.user_message(
                "[python-error-hint]\n"
                "The previous python failed with list/dict indexing mismatch.\n"
                "Fix pattern:\n"
                "- `hits = search_data.get('hits', [])` (hits is a list)\n"
                "- check `if not hits: raise RuntimeError(...)`\n"
                "- use `hit = hits[0]` then access dict keys from `hit`\n"
                "- do NOT use string key indexing directly on `hits` list."
            ))
        if "cannot unpack non-iterable nonetype object" in txt.lower():
            messages.append(self.client.user_message(
                "[python-error-hint]\n"
                "Previous python failed with None unpack.\n"
                "Defensive checks before unpacking are required:\n"
                "- verify function return is not None before `a, b = value`\n"
                "- verify contour/candidate list is non-empty before selecting best\n"
                "- for cv2 reads, check `img is not None` before using shape/unpack\n"
                "- for optional matches, use explicit `if value is None: ...` fallback path."
            ))

    def _build_phase_summary(self, phase: str, steps: list[AgentStep]) -> str:
        max_lines = max(5, int(getattr(self.cfg, "context_phase_summary_max_lines", 8)))
        tool_names: list[str] = []
        artifacts: list[str] = []
        for st in steps:
            for c in st.tool_calls:
                nm = c.get("name")
                if isinstance(nm, str) and nm and nm not in tool_names:
                    tool_names.append(nm)
            for r in st.tool_results:
                txt = str(r.get("text", ""))
                for p in self._extract_artifact_paths(txt):
                    if p not in artifacts:
                        artifacts.append(p)
                    if len(artifacts) >= 4:
                        break
        lines: list[str] = [
            f"[phase-summary] {phase}",
            f"- steps: {len(steps)}",
            f"- tools: {', '.join(tool_names[:6]) if tool_names else '(none)'}",
        ]
        if artifacts:
            lines.append(f"- key artifacts: {', '.join(artifacts[:4])}")
        last = steps[-1] if steps else None
        if last and last.tool_results:
            lines.append(f"- latest result: {truncate(str(last.tool_results[-1].get('text', '')), 180)}")
        return "\n".join(lines[:max_lines])

    def _maybe_emit_phase_summary(
        self,
        messages: list[dict[str, Any]],
        phase_steps: dict[str, list[AgentStep]],
        current_phase: str | None,
        latest_step: AgentStep,
    ) -> str | None:
        latest_phase = self._infer_step_part(latest_step, prev_part=current_phase)
        phase_steps.setdefault(latest_phase, []).append(latest_step)
        if current_phase is None:
            return latest_phase
        if latest_phase == current_phase:
            return current_phase
        if current_phase != "unknown" and not self._phase_isolation_enabled():
            summary = self._build_phase_summary(current_phase, phase_steps.get(current_phase, []))
            messages.append(self.client.user_message(summary))
            self._compact_message_window(messages)
        return latest_phase

    def _supports_inline_images(self) -> bool:
        """Best-effort capability check for `image_url` message blocks."""
        model = self.cfg.model.lower()
        base = self.cfg.base_url.lower()

        # DeepSeek V4 text models reject image_url blocks.
        if "deepseek" in base and ("deepseek-v4-pro" in model or "deepseek-v4-flash" in model):
            return False

        # Common multimodal model-name hints.
        vision_hints = (
            "vision", "vl", "4v", "gpt-4o", "gpt-4.1",
            "internvl", "llava", "minicpm-v", "doubao-vision",
        )
        if any(h in model for h in vision_hints):
            return True

        # Default optimistic for unknown providers.
        return True

    def _effective_task_question(self, question: str) -> str:
        """Append mode-specific appendix (e.g. ``--mode vlm_test``) to the YAML task."""
        text = question.strip()
        if getattr(self.cfg, "workflow_mode", "default") != "vlm_test":
            return text
        return text + "\n\n" + CLI_WORKFLOW_MODE_VLM_TEST_APPEND_ZH

    def _initial_messages(self, question: str,
                          inputs: dict[str, Any]) -> list[dict[str, Any]]:
        sys_text = self.system_prompt
        if not self.cfg.use_native_tools:
            sys_text += "\n\n" + FALLBACK_TOOL_PROTOCOL.replace(
                "{tool_list}", self.registry.describe_for_prompt()
            )
        messages: list[dict[str, Any]] = [self.client.system_message(sys_text)]

        text_context_lines: list[str] = [
            "## Task",
            question.strip(),
            "",
        ]
        requested_side = str(inputs.get("target_board_side") or "auto").strip().lower()
        requested_side = {
            "top": "front",
            "bottom": "back",
            "bot": "back",
        }.get(requested_side, requested_side)
        text_context_lines.extend([
            "## Board registration policy (MANDATORY)",
            "- Use PCB outline plus mounting/tooling-hole registration for both front and back.",
            "- Largest-IC and all IC-anchor tools remain installed but are forbidden in this workflow.",
        ])
        if requested_side in {"front", "back"}:
            locator_page = "TOP/page 1" if requested_side == "front" else "BOTTOM/page 2"
            text_context_lines.extend([
                f"- Board side is locked to `{requested_side}`.",
                f"- Search only the {locator_page} assembly locator page.",
                f"- Register only onto `INPUT_PATHS.{requested_side}_board_photo`.",
                f"- Final `camera_view` must be `{requested_side}`.",
            ])
        else:
            text_context_lines.append(
                "- Decide the physical side once from locator evidence, then lock the matching "
                "front_board_photo or back_board_photo and use the same outline/hole algorithm."
            )
        text_context_lines.extend(["", "## Provided inputs"])
        image_parts: list[dict[str, Any]] = []

        inline_images_ok = self._supports_inline_images()
        if not inline_images_ok:
            text_context_lines.append(
                "- [note] inline image blocks disabled for current model/provider; "
                "the agent should use file paths plus tools (`run_python`, "
                "`crop_image`, `annotate_image`) to inspect images."
            )

        for key, value in inputs.items():
            if isinstance(value, str) and Path(value).suffix.lower() in self._IMG_EXT:
                path = Path(value)
                if path.exists():
                    text_context_lines.append(
                        f"- [image-path] {key} = {path} "
                        "(use `view_image` when this step needs visual evidence)"
                    )
                else:
                    text_context_lines.append(
                        f"- [image-missing] {key} = {path}"
                    )
            elif isinstance(value, str) and Path(value).exists():
                text_context_lines.append(f"- [file] {key} = {value}")
            else:
                text_context_lines.append(f"- {key}: {value}")

        str_inputs = {k: v for k, v in inputs.items() if isinstance(v, str)}
        if str_inputs:
            text_context_lines.append("")
            text_context_lines.append(self._format_input_paths_block(str_inputs))

        text_context_lines.append("")
        text_context_lines.append(
            "## Path discipline (MANDATORY)\n"
            "- **Input files are already resolved above** in `INPUT_PATHS`. Use those paths in tools.\n"
            "- **Do NOT** call `list_files` to discover inputs. There is **no** `inputs/` subdirectory.\n"
            "- In `run_python`, read source files from `INPUT_PATHS[...]` whenever possible.\n"
            "- `WORKSPACE` and `PROJECT_ROOT` variables are available in `run_python`.\n"
            "- For tool output paths (crop/annotate/save), use relative paths like "
            "`progress/step_01.md` or `artifacts/result.png` (DO NOT prefix with `workspace/`)."
        )
        text_context_lines.append("")
        text_context_lines.append(
            "## Step2 gate (when Task requires red anchor boxes)\n"
            "If the **Task** text says to **skip Step2** and use only "
            "`match_green_tp_roi_to_board`, obey the Task (no red-box PNGs).\n"
            "Otherwise, for red-anchor workflows:\n"
            "- Anchor: IC with **clear silkscreen** on both images.\n"
            "- **Do not rotate/mirror/warp** images; native pixel coordinates only.\n"
            "- **Board:** prefer **`annotate_image`** `{bbox, color:red}` or `run_python` with "
            "**`INPUT_PATHS['front_board_photo']`** (never paste broken `F:\\...\\` strings — "
            "in Python `\"...\\test...\"` turns `\\t` into a tab and breaks paths).\n"
            "- **Locator:** red box on `debug/step01_locator_front_anchor.png`, save "
            "`debug/step02_locator_front_anchor.png`.\n"
            "- Do **not** call `run_step3_mapping` until both step02 files exist."
        )
        text_context_lines.append("")
        text_context_lines.append(
            "## Step3 execution constraints (MANDATORY when task starts from Step3)\n"
            "- Prefer **`match_green_tp_roi_to_board`** when the Task skips red anchors; "
            "omit `board_path` so `INPUT_PATHS['front_board_photo']` is used.\n"
            "- Reuse tool defaults for template matching unless evidence suggests otherwise: "
            "`search_max_dim=1200`, `min_match_score=0.2`, `scale_steps=26`, `margin_px=80`.\n"
            "- For STANDARD_WORKFLOW case011 (part0->partB->partA->partD), "
            "do **not** use `run_step3_mapping`; use `case12_build_and_align_from_step02_anchors` in PartD.\n"
            "- `run_step3_mapping` is only for non-case12 legacy Step3 flows explicitly requested by Task.\n"
            "- Use HSV red ranges [0,100,100]-[10,255,255] and [170,100,100]-[180,255,255].\n"
            "- Use HSV green range [40,100,100]-[80,255,255].\n"
            "- Detect largest contour and use boundingRect/moments.\n"
            "- Use scalar mapping formula u,v -> px,py (no cv2.transform).\n"
            "- Do not clamp u/v by default; values outside [0,1] can be valid.\n"
            "- Emit debug/step03_mapping.json and debug/step03_prior_on_board.png.\n"
            "- Optional: prefer tool `match_green_tp_roi_to_board` (green ROI template) "
            "when red anchors are weak; same JSON schema."
        )
        text_context_lines.append("")
        wf_mode_run = getattr(self.cfg, "workflow_mode", "default")
        part_d_primary = (
            "- **Default Part D (`STANDARD_WORKFLOW`)** uses **`case12_step02_opencv_ic_align`** "
            "（两框 OpenCV）："
            "`run_build_step02_locator_graph` → `run_align_locator_graph_to_board_ic_bbox` "
            "(OpenCV **实物 IC 红框**) "
            "→ call **`emit_step08_from_case12_aligned`** "
            "(from **`case12_board_points_aligned.json`** to **`step08_final_tp.png`** + "
            "**`step08_result.json`** + `step03_mapping.json.mapping_method`) "
            "→ **`finish`**.\n"
        )
        if wf_mode_run == "vlm_test":
            part_d_primary = (
                "- **This CLI run (`workflow_mode=vlm_test`)** uses **`case12_step02_vlm_ic_align`**: "
                "see **`vlm_test` appendix inside ## Task above** — `run_build_step02_locator_graph` → "
                "VLM → **`case12_board_largest_ic_bbox_vlm.json`** → "
                "**`run_align_locator_graph_to_board_ic_bbox_vlm`** → **`case12_board_points_aligned.json`** "
                "(**`source`=`vlm_ic_correspondence_isotropic_align`**) → **`step08_*` → `finish`**.\n"
            )
        text_context_lines.append(
            "## Step4–8 tool mandate (full-flow tasks)\n"
            "- Produce the **`debug/*.png` / `debug/*.json`** artifacts your Task requires; "
            "`progress/step_*.md` notes are **optional** (runtime does not gate `finish` on them).\n"
            + part_d_primary +
            "- **Legacy Part D** (`case10_dual_roi_layout`): **Step4** — **`read_text_file`** "
            "(step03_mapping.json), **`image_info`** (board), "
            "**`crop_image`** → `debug/step04_roi_crop.png`, **`view_image`** as needed.\n"
            "- **VLM Path C variant:** Step4 is **`annotate_image`** on "
            "`debug/step03_locator_roi.png` "
            "→ `debug/step04_locator_landmarks.png` (red landmark boxes), "
            "then board prior/mapping; "
            "board ROI crop stays `debug/step04_roi_crop.png` after mapping.\n"
            "- **Dual-ROI path Step5–7:** must call **`run_candidate_pipeline`** at least "
            "once with `roi_bbox`, "
            "`prior_board`, and board image path (see Task / SKILL).\n"
            "- **Step8:** must call **`annotate_image`** on the **full** board → "
            "`debug/step08_final_tp.png`, "
            "write consistent **`debug/step08_result.json`**, then **`finish`** with **`pixel`** [x,y] "
            "aligned "
            "to that board frame.\n"
            "- Describing a step without the matching tool call is incomplete."
        )
        text_context_lines.append("")
        text_context_lines.append(
            "Start with **Part 0**: `search_pdf_text` on `INPUT_PATHS['schematic_pdf']` "
            "(query=`target_signal`), then `assembly_drawing_pdf`, then "
            "`mark_tp_on_assembly_from_pdf_hit`. "
            "**Do not** spend steps on `list_files` or `run_python` PDF library probes. "
            "Use tools for every non-trivial step (especially Step4–8 — each step needs real tool calls). "
            "Finish by calling the `finish` tool."
        )

        user_parts: list[dict[str, Any]] = [
            self.client.text_part("\n".join(text_context_lines))
        ]
        user_parts.extend(image_parts)
        messages.append(self.client.user_message(user_parts))
        messages.append(self.client.user_message(
            self._build_global_summary_message(question, inputs)
        ))
        return messages

    @staticmethod
    def _format_input_paths_block(str_inputs: dict[str, str]) -> str:
        """Explicit resolved paths so the model does not hunt for an ``inputs/`` folder."""
        payload = json.dumps(str_inputs, ensure_ascii=False, indent=2)
        return "\n".join([
            "## INPUT_PATHS (resolved — authoritative)",
            "Use these keys in tools and in `run_python` as `INPUT_PATHS['key']`.",
            "**Forbidden:** `list_files` on `.`, `inputs/`, or case folders to discover assets.",
            "",
            "```json",
            payload,
            "```",
        ])

    @staticmethod
    def _build_global_summary_message(question: str, inputs: dict[str, Any]) -> str:
        lines: list[str] = [
            "[global-summary]",
            "- Goal: locate target TP on board and return finish.answer.pixel [x,y].",
            "- Keep tool-first workflow; each required step still needs real tool calls.",
            "- Prefer compact carry-over: use artifacts in debug/*.png|json as source of truth.",
            "- Retry rule: when finish is rejected, only patch missing artifacts then finish again.",
            f"- Task digest: {truncate(' '.join(question.split()), 260)}",
        ]
        inp_keys = ", ".join(sorted(str(k) for k in inputs.keys())[:12])
        if inp_keys:
            lines.append(f"- Input keys: {inp_keys}")
        path_hints = [
            f"{k}={v}" for k, v in sorted(inputs.items())
            if isinstance(v, str) and Path(v).exists()
        ][:6]
        if path_hints:
            lines.append("- Resolved paths (sample): " + "; ".join(path_hints))
        return "\n".join(lines)

    def _validate_finish_answer(self, answer: Any) -> list[str]:
        """Require a board pixel [x,y] in finish payload; align with step08_result when present."""
        errors: list[str] = []
        if answer is None:
            return ["finish tool requires an `answer` object"]
        if not isinstance(answer, dict):
            if isinstance(answer, str):
                try:
                    answer = json.loads(answer)
                except json.JSONDecodeError:
                    return ["finish answer must be a JSON object"]
            else:
                return ["finish answer must be a JSON object"]
        if answer.get("needs_user_help") is True:
            return errors

        locked_side = self._resolved_board_side()
        answer_side = str(answer.get("camera_view", "")).strip().lower()
        if locked_side in {"front", "back"} and answer_side != locked_side:
            errors.append(
                "finish.answer.camera_view must match the locked board side "
                f"({locked_side}); got {answer_side or 'missing'}."
            )
        errors.extend(self._validate_resolved_side_artifacts())

        pixel = answer.get("pixel")
        if not (isinstance(pixel, list) and len(pixel) == 2):
            errors.append(
                "finish.answer.pixel must be [x, y] on the full board image "
                "(same coordinate frame as debug/step08_final_tp.png) unless needs_user_help=true."
            )
        else:
            try:
                fx, fy = float(pixel[0]), float(pixel[1])
                if not (math.isfinite(fx) and math.isfinite(fy)):
                    errors.append("finish.answer.pixel values must be finite numbers")
            except Exception:
                errors.append("finish.answer.pixel must be numeric [x, y]")

        ws = self.cfg.workspace_dir.resolve()
        step08_res = next(
            (
                p
                for p in (
                    ws / "debug" / "step08_result.json",
                    ws / "workspace" / "debug" / "step08_result.json",
                )
                if p.exists()
            ),
            None,
        )
        if (
            step08_res is not None
            and isinstance(pixel, list)
            and len(pixel) == 2
            and not errors
        ):
            try:
                robj = json.loads(step08_res.read_text(encoding="utf-8"))
                sp = robj.get("pixel")
                if isinstance(sp, list) and len(sp) == 2:
                    if abs(float(sp[0]) - float(pixel[0])) > 1.0 or abs(
                        float(sp[1]) - float(pixel[1])
                    ) > 1.0:
                        errors.append(
                            "finish.answer.pixel must match debug/step08_result.json pixel (±1px)."
                        )
            except Exception as e:  # noqa: BLE001
                errors.append(f"Could not cross-check finish pixel with step08_result.json: {e}")

        return errors

    def _part_b_assembly_stepb3_qc_errors(self, ws: Path) -> list[str]:
        """Shared Part B StepB3 gates (assembly largest IC JSON + view gate + mtime order)."""
        errors: list[str] = []
        asm_ic_path = self._path_first_existing(
            [
                ws / "debug" / "case10_assembly_largest_ic.json",
                ws / "workspace" / "debug" / "case10_assembly_largest_ic.json",
            ],
        )
        if asm_ic_path is not None:
            try:
                asm_ic_obj = json.loads(asm_ic_path.read_text(encoding="utf-8"))
                qc_b = asm_ic_obj.get("part_b_stepb3_qc")
                if not isinstance(qc_b, dict):
                    errors.append(
                        "case10_assembly_largest_ic.json must include object "
                        "part_b_stepb3_qc (Part B StepB3 QC gate); see STANDARD_WORKFLOW StepB3."
                    )
                else:
                    fv = qc_b.get("final_verdict")
                    if fv not in ("QC_PASS", "QC_PASS_WITH_CAVEATS"):
                        errors.append(
                            "part_b_stepb3_qc.final_verdict must be QC_PASS or "
                            "QC_PASS_WITH_CAVEATS after StepB3 (Revise rounds must finish before "
                            "writing this JSON)."
                        )
                    if qc_b.get("viewed_largest_ic_box_png_before_final_json") is not True:
                        errors.append(
                            "part_b_stepb3_qc.viewed_largest_ic_box_png_before_final_json "
                            "must be true: call view_image(debug/case10_assembly_largest_ic_box.png), "
                            "run the StepB3 checklist, declare QC_PASS or QC_REVISE, and only "
                            "then write case10_assembly_largest_ic.json."
                        )
                    if qc_b.get(
                        "final_annotate_overwrote_png_immediately_before_json"
                    ) is not True:
                        errors.append(
                            "part_b_stepb3_qc.final_annotate_overwrote_png_immediately_before_json "
                            "must be true: after the final bbox is fixed (including after QC_REVISE), "
                            "you must call annotate_image to overwrite "
                            "debug/case10_assembly_largest_ic_box.png, then write the JSON with the "
                            "same bbox — do not update JSON without re-exporting the PNG."
                        )
                    note = qc_b.get("whole_page_largest_package_checked_zh")
                    if not isinstance(note, str) or len(note.strip()) < 6:
                        errors.append(
                            "part_b_stepb3_qc.whole_page_largest_package_checked_zh must be a "
                            "short Chinese note that the full-page largest package was verified "
                            "(not a smaller neighbor IC)."
                        )
                    qru = qc_b.get("qc_rounds_used")
                    try:
                        qru_n = int(qru)  # JSON may ship small ints only
                    except (TypeError, ValueError):
                        qru_n = -1
                    if isinstance(qru, bool):
                        qru_n = -1
                    if qru_n < 2:
                        errors.append(
                            "part_b_stepb3_qc.qc_rounds_used must be an integer >= 2 for "
                            "Part B (assembly drawing): you must run at least one QC_REVISE "
                            "cycle (revise case10_assembly_vlm_hints.json → StepB2 run_python → "
                            "re-annotate case10_assembly_largest_ic_box.png → view_image) before "
                            "final QC_PASS. Part A (board photo) has no such minimum."
                        )
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON in case10_assembly_largest_ic.json: {e}")
            except OSError as e:
                errors.append(f"Failed to read case10_assembly_largest_ic.json: {e}")

            box_asm = self._path_first_existing(
                [
                    ws / "debug" / "case10_assembly_largest_ic_box.png",
                    ws / "workspace" / "debug" / "case10_assembly_largest_ic_box.png",
                ],
            )
            if box_asm is not None:
                gate_p = self._path_first_existing(
                    [
                        ws / "debug" / "case10_stepb3_viewed_largest_ic_box.json",
                        ws / "workspace" / "debug" / "case10_stepb3_viewed_largest_ic_box.json",
                    ],
                )
                if gate_p is None:
                    errors.append(
                        "Part B StepB3 (tool-enforced): missing "
                        "`debug/case10_stepb3_viewed_largest_ic_box.json`. "
                        "After `annotate_image` → `debug/case10_assembly_largest_ic_box.png`, "
                        "you MUST call `view_image` on that exact PNG (runtime records the gate), "
                        "then save `case10_assembly_largest_ic.json`. "
                        "QC_REVISE: re-annotate → re-view → then JSON."
                    )
                else:
                    try:
                        gobj = json.loads(gate_p.read_text(encoding="utf-8"))
                        gmt = gobj.get("png_mtime")
                        bmt = box_asm.stat().st_mtime
                        if gmt is None or not math.isclose(
                            float(gmt), float(bmt), rel_tol=0, abs_tol=1e-3
                        ):
                            errors.append(
                                "Part B StepB3 (tool-enforced): "
                                "`case10_stepb3_viewed_largest_ic_box.json` is stale — "
                                "it does not match the current on-disk "
                                "`case10_assembly_largest_ic_box.png`. "
                                "Call `view_image(debug/case10_assembly_largest_ic_box.png)` "
                                "again after the latest `annotate_image` (required after QC_REVISE)."
                            )
                    except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
                        errors.append(
                            "Part B StepB3 (tool-enforced): invalid view gate "
                            f"`debug/case10_stepb3_viewed_largest_ic_box.json`: {e}"
                        )
                    if asm_ic_path is not None:
                        try:
                            if asm_ic_path.stat().st_mtime + 0.001 < gate_p.stat().st_mtime:
                                errors.append(
                                    "Part B StepB3 (tool-enforced): "
                                    "`case10_assembly_largest_ic.json` must be saved AFTER "
                                    "`view_image` on the final red-box PNG (gate newer than JSON)."
                                )
                        except OSError:
                            pass
        return errors

    @staticmethod
    def _path_first_existing(paths: list[Path]) -> Path | None:
        for q in paths:
            if q.exists():
                return q
        return None

    @staticmethod
    def _infer_case12_mapping_method_from_workspace(ws: Path) -> str | None:
        """If ``step03_mapping.json`` lacks ``mapping_method``, infer from aligned output."""
        for rel in (
            ("debug", "case12_board_points_aligned.json"),
            ("workspace", "debug", "case12_board_points_aligned.json"),
        ):
            p = ws.joinpath(*rel)
            if not p.is_file():
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            src = obj.get("source")
            if src == "opencv_ic_bbox_isotropic_align":
                return "case12_step02_opencv_ic_align"
            if src == "vlm_ic_correspondence_isotropic_align":
                return "case12_step02_vlm_ic_align"
        return None

    def _validate_skill_contract_case12_opencv_ic_align(self, ws: Path) -> list[str]:
        """Part D default: case12 locator graph + OpenCV red IC boxes; TP from aligned JSON."""
        errors: list[str] = []
        if self._is_step3_premarked_case(self._run_inputs):
            case10_prefix = [
                "debug/step02_locator_front_anchor.png",
                "debug/step02_board_front_anchor.png",
            ]
        else:
            case10_prefix = [
                "debug/case10_signal_to_tp.json",
                "debug/case10_target_tp_pdf_search.json",
                "debug/case10_assembly_drawing.png",
                "debug/case10_assembly_drawing_tp_marked.png",
                "debug/case10_assembly_largest_ic_box.png",
                "debug/case10_assembly_largest_ic.json",
                "debug/case10_largest_ic_box.png",
                "debug/case10_largest_ic.json",
                "debug/step02_locator_front_anchor.png",
                "debug/step02_board_front_anchor.png",
                # case12_step02_opencv_ic_align skips Part C board_tp_marked homography overlay.
            ]
        case12_tail = [
            "debug/step03_mapping.json",
            "debug/case12_step02_locator_graph.json",
            "debug/case12_board_points_aligned.json",
            "debug/step08_final_tp.png",
            "debug/step08_result.json",
        ]
        for rel in case10_prefix + case12_tail:
            p1 = ws / rel
            p2 = ws / "workspace" / rel
            if not (p1.exists() or p2.exists()):
                errors.append(f"Missing required debug artifact: {rel}")

        s02b = self._path_first_existing(
            [
                ws / "debug" / "step02_board_front_anchor.png",
                ws / "workspace" / "debug" / "step02_board_front_anchor.png",
            ],
        )

        aligned_p = self._path_first_existing(
            [
                ws / "debug" / "case12_board_points_aligned.json",
                ws / "workspace" / "debug" / "case12_board_points_aligned.json",
            ],
        )
        if aligned_p is not None:
            try:
                al = json.loads(aligned_p.read_text(encoding="utf-8"))
                if al.get("source") != "opencv_ic_bbox_isotropic_align":
                    errors.append(
                        "case12_board_points_aligned.json source must be "
                        "`opencv_ic_bbox_isotropic_align` for mapping_method "
                        "`case12_step02_opencv_ic_align`."
                    )
                tp = al.get("board_roi_target_px_approx")
                if not (isinstance(tp, list) and len(tp) == 2):
                    errors.append(
                        "case12_board_points_aligned.json missing board_roi_target_px_approx [x,y]."
                    )
                else:
                    step08r = self._path_first_existing(
                        [
                            ws / "debug" / "step08_result.json",
                            ws / "workspace" / "debug" / "step08_result.json",
                        ],
                    )
                    if step08r is not None:
                        try:
                            so = json.loads(step08r.read_text(encoding="utf-8"))
                            sp = so.get("pixel")
                            if isinstance(sp, list) and len(sp) == 2:
                                exp = self._expected_step08_pixel_from_aligned(tp, s02b)
                                if exp is None:
                                    exp = (float(tp[0]), float(tp[1]))
                                if abs(float(sp[0]) - exp[0]) > 1.0 or abs(
                                    float(sp[1]) - exp[1]
                                ) > 1.0:
                                    errors.append(
                                        "step08_result.json pixel must match "
                                        "case12_board_points_aligned.json "
                                        "board_roi_target_px_approx (±1px)."
                                    )
                        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
                            errors.append(f"Invalid step08_result.json: {e}")
            except (json.JSONDecodeError, OSError) as e:
                errors.append(f"Invalid case12_board_points_aligned.json: {e}")

        step08 = self._path_first_existing(
            [
                ws / "debug" / "step08_final_tp.png",
                ws / "workspace" / "debug" / "step08_final_tp.png",
            ],
        )
        if s02b is not None and step08 is not None:
            try:
                from PIL import Image

                with Image.open(s02b) as im_b:
                    wb = im_b.size
                with Image.open(step08) as im8:
                    w8 = im8.size
                if wb != w8:
                    errors.append(
                        "step08_final_tp.png must match step02_board_front_anchor.png "
                        "width×height (full board frame)."
                    )
            except Exception as e:  # noqa: BLE001
                errors.append(f"Failed to validate step08 vs board size: {e}")

        return errors

    def _validate_skill_contract_case12_vlm_ic_align(self, ws: Path) -> list[str]:
        """Part D CLI ``vlm_test``: skip Part A board IC PNG; locator graph + VLM IC JSON → aligned."""
        errors: list[str] = []
        prefix = [
            "debug/case10_signal_to_tp.json",
            "debug/case10_target_tp_pdf_search.json",
            "debug/case10_assembly_drawing.png",
            "debug/case10_assembly_drawing_tp_marked.png",
            "debug/case10_assembly_largest_ic_box.png",
            "debug/case10_assembly_largest_ic.json",
            "debug/step02_locator_front_anchor.png",
            "debug/step02_board_front_anchor.png",
        ]
        tail = [
            "debug/step03_mapping.json",
            "debug/case12_step02_locator_graph.json",
            "debug/case12_board_largest_ic_bbox_vlm.json",
            "debug/case12_board_points_aligned.json",
            "debug/step08_final_tp.png",
            "debug/step08_result.json",
        ]
        for rel in prefix + tail:
            p1 = ws / rel
            p2 = ws / "workspace" / rel
            if not (p1.exists() or p2.exists()):
                errors.append(f"Missing required debug artifact: {rel}")

        map_p = self._path_first_existing(
            [
                ws / "debug" / "step03_mapping.json",
                ws / "workspace" / "debug" / "step03_mapping.json",
            ]
        )
        if map_p is not None:
            try:
                mm = json.loads(map_p.read_text(encoding="utf-8")).get(
                    "mapping_method"
                )
                if mm != "case12_step02_vlm_ic_align":
                    errors.append(
                        "step03_mapping.json mapping_method must be "
                        "`case12_step02_vlm_ic_align` when using this validator."
                    )
            except (json.JSONDecodeError, OSError, TypeError) as e:
                errors.append(f"Invalid step03_mapping.json: {e}")

        s02b = self._path_first_existing(
            [
                ws / "debug" / "step02_board_front_anchor.png",
                ws / "workspace" / "debug" / "step02_board_front_anchor.png",
            ],
        )

        aligned_p = self._path_first_existing(
            [
                ws / "debug" / "case12_board_points_aligned.json",
                ws / "workspace" / "debug" / "case12_board_points_aligned.json",
            ]
        )
        if aligned_p is not None:
            try:
                al = json.loads(aligned_p.read_text(encoding="utf-8"))
                if al.get("source") != "vlm_ic_correspondence_isotropic_align":
                    errors.append(
                        "case12_board_points_aligned.json source must be "
                        "`vlm_ic_correspondence_isotropic_align` for "
                        "`case12_step02_vlm_ic_align`."
                    )
                tp = al.get("board_roi_target_px_approx")
                if not (isinstance(tp, list) and len(tp) == 2):
                    errors.append(
                        "case12_board_points_aligned.json missing "
                        "board_roi_target_px_approx [x,y]."
                    )
                else:
                    step08r = self._path_first_existing(
                        [
                            ws / "debug" / "step08_result.json",
                            ws / "workspace" / "debug" / "step08_result.json",
                        ]
                    )
                    if step08r is not None:
                        try:
                            so = json.loads(step08r.read_text(encoding="utf-8"))
                            sp = so.get("pixel")
                            if isinstance(sp, list) and len(sp) == 2:
                                exp = self._expected_step08_pixel_from_aligned(tp, s02b)
                                if exp is None:
                                    exp = (float(tp[0]), float(tp[1]))
                                if abs(float(sp[0]) - exp[0]) > 1.0 or abs(
                                    float(sp[1]) - exp[1]
                                ) > 1.0:
                                    errors.append(
                                        "step08_result.json pixel must match "
                                        "case12_board_points_aligned.json "
                                        "board_roi_target_px_approx (±1px)."
                                    )
                        except (
                            json.JSONDecodeError,
                            OSError,
                            TypeError,
                            ValueError,
                        ) as e:
                            errors.append(f"Invalid step08_result.json: {e}")
            except (json.JSONDecodeError, OSError) as e:
                errors.append(f"Invalid case12_board_points_aligned.json: {e}")

        s02b = self._path_first_existing(
            [
                ws / "debug" / "step02_board_front_anchor.png",
                ws / "workspace" / "debug" / "step02_board_front_anchor.png",
            ]
        )
        step08 = self._path_first_existing(
            [
                ws / "debug" / "step08_final_tp.png",
                ws / "workspace" / "debug" / "step08_final_tp.png",
            ]
        )
        if s02b is not None and step08 is not None:
            try:
                from PIL import Image

                with Image.open(s02b) as im_b:
                    wb = im_b.size
                with Image.open(step08) as im8:
                    w8 = im8.size
                if wb != w8:
                    errors.append(
                        "step08_final_tp.png must match step02_board_front_anchor.png "
                        "width×height (full board frame)."
                    )
            except Exception as e:  # noqa: BLE001
                errors.append(f"Failed to validate step08 vs board size: {e}")

        return errors

    def _validate_skill_contract_back_board_registration(self, ws: Path) -> list[str]:
        """Finish contract for the IC-free back-side geometric registration path."""
        errors: list[str] = []
        required = [
            "debug/case10_signal_to_tp.json",
            "debug/case10_target_tp_pdf_search.json",
            "debug/case10_assembly_drawing.png",
            "debug/case10_assembly_drawing_tp_marked.png",
            "debug/back_board_registration.json",
            "debug/back_board_registration_overlay.png",
            "debug/back_01_locator_outline.png",
            "debug/back_01_photo_outline.png",
            "debug/back_04_reprojection_overlay.png",
            "debug/back_04_registration_validation.json",
            "debug/back_05_tp_projection.png",
            "debug/back_06_tp_local_candidates.png",
            "debug/back_06_tp_local_candidates.json",
            "debug/step03_mapping.json",
            "debug/step08_final_tp.png",
            "debug/step08_result.json",
        ]
        for rel in required:
            if self._path_first_existing([ws / rel, ws / "workspace" / rel]) is None:
                errors.append(f"Missing required back-board artifact: {rel}")
        registration = self._path_first_existing([
            ws / "debug/back_board_registration.json",
            ws / "workspace/debug/back_board_registration.json",
        ])
        final = self._path_first_existing([
            ws / "debug/step08_result.json",
            ws / "workspace/debug/step08_result.json",
        ])
        review_file = self._path_first_existing([
            ws / "debug/back_03_vlm_edge_hole_review.json",
            ws / "workspace/debug/back_03_vlm_edge_hole_review.json",
            ws / "debug/back_vlm_landmark_review.json",
            ws / "workspace/debug/back_vlm_landmark_review.json",
        ])
        if review_file is not None:
            try:
                review = json.loads(review_file.read_text(encoding="utf-8"))
                if review.get("review_source") != "vlm_visual_semantic_review":
                    errors.append("Back landmark review must come from VLM visual semantic review.")
                matches = review.get("matches")
                if not isinstance(matches, list) or len(matches) == 1:
                    errors.append("Back landmark review must contain zero matches for fallback or at least two semantic matches.")
                for match in matches or []:
                    if float(match.get("confidence", 0.0)) < 0.75:
                        errors.append("Back edge-hole matches require VLM confidence >= 0.75.")
            except Exception as e:  # noqa: BLE001
                errors.append(f"Invalid back VLM landmark review: {e}")
        if registration is not None and final is not None:
            try:
                reg = json.loads(registration.read_text(encoding="utf-8"))
                out = json.loads(final.read_text(encoding="utf-8"))
                target = reg.get("board_roi_target_px_approx")
                pixel = out.get("pixel")
                if not (isinstance(target, list) and len(target) == 2):
                    errors.append("back_board_registration.json missing board_roi_target_px_approx [x,y].")
                elif not (isinstance(pixel, list) and len(pixel) == 2):
                    errors.append("step08_result.json missing pixel [x,y].")
                elif abs(float(target[0]) - float(pixel[0])) > 1.0 or abs(float(target[1]) - float(pixel[1])) > 1.0:
                    errors.append("Back Step08 pixel must match back_board_registration target (±1px).")
                # Skip inlier-hole gate when registration did not use hole refinement
                # (fixed_outline / fixed_fixture orientation), or when orientation was
                # contract-locked via bottom_view_display contract.
                hole_refinement_attempted = (
                    reg.get("edge_hole_refinement_validation", {}) or {}
                ).get("attempted", True)
                if (
                    hole_refinement_attempted
                    and int(reg.get("inlier_hole_count", 0)) < 2
                    and reg.get("orientation_source") != "bottom_view_display_contract_tl_to_tl"
                ):
                    errors.append("Back registration requires at least two inlier mounting/tooling landmarks.")
                p95 = float(reg.get("p95_landmark_error_px", float("inf")))
                threshold = float(reg.get("landmark_acceptance_threshold_px", 0.0))
                if threshold <= 0 or p95 > threshold:
                    errors.append("Back registration landmark reprojection error exceeds its acceptance threshold.")
                if float(reg.get("orientation_score_margin", 0.0)) < 0.12:
                    errors.append("Back registration orientation remains ambiguous.")
            except Exception as e:  # noqa: BLE001
                errors.append(f"Invalid back-board registration artifacts: {e}")
        return errors

    def _validate_skill_contract(self) -> list[str]:
        """Check required debug artifacts before accepting finish() (no progress/*.md gate)."""
        errors: list[str] = []
        ws = self.cfg.workspace_dir.resolve()
        side_errors = self._validate_resolved_side_artifacts()
        if side_errors:
            return side_errors

        # Step 3 mapping evidence must be machine-generated.
        mapping_json_candidates = [
            ws / "debug" / "step03_mapping.json",
            ws / "workspace" / "debug" / "step03_mapping.json",
        ]
        mapping_json = next((p for p in mapping_json_candidates if p.exists()), None)
        mapping_method_early: str | None = None
        if mapping_json is not None:
            try:
                mapping_method_early = json.loads(
                    mapping_json.read_text(encoding="utf-8")
                ).get("mapping_method")
            except Exception:
                mapping_method_early = None
        if isinstance(mapping_method_early, str):
            mapping_method_early = mapping_method_early.strip() or None
        # When the model writes step03_mapping with only notes / omitting mapping_method,
        # fall back to aligned JSON produced by case12_graph (avoid legacy Step5–8 gates).
        inferred = self._infer_case12_mapping_method_from_workspace(ws)
        if mapping_method_early is None and inferred:
            mapping_method_early = inferred
        # If mapping method and aligned source disagree, trust aligned source to avoid
        # false finish rejections after Step8 is already produced.
        if (
            inferred in {"case12_step02_opencv_ic_align", "case12_step02_vlm_ic_align"}
            and mapping_method_early in {"case12_step02_opencv_ic_align", "case12_step02_vlm_ic_align"}
            and inferred != mapping_method_early
        ):
            mapping_method_early = inferred
        if mapping_method_early == "case12_step02_opencv_ic_align":
            errors.extend(self._validate_skill_contract_case12_opencv_ic_align(ws))
            return errors
        if mapping_method_early == "case12_step02_vlm_ic_align":
            errors.extend(self._validate_skill_contract_case12_vlm_ic_align(ws))
            return errors
        if mapping_method_early in {
            "front_board_outline_holes",
            "back_board_outline_holes",
        }:
            errors.extend(self._validate_skill_contract_back_board_registration(ws))
            return errors

        if mapping_json is None:
            errors.append("Missing required mapping artifact: debug/step03_mapping.json")
        else:
            try:
                mapping_obj = json.loads(mapping_json.read_text(encoding="utf-8"))
                required_keys = {
                    "locator_box", "board_box", "tp_locator_center",
                    "u", "v", "tp_prior_board",
                }
                missing_keys = sorted(k for k in required_keys if k not in mapping_obj)
                if missing_keys:
                    errors.append(
                        "step03_mapping.json missing keys: " + ", ".join(missing_keys)
                    )
                else:
                    def _is_num(v: Any) -> bool:
                        return isinstance(v, (int, float)) and math.isfinite(float(v))

                    def _is_point(v: Any, n: int) -> bool:
                        return (
                            isinstance(v, list)
                            and len(v) == n
                            and all(_is_num(x) for x in v)
                        )

                    locator_box = mapping_obj.get("locator_box")
                    board_box = mapping_obj.get("board_box")
                    tp_locator_center = mapping_obj.get("tp_locator_center")
                    tp_prior_board = mapping_obj.get("tp_prior_board")
                    u = mapping_obj.get("u")
                    v = mapping_obj.get("v")

                    if not _is_point(locator_box, 4):
                        errors.append("step03_mapping.json locator_box must be 4 numeric values.")
                    if not _is_point(board_box, 4):
                        errors.append("step03_mapping.json board_box must be 4 numeric values.")
                    if _is_point(locator_box, 4) and not (locator_box[2] > locator_box[0] and locator_box[3] > locator_box[1]):
                        errors.append("step03_mapping.json locator_box must satisfy x2>x1 and y2>y1.")
                    if _is_point(board_box, 4) and not (board_box[2] > board_box[0] and board_box[3] > board_box[1]):
                        errors.append("step03_mapping.json board_box must satisfy x2>x1 and y2>y1.")
                    if not _is_point(tp_locator_center, 2):
                        errors.append("step03_mapping.json tp_locator_center must be 2 numeric values.")
                    if not _is_point(tp_prior_board, 2):
                        errors.append("step03_mapping.json tp_prior_board must be 2 numeric values.")
                    if not _is_num(u) or not _is_num(v):
                        errors.append("step03_mapping.json u and v must be finite numbers.")
            except Exception as e:  # noqa: BLE001
                errors.append(f"Failed to parse step03_mapping.json: {e}")

        mapping_method_tag: str | None = None
        if mapping_json is not None:
            try:
                mapping_method_tag = json.loads(
                    mapping_json.read_text(encoding="utf-8")
                ).get("mapping_method")
            except Exception:
                mapping_method_tag = None
        if not (isinstance(mapping_method_tag, str) and mapping_method_tag.strip()):
            inferred = self._infer_case12_mapping_method_from_workspace(ws)
            if inferred:
                mapping_method_tag = inferred
        mapping_method_vlm = mapping_method_tag in (
            "vlm_neighborhood_layout_match",
            "case10_dual_roi_layout",
        )
        mapping_method_case10_layout = mapping_method_tag == "case10_dual_roi_layout"
        mapping_method_case12_layout = mapping_method_tag == "case12_step02_anchor_match"
        mapping_method_case12_direct = mapping_method_tag in (
            "case12_step02_opencv_ic_align",
            "case12_step02_vlm_ic_align",
        )

        def _first_existing(paths: list[Path]) -> Path | None:
            for q in paths:
                if q.exists():
                    return q
            return None

        s01 = _first_existing(
            [
                ws / "debug" / "step01_locator_front_anchor.png",
                ws / "workspace" / "debug" / "step01_locator_front_anchor.png",
            ]
        )
        s02_loc = _first_existing(
            [
                ws / "debug" / "step02_locator_front_anchor.png",
                ws / "workspace" / "debug" / "step02_locator_front_anchor.png",
            ]
        )
        s02_board = _first_existing(
            [
                ws / "debug" / "step02_board_front_anchor.png",
                ws / "workspace" / "debug" / "step02_board_front_anchor.png",
            ]
        )
        skip_step2_gate = mapping_method_tag in (
            "green_roi_template_match",
            "vlm_neighborhood_layout_match",
            "case10_dual_roi_layout",
        )

        if s01 is not None and mapping_json is not None and not skip_step2_gate:
            if s02_loc is None or s02_board is None:
                errors.append(
                    "Full-flow Step2 required: save BOTH debug/step02_locator_front_anchor.png "
                    "and debug/step02_board_front_anchor.png (red boxes) before Step3 / prior."
                )
            else:
                t01 = s01.stat().st_mtime
                t2 = min(s02_loc.stat().st_mtime, s02_board.stat().st_mtime)
                t3 = mapping_json.stat().st_mtime
                if t2 + 1e-3 < t01:
                    errors.append(
                        "step02_*_anchor.png must be generated after "
                        "step01_locator_front_anchor.png."
                    )
                if t3 + 0.5 < t2:
                    errors.append(
                        "step03_mapping.json must be newer than both step02 anchor PNGs "
                        "(run Step3 only after saving the two red-box images)."
                    )

        # Required debug artifacts (emphasized PNG/JSON only; no progress/*.md).
        required_debug: list[str] = [
            "debug/step03_mapping.json",
            "debug/step03_prior_on_board.png",
            "debug/step04_roi_crop.png",
            "debug/step05_candidates.json",
            "debug/step57_candidates_scored.png",
            "debug/step08_final_tp.png",
            "debug/step08_result.json",
        ]
        if mapping_method_vlm:
            required_debug.insert(
                3,
                (
                    "debug/step04_locator_roi_crop.png"
                    if mapping_method_case10_layout
                    else "debug/step04_locator_landmarks.png"
                ),
            )

        if mapping_method_case10_layout:
            case10_prefix = [
                "debug/case10_signal_to_tp.json",
                "debug/case10_target_tp_pdf_search.json",
                "debug/case10_assembly_drawing.png",
                "debug/case10_assembly_drawing_tp_marked.png",
                "debug/case10_assembly_largest_ic_box.png",
                "debug/case10_assembly_largest_ic.json",
                "debug/case10_largest_ic_box.png",
                "debug/case10_largest_ic.json",
                "debug/step02_locator_front_anchor.png",
                "debug/step02_board_front_anchor.png",
                "debug/board_tp_marked.png",
            ]
            tail = [x for x in required_debug if x not in case10_prefix]
            required_debug = case10_prefix + tail

            path_a = _first_existing(
                [
                    ws / "debug" / "case10_tp_dual_roi_direct_vlm.json",
                    ws / "workspace" / "debug" / "case10_tp_dual_roi_direct_vlm.json",
                ]
            )
            if path_a is not None:
                anchor = "debug/step03_mapping.json"
                insert_at = required_debug.index(anchor)
                for name in (
                    "debug/case10_dual_roi_locator_refs.json",
                    "debug/step04_locator_roi_refs.png",
                    "debug/case10_tp_dual_roi_direct_vlm.json",
                    "debug/step04_dual_roi_approx_only.png",
                    "debug/case10_tp_dual_roi_direct_refined.json",
                    "debug/step04_dual_roi_direct_snap.png",
                ):
                    required_debug.insert(insert_at, name)
                    insert_at += 1
            else:
                anchor = "debug/step03_mapping.json"
                insert_at = required_debug.index(anchor)
                for name in (
                    "debug/case10_tp_roi_layout_vlm.json",
                    "debug/case10_tp_roi_layout_hints.json",
                ):
                    required_debug.insert(insert_at, name)
                    insert_at += 1

        if mapping_method_case12_layout:
            case12_prefix = [
                "debug/step02_locator_front_anchor.png",
                "debug/step02_board_front_anchor.png",
                "debug/case12_step02_locator_graph.json",
                "debug/case12_board_largest_ic_bbox_vlm.json",
                "debug/case12_board_points_aligned.json",
                "debug/case12_board_points_vlm_refine.json",
                "debug/case12_board_points_refined.json",
                "debug/case12_board_approx_overlay.png",
            ]
            aligned_case12 = _first_existing(
                [
                    ws / "debug" / "case12_board_points_aligned.json",
                    ws / "workspace" / "debug" / "case12_board_points_aligned.json",
                ]
            )
            if aligned_case12 is not None and aligned_case12.is_file():
                try:
                    _al = json.loads(aligned_case12.read_text(encoding="utf-8"))
                    if _al.get("source") == "opencv_ic_bbox_isotropic_align":
                        case12_prefix = [
                            x
                            for x in case12_prefix
                            if x != "debug/case12_board_largest_ic_bbox_vlm.json"
                        ]
                except Exception:
                    pass
            tail = [x for x in required_debug if x not in case12_prefix]
            required_debug = case12_prefix + tail

        for rel in required_debug:
            p1 = ws / rel
            p2 = ws / "workspace" / rel
            if not (p1.exists() or p2.exists()):
                errors.append(f"Missing required debug artifact: {rel}")

        # Step8 final image must be on full board, not ROI-sized crop.
        step08 = next((p for p in [ws / "debug" / "step08_final_tp.png",
                                   ws / "workspace" / "debug" / "step08_final_tp.png"]
                       if p.exists()), None)
        step04_roi = next((p for p in [ws / "debug" / "step04_roi_crop.png",
                                       ws / "workspace" / "debug" / "step04_roi_crop.png"]
                           if p.exists()), None)
        step03_prior = next((p for p in [ws / "debug" / "step03_prior_on_board.png",
                                         ws / "workspace" / "debug" / "step03_prior_on_board.png"]
                             if p.exists()), None)
        if step08 is not None:
            try:
                from PIL import Image
                with Image.open(step08) as im8:
                    size8 = im8.size
                if step04_roi is not None:
                    with Image.open(step04_roi) as im4:
                        size4 = im4.size
                    if size8 == size4:
                        errors.append(
                            "step08_final_tp.png appears ROI-sized; final annotation must be on full board image."
                        )
                if step03_prior is not None:
                    with Image.open(step03_prior) as im3:
                        size3 = im3.size
                    if size8 != size3:
                        errors.append(
                            "step08_final_tp.png size mismatch with board-scale prior image "
                            "(expected same size as step03_prior_on_board.png)."
                        )
            except Exception as e:  # noqa: BLE001
                errors.append(f"Failed to validate step08_final_tp.png size: {e}")

        step03_mapping = next((p for p in [ws / "debug" / "step03_mapping.json",
                                           ws / "workspace" / "debug" / "step03_mapping.json"]
                               if p.exists()), None)
        step04_roi = next((p for p in [ws / "debug" / "step04_roi_crop.png",
                                       ws / "workspace" / "debug" / "step04_roi_crop.png"]
                           if p.exists()), None)
        step05_candidates_for_time = next((p for p in [ws / "debug" / "step05_candidates.json",
                                                       ws / "workspace" / "debug" / "step05_candidates.json"]
                                           if p.exists()), None)
        if not mapping_method_case12_direct:
            try:
                step04_lm = _first_existing(
                    [
                        ws / "debug" / "step04_locator_landmarks.png",
                        ws / "workspace" / "debug" / "step04_locator_landmarks.png",
                    ]
                )
                step04_loc_roi_crop = _first_existing(
                    [
                        ws / "debug" / "step04_locator_roi_crop.png",
                        ws / "workspace" / "debug" / "step04_locator_roi_crop.png",
                    ]
                )
                step03_loc_roi = _first_existing(
                    [
                        ws / "debug" / "step03_locator_roi.png",
                        ws / "workspace" / "debug" / "step03_locator_roi.png",
                    ]
                )
                if (
                    mapping_method_vlm
                    and not mapping_method_case10_layout
                    and step04_lm is not None
                    and step03_loc_roi is not None
                ):
                    if step04_lm.stat().st_mtime + 1e-3 < step03_loc_roi.stat().st_mtime:
                        errors.append(
                            "step04_locator_landmarks.png must be newer than "
                            "step03_locator_roi.png (run Step4 after Step3A)."
                        )
                if (
                    mapping_method_vlm
                    and not mapping_method_case10_layout
                    and step03_mapping is not None
                    and step04_lm is not None
                ):
                    if step03_mapping.stat().st_mtime + 1e-3 < step04_lm.stat().st_mtime:
                        errors.append(
                            "step03_mapping.json is older than step04_locator_landmarks.png. "
                            "Write prior/mapping only after Step4 locator landmarks."
                        )
                if (
                    mapping_method_case10_layout
                    and step03_mapping is not None
                    and step04_loc_roi_crop is not None
                ):
                    if step04_loc_roi_crop.stat().st_mtime + 1e-3 < step03_mapping.stat().st_mtime:
                        errors.append(
                            "step04_locator_roi_crop.png must be newer than step03_mapping.json "
                            "(take locator ROI after Step3 mapping / prior is fixed)."
                        )
                if step03_mapping is not None and step04_roi is not None:
                    t3 = step03_mapping.stat().st_mtime
                    t4 = step04_roi.stat().st_mtime
                    if t4 + 1e-3 < t3:
                        errors.append(
                            "step04_roi_crop.png is older than step03_mapping.json. "
                            "Regenerate the board ROI crop after Step3 mapping in this run."
                        )
                if step04_roi is not None and step05_candidates_for_time is not None:
                    t4 = step04_roi.stat().st_mtime
                    t5 = step05_candidates_for_time.stat().st_mtime
                    if t5 + 1e-3 < t4:
                        errors.append(
                            "step05_candidates.json is older than step04_roi_crop.png. "
                            "Step5/6/7 must run after Step4 ROI generation."
                        )
            except Exception as e:  # noqa: BLE001
                errors.append(f"Failed to validate Step3->Step4->Step5 artifact timeline: {e}")

        # Scheme C: Step5 candidates must provide stable global coordinates
        # + adaptive visualization radius, and Step8 should consume them.
        step05_candidates = next((p for p in [ws / "debug" / "step05_candidates.json",
                                              ws / "workspace" / "debug" / "step05_candidates.json"]
                                  if p.exists()), None)
        if step05_candidates is not None and not mapping_method_case12_direct:
            try:
                cand_obj = json.loads(step05_candidates.read_text(encoding="utf-8"))
                if isinstance(cand_obj, dict):
                    cand_list = cand_obj.get("candidates", [])
                elif isinstance(cand_obj, list):
                    cand_list = cand_obj
                else:
                    cand_list = []
                valid_candidates: list[dict[str, Any]] = []
                for c in cand_list:
                    if not isinstance(c, dict):
                        continue
                    if not all(k in c for k in ("id", "gx", "gy", "r_vis")):
                        continue
                    try:
                        gx = float(c["gx"])
                        gy = float(c["gy"])
                        rv = float(c["r_vis"])
                    except Exception:
                        continue
                    if not (math.isfinite(gx) and math.isfinite(gy) and math.isfinite(rv) and rv > 0):
                        continue
                    valid_candidates.append(c)

                if not valid_candidates:
                    errors.append(
                        "step05_candidates.json must include candidate schema fields "
                        "`id`, `gx`, `gy`, `r_vis` (finite values) so Step8 can reuse "
                        "global coordinates and adaptive marker radius."
                    )

                step08_result = next((p for p in [ws / "debug" / "step08_result.json",
                                                  ws / "workspace" / "debug" / "step08_result.json"]
                                      if p.exists()), None)
                if step08_result is None:
                    errors.append(
                        "Missing debug/step08_result.json with selected candidate id and marker radius."
                    )
                elif valid_candidates:
                    try:
                        result_obj = json.loads(step08_result.read_text(encoding="utf-8"))
                        selected_id = result_obj.get("selected_id")
                        marker_radius = result_obj.get("marker_radius")
                        pixel = result_obj.get("pixel")

                        if selected_id is None:
                            errors.append(
                                "step08_result.json missing `selected_id` "
                                "(must reference a candidate id from step05_candidates.json)."
                            )
                            selected_candidate = None
                        else:
                            selected_candidate = next(
                                (c for c in valid_candidates if str(c.get("id")) == str(selected_id)),
                                None,
                            )
                            if selected_candidate is None:
                                errors.append(
                                    "step08_result.json selected_id does not exist in Step5 candidates."
                                )

                        if marker_radius is None:
                            errors.append(
                                "step08_result.json missing `marker_radius` "
                                "(must reuse candidate `r_vis`)."
                            )
                        elif selected_candidate is not None:
                            try:
                                mr = float(marker_radius)
                                rv = float(selected_candidate["r_vis"])
                                if not math.isfinite(mr):
                                    errors.append("step08_result.json marker_radius must be finite.")
                                elif abs(mr - rv) > 1.0:
                                    errors.append(
                                        "step08_result.json marker_radius must match selected candidate "
                                        "`r_vis` (±1px tolerance)."
                                    )
                            except Exception:
                                errors.append("step08_result.json marker_radius must be numeric.")

                        if not (isinstance(pixel, list) and len(pixel) == 2):
                            errors.append(
                                "step08_result.json missing/invalid `pixel`; expected [x, y]."
                            )
                        elif selected_candidate is not None:
                            try:
                                px, py = float(pixel[0]), float(pixel[1])
                                gx = float(selected_candidate["gx"])
                                gy = float(selected_candidate["gy"])
                                if abs(px - gx) > 1.0 or abs(py - gy) > 1.0:
                                    errors.append(
                                        "step08_result.json pixel must match selected candidate "
                                        "global coordinate (`gx`,`gy`) (±1px tolerance)."
                                    )
                            except Exception:
                                errors.append("step08_result.json pixel must contain numeric values.")

                        # Verify Step8 rendered marker radius on final image
                        # is consistent with the selected candidate r_vis.
                        if (
                            step08 is not None
                            and selected_candidate is not None
                            and isinstance(pixel, list)
                            and len(pixel) == 2
                            and marker_radius is not None
                        ):
                            try:
                                from PIL import Image

                                with Image.open(step08) as im8:
                                    img = im8.convert("RGB")
                                    w, h = img.size
                                    pix = img.load()

                                px = int(round(float(pixel[0])))
                                py = int(round(float(pixel[1])))
                                mr = float(marker_radius)
                                if not math.isfinite(mr) or mr <= 0:
                                    raise ValueError("invalid marker_radius for image check")

                                # Search around the selected TP for red ring pixels.
                                # Exclude near-axis pixels to reduce cross-hair influence.
                                search_r = int(max(20, min(120, round(mr * 8 + 16))))
                                x1 = max(0, px - search_r)
                                y1 = max(0, py - search_r)
                                x2 = min(w - 1, px + search_r)
                                y2 = min(h - 1, py + search_r)

                                dists: list[float] = []
                                for yy in range(y1, y2 + 1):
                                    for xx in range(x1, x2 + 1):
                                        dx = xx - px
                                        dy = yy - py
                                        if abs(dx) <= 2 or abs(dy) <= 2:
                                            continue
                                        r, g, b = pix[xx, yy]
                                        is_red = (r >= 150) and (g <= 120) and (b <= 120) and (r - max(g, b) >= 35)
                                        if not is_red:
                                            continue
                                        d = math.hypot(dx, dy)
                                        if 1.5 <= d <= search_r:
                                            dists.append(d)

                                if len(dists) < 16:
                                    errors.append(
                                        "Unable to verify Step8 rendered marker radius from step08_final_tp.png; "
                                        "insufficient red ring pixels around selected point."
                                    )
                                else:
                                    dists.sort()
                                    est_r = dists[len(dists) // 2]
                                    # Keep tolerance moderate because anti-aliasing/line thickness
                                    # can shift the observed ring by a few pixels.
                                    if abs(est_r - mr) > 4.0:
                                        errors.append(
                                            "step08_final_tp.png rendered marker radius does not match "
                                            "step08_result.json marker_radius / candidate r_vis "
                                            f"(observed~{est_r:.1f}px vs expected~{mr:.1f}px)."
                                        )
                            except Exception as e:  # noqa: BLE001
                                errors.append(f"Failed to validate rendered Step8 marker radius: {e}")
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"Failed to validate step08_result.json: {e}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"Failed to parse step05_candidates.json: {e}")
        return errors

    # --------------------------- logging ----------------------------- #

    def _prepare_run_dir(self, name: str | None) -> Path:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = f"{stamp}" + (f"-{name}" if name else "")
        run_dir = self.cfg.workspace_dir / "runs" / slug
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _log_jsonl(self, run_dir: Path, filename: str, records: list[Any]) -> None:
        with open(run_dir / filename, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def _append_step_record(self, run_dir: Path, record: dict[str, Any]) -> None:
        with open(run_dir / "step_records.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _snapshot_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            return json.loads(json.dumps(messages, ensure_ascii=False, default=str))
        except Exception:  # noqa: BLE001
            return list(messages)

    def _persist_run(self, run_dir: Path, result: AgentRun,
                     messages: list[dict[str, Any]]) -> None:
        summary = {
            "task_question": result.task_question,
            "stopped_reason": result.stopped_reason,
            "final_answer": result.final_answer,
            "last_error": result.last_error,
            "part_timing": result.part_timing,
            "steps": [
                {
                    "index": s.index,
                    "assistant_content": s.assistant_content,
                    "tool_calls": s.tool_calls,
                    "tool_results": s.tool_results,
                    "timing": s.timing,
                    "tool_timing": s.tool_timing,
                    "usage": s.usage,
                }
                for s in result.steps
            ],
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self._log_jsonl(run_dir, "messages.final.jsonl", messages)

    def _infer_step_part(self, step: AgentStep, prev_part: str | None = None) -> str:
        blob_parts: list[str] = [step.assistant_content or ""]
        for c in step.tool_calls:
            nm = c.get("name")
            if nm:
                blob_parts.append(str(nm))
            try:
                blob_parts.append(json.dumps(c.get("arguments", {}), ensure_ascii=False))
            except Exception:  # noqa: BLE001
                blob_parts.append(str(c.get("arguments", "")))
        for r in step.tool_results:
            blob_parts.append(str(r.get("text", "")))
        blob = "\n".join(blob_parts).lower()

        patterns: list[tuple[str, list[str]]] = [
            ("partd", [
                "case12_", "step08_", "step03_mapping", "mapping_method",
                "run_align_locator_graph_to_board", "case12_step02",
            ]),
            ("partc", [
                "step02_locator_front_anchor", "step02_board_front_anchor",
                "board_tp_marked",
            ]),
            ("parta", [
                "case10_largest_ic_box", "case10_largest_ic.json",
                "case10_vlm_hints.json",
            ]),
            ("partb", [
                "case10_assembly_largest_ic_box", "case10_assembly_largest_ic.json",
                "case10_assembly_vlm_hints", "stepb3",
            ]),
            ("part0", [
                "case10_signal_to_tp", "case10_target_tp_pdf_search",
                "case10_assembly_drawing_tp_marked",
                "search_pdf_text",
            ]),
        ]
        for label, kws in patterns:
            if any(k in blob for k in kws):
                return label

        # Fallback for malformed <tool_call> finish loops.
        if re.search(r"<tool_call>\s*\{\s*\"name\"\s*:\s*\"finish\"", blob):
            return "partd"
        if prev_part:
            return prev_part
        return "unknown"

    def _compute_part_timing(self, result: AgentRun) -> dict[str, float]:
        out: dict[str, float] = {
            "part0_s": 0.0,
            "partb_s": 0.0,
            "parta_s": 0.0,
            "partc_s": 0.0,
            "partd_s": 0.0,
            "unknown_s": 0.0,
            "total_s": 0.0,
        }
        prev_part: str | None = None
        for s in result.steps:
            t = float((s.timing or {}).get("step_total_s", 0.0) or 0.0)
            out["total_s"] += t
            part = self._infer_step_part(s, prev_part=prev_part)
            prev_part = part if part != "unknown" else prev_part
            key = {
                "part0": "part0_s",
                "partb": "partb_s",
                "parta": "parta_s",
                "partc": "partc_s",
                "partd": "partd_s",
            }.get(part, "unknown_s")
            out[key] += t
        return {k: round(v, 4) for k, v in out.items()}

    def _render_part_timing_lines(self, part_timing: dict[str, float]) -> str:
        if not part_timing:
            return ""
        return (
            "[bold]part timing[/bold]\n"
            f"- part0 = {part_timing.get('part0_s', 0.0):.2f}s\n"
            f"- partB = {part_timing.get('partb_s', 0.0):.2f}s\n"
            f"- partA = {part_timing.get('parta_s', 0.0):.2f}s\n"
            f"- partC = {part_timing.get('partc_s', 0.0):.2f}s\n"
            f"- partD = {part_timing.get('partd_s', 0.0):.2f}s\n"
            f"- unknown = {part_timing.get('unknown_s', 0.0):.2f}s\n"
            f"- total = {part_timing.get('total_s', 0.0):.2f}s"
        )

    # --------------------------- rendering --------------------------- #

    def _render_assistant(self, step: int, reply: AssistantReply, dt: float) -> None:
        body = reply.content.strip() or "(no text, only tool calls)"
        self.console.print(Panel(
            truncate(body, 1500),
            title=f"[cyan]assistant · step {step} · {dt:.2f}s · {len(reply.tool_calls)} tool-call(s)[/cyan]",
            border_style="cyan",
        ))

    def _render_step_timing(
        self, step: int, timing: dict[str, float], tool_timing: list[dict[str, Any]]
    ) -> None:
        parts = [
            f"llm={timing.get('llm_s', 0.0):.2f}s",
            f"tools={timing.get('tools_s', 0.0):.2f}s",
            f"attach={timing.get('images_attach_s', 0.0):.2f}s",
            f"total={timing.get('step_total_s', 0.0):.2f}s",
        ]
        llm_sub = (
            "prep={:.2f}s api={:.2f}s retry_sleep={:.2f}s parse={:.2f}s".format(
                float(timing.get("llm_prep_s", 0.0)),
                float(timing.get("llm_api_call_s", 0.0)),
                float(timing.get("llm_retry_sleep_s", 0.0)),
                float(timing.get("llm_parse_s", 0.0)),
            )
            if any(
                k in timing
                for k in (
                    "llm_prep_s",
                    "llm_api_call_s",
                    "llm_retry_sleep_s",
                    "llm_parse_s",
                )
            )
            else "(not available)"
        )
        llm_est = (
            "est_prefill={:.2f}s est_decode={:.2f}s completion_tokens={:.0f} (coarse)".format(
                float(timing.get("llm_est_prefill_s", 0.0)),
                float(timing.get("llm_est_decode_s", 0.0)),
                float(timing.get("llm_completion_tokens", 0.0)),
            )
            if any(
                k in timing
                for k in (
                    "llm_est_prefill_s",
                    "llm_est_decode_s",
                    "llm_completion_tokens",
                )
            )
            else "(not available)"
        )
        if tool_timing:
            tool_str = ", ".join(
                f"{t.get('name')}={float(t.get('duration_s', 0.0)):.2f}s"
                for t in tool_timing
            )
        else:
            tool_str = "(none)"
        self.console.print(Panel(
            f"[bold]breakdown[/bold]: {' | '.join(parts)}\n"
            f"[bold]llm detail[/bold]: {llm_sub}\n"
            f"[bold]llm estimate[/bold]: {llm_est}\n"
            f"[bold]tools detail[/bold]: {tool_str}",
            title=f"[magenta]step {step} timing[/magenta]",
            border_style="magenta",
        ))

    def _render_tool(self, call: ToolInvocation, result: ToolResult) -> None:
        args_repr = truncate(json.dumps(call.arguments, ensure_ascii=False), 400)
        style = "green" if result.ok else "red"
        self.console.print(Panel(
            f"[bold]args[/bold]: {args_repr}\n\n"
            f"{truncate(result.text, 1500)}",
            title=f"[{style}]tool · {call.name}[/] {'[OK]' if result.ok else '[FAIL]'}"
                  + (" [final]" if result.is_final else ""),
            border_style=style,
        ))
