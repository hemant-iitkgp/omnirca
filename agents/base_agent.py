"""
omnirca/agents/base_agent.py — Shared base class for all sub-agents.

Every sub-agent in the Phase 5 system inherits from SubAgent.
SubAgent wraps a single-purpose LLM prompt + a fixed set of tools,
providing a uniform `run(context) -> SubAgentReport` interface that the
MainAgent orchestrator can call without knowing each agent's internals.

Key design choices
──────────────────
• Each sub-agent has its OWN system prompt tuned to its narrow role.
• Each sub-agent can only call the tools relevant to its domain — not all 24.
  This reduces hallucination by shrinking the available action space.
• Sub-agents do NOT call finalize_rca — only MainAgent does.
• The `context` dict is passed in by MainAgent and carries the accumulated
  findings from previously-run sub-agents (so later agents can build on them).
• Tool results > 3000 chars are truncated (smaller budget than MainAgent).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from omnirca.llm_client import chat
from omnirca.tools.base import ToolResult
from omnirca.logger import get_logger


_MAX_TOOL_CHARS = 3_000   # tighter than main agent — sub-agents stay focused


def _short_args(args: dict) -> str:
    """Format tool args as a short key=value string for log lines."""
    parts = []
    for k, v in args.items():
        sv = str(v)
        parts.append(f"{k}={sv[:60]!r}" if len(sv) > 60 else f"{k}={sv!r}")
    return ", ".join(parts)


@dataclass
class SubAgentReport:
    """Structured output from a single sub-agent run."""
    agent_name:    str
    findings:      dict[str, Any]        # key results produced by the agent
    tool_calls:    list[dict]            # tool-call trace for debugging
    steps_taken:   int
    duration_s:    float
    raw_conclusion: str                  # the LLM's final text when it stopped tool-calling

    def get(self, key: str, default: Any = None) -> Any:
        return self.findings.get(key, default)

    def __repr__(self) -> str:
        return (
            f"<SubAgentReport agent={self.agent_name!r} "
            f"steps={self.steps_taken} "
            f"keys={list(self.findings.keys())}>"
        )


class SubAgent:
    """
    Base class for all Phase 5 sub-agents.

    Subclasses must define:
      AGENT_NAME  : str          — short label for logging
      SYSTEM_PROMPT: str         — role-specific system prompt
      TOOL_SCHEMAS : list[dict]  — subset of the 24 schemas this agent may call
      TOOL_MAP     : dict        — name → callable for the tools above
      MAX_STEPS    : int         — maximum tool call iterations
    """

    AGENT_NAME:   str       = "base"
    SYSTEM_PROMPT: str      = "You are an SRE agent."
    TOOL_SCHEMAS:  list     = []
    TOOL_MAP:      dict     = {}
    MAX_STEPS:     int      = 8

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def _dispatch(self, name: str, args: dict) -> ToolResult:
        fn = self.TOOL_MAP.get(name)
        if fn is None:
            return ToolResult(
                summary=f"[SubAgent] Tool '{name}' not available to {self.AGENT_NAME}.",
                data={},
            )
        try:
            return fn(**args)
        except Exception as exc:
            return ToolResult(
                summary=f"[SubAgent] Tool '{name}' raised: {exc}",
                data={"error": str(exc)},
            )

    def run(self, context: dict[str, Any]) -> SubAgentReport:
        """
        Run this sub-agent given the current investigation context.

        `context` must always contain:
            t_start (str), t_end (str)
        and may additionally contain findings from earlier sub-agents.
        """
        t_start = context["t_start"]
        t_end   = context["t_end"]

        log = get_logger(f"omnirca.agent.{self.AGENT_NAME}")
        log.info("[%s] started | window %s → %s", self.AGENT_NAME, t_start, t_end)

        user_msg = self._build_user_message(context)

        messages: list[dict] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

        tool_call_log: list[dict] = []
        step  = 0
        final_text = ""

        run_start = time.monotonic()

        while step < self.MAX_STEPS:
            try:
                response = chat(
                    messages,
                    tools=self.TOOL_SCHEMAS if self.TOOL_SCHEMAS else None,
                    tool_choice="auto" if self.TOOL_SCHEMAS else "none",
                    temperature=0.0,
                    max_tokens=1024,
                )
            except Exception as exc:
                log.error("[%s] LLM call failed at step %d: %s", self.AGENT_NAME, step + 1, exc)
                final_text = f"LLM call failed: {exc}"
                break

            msg = response.choices[0].message

            assistant_entry: dict[str, Any] = {
                "role":    "assistant",
                "content": msg.content or "",
            }
            if msg.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id":   tc.id,
                        "type": "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_entry)

            if not msg.tool_calls:
                final_text = msg.content or ""
                break

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                if self.verbose:
                    print(f"  [{self.AGENT_NAME}:{step+1}] {name}({list(args.keys())})")

                log.info("[%s] [step %d] TOOL %s | %s", self.AGENT_NAME, step + 1, name, _short_args(args))

                result = self._dispatch(name, args)
                result_text = result.summary[:_MAX_TOOL_CHARS]

                snippet = result.summary[:200].replace("\n", " ")
                log.info("[%s] [step %d] RESULT (%d chars): %s", self.AGENT_NAME, step + 1, len(result.summary), snippet)

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_text,
                })

                tool_call_log.append({
                    "step": step + 1,
                    "tool": name,
                    "args": args,
                    "result": result.summary[:200],
                    "result_data": result.data,
                })

            step += 1

        duration = time.monotonic() - run_start
        try:
            findings = self._parse_findings(messages, tool_call_log, final_text, context)
        except Exception as exc:
            log.error("[%s] _parse_findings failed: %s — using raw text", self.AGENT_NAME, exc)
            findings = {"summary": final_text, "raw_parse_error": str(exc)}

        log.info(
            "[%s] completed | steps=%d, duration=%.1fs | findings=%s",
            self.AGENT_NAME, step, duration, list(findings.keys()),
        )

        return SubAgentReport(
            agent_name=self.AGENT_NAME,
            findings=findings,
            tool_calls=tool_call_log,
            steps_taken=step,
            duration_s=round(duration, 2),
            raw_conclusion=final_text,
        )

    def _build_user_message(self, context: dict) -> str:
        """Override in subclass to build a role-specific user message."""
        return (
            f"Fault window: {context['t_start']} → {context['t_end']}\n"
            f"Context so far: {json.dumps({k: str(v)[:200] for k, v in context.items()}, indent=2)}\n\n"
            f"Perform your analysis and provide your findings."
        )

    def _parse_findings(
        self,
        messages: list[dict],
        tool_calls: list[dict],
        final_text: str,
        context: dict,
    ) -> dict[str, Any]:
        """
        Override in subclass to extract structured findings from the LLM output.
        Default: returns the final text as 'summary'.
        """
        return {"summary": final_text, "tool_call_count": len(tool_calls)}
