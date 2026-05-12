"""
omnirca/agent/react_agent.py — ReAct (Reasoning + Acting) agent core.

The agent:
  1. Receives an incident query (free text) + a fault window (t_start, t_end).
  2. Runs a tool-augmented LLM conversation loop using all 24 tools.
  3. Terminates when finalize_rca is called OR max_steps is reached.
  4. Returns a structured AgentResult dict.

Design notes:
  - Uses OpenAI function-calling API (tools parameter) — more reliable than
    text-based ReAct parsing and requires no output-format enforcement.
  - Tool results are truncated to 4 000 chars before being added to context
    to prevent token exhaustion on long data tables.
  - DataFrame objects in ToolResult.data are not passed to the LLM;
    only ToolResult.summary (the human-readable string) is used.
  - Conversation history is preserved for debugging / Phase 7 evaluation.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from omnirca.llm_client import chat
from omnirca.agent.prompts import SYSTEM_PROMPT
from omnirca.agent.tool_schema import TOOL_SCHEMAS, dispatch_tool
from omnirca.tools.base import ToolResult
from omnirca.logger import get_logger

_log = get_logger("omnirca.react_agent")

# Maximum characters of a tool result passed back to the LLM per call.
_MAX_TOOL_RESULT_CHARS = 4_000


@dataclass
class ToolCallRecord:
    step:           int
    tool:           str
    args:           dict
    result_summary: str
    duration_ms:    float


@dataclass
class AgentResult:
    """Structured output from a single agent run."""
    # Core verdict (None if agent never called finalize_rca)
    verdict:           ToolResult | None

    # Run metadata
    steps_taken:       int
    reached_conclusion: bool
    total_duration_s:  float

    # Detailed trace
    tool_calls:        list[ToolCallRecord] = field(default_factory=list)
    conversation:      list[dict]           = field(default_factory=list)

    # Convenience accessors
    @property
    def root_cause(self) -> str | None:
        return self.verdict.data.get("root_cause") if self.verdict else None

    @property
    def fault_category(self) -> str | None:
        return self.verdict.data.get("fault_category") if self.verdict else None

    @property
    def confidence(self) -> str | None:
        return self.verdict.data.get("confidence") if self.verdict else None

    def summary(self) -> str:
        if not self.reached_conclusion:
            return (
                f"[NO CONCLUSION] Agent ran {self.steps_taken} steps "
                f"({self.total_duration_s:.1f}s) without calling finalize_rca."
            )
        return (
            f"Root cause: {self.root_cause}  |  "
            f"Category: {self.fault_category}  |  "
            f"Confidence: {self.confidence}  |  "
            f"Steps: {self.steps_taken}  |  "
            f"Time: {self.total_duration_s:.1f}s"
        )


class ReActAgent:
    """
    Tool-augmented LLM agent for root cause analysis.

    Parameters
    ----------
    max_steps : int
        Maximum number of LLM ↔ tool iterations before forced termination.
        Default 25 — enough for thorough investigation.
    verbose : bool
        If True, print each tool call and result to stdout as the agent runs.
    """

    def __init__(self, max_steps: int = 25, verbose: bool = False) -> None:
        self.max_steps = max_steps
        self.verbose   = verbose

    def run(
        self,
        incident_query: str,
        t_start:        str,
        t_end:          str,
    ) -> AgentResult:
        """
        Run the RCA agent on a single incident.

        Parameters
        ----------
        incident_query : str
            Natural-language description of the observed symptoms / incident.
        t_start : str
            Start of the fault window ("2024-01-01 HH:MM:SS").
        t_end : str
            End of the fault window ("2024-01-01 HH:MM:SS").

        Returns
        -------
        AgentResult with the verdict, tool call trace, and conversation history.
        """
        run_start = time.monotonic()

        _log.info("=== ReActAgent started | window %s → %s ===", t_start, t_end)

        # ── Build initial conversation ────────────────────────────────────────
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Incident report: {incident_query}\n"
                    f"Fault window: {t_start}  →  {t_end}\n\n"
                    f"Investigate this incident systematically and identify the "
                    f"root cause service and fault category. "
                    f"Remember to call syscall_multi_service_compare FIRST, "
                    f"then follow the investigation protocol. "
                    f"End by calling finalize_rca."
                ),
            },
        ]

        verdict:       ToolResult | None = None
        tool_call_log: list[ToolCallRecord] = []
        step = 0

        # ── ReAct loop ────────────────────────────────────────────────────────
        while step < self.max_steps:
            response = chat(messages, tools=TOOL_SCHEMAS, tool_choice="auto")
            message  = response.choices[0].message

            # Add assistant turn to history
            assistant_msg: dict[str, Any] = {
                "role":    "assistant",
                "content": message.content or "",
            }
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id":   tc.id,
                        "type": "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            messages.append(assistant_msg)

            # Terminate if the model returned plain text (no tool calls)
            if not message.tool_calls:
                if self.verbose:
                    print(f"\n[Agent] Text response (no tool call) — ending loop.\n"
                          f"{message.content}")
                _log.info("[ReActAgent] LLM returned text (no tool calls) — loop ended at step %d", step + 1)
                break

            # Dispatch each tool call in the response
            for tc in message.tool_calls:
                tool_name = tc.function.name

                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                if self.verbose:
                    print(f"\n[Step {step + 1}] → {tool_name}({_fmt_args(args)})")

                _log.info("[ReActAgent] [step %d] TOOL %s | %s", step + 1, tool_name, _fmt_args(args))

                call_start = time.monotonic()
                result     = dispatch_tool(tool_name, args)
                call_ms    = (time.monotonic() - call_start) * 1000

                if self.verbose:
                    print(f"   \u2190 {result.summary[:300]}")

                snippet = result.summary[:200].replace("\n", " ")
                _log.info("[ReActAgent] [step %d] RESULT (%.0fms, %d chars): %s",
                          step + 1, call_ms, len(result.summary), snippet)

                # Truncate result to avoid token exhaustion
                result_text = result.summary[:_MAX_TOOL_RESULT_CHARS]
                if len(result.summary) > _MAX_TOOL_RESULT_CHARS:
                    result_text += "\n[... result truncated ...]"

                # Add tool result to conversation
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_text,
                })

                # Record the call
                tool_call_log.append(ToolCallRecord(
                    step           = step + 1,
                    tool           = tool_name,
                    args           = args,
                    result_summary = result.summary[:500],
                    duration_ms    = round(call_ms, 1),
                ))

                # Capture finalize_rca verdict
                if tool_name == "finalize_rca":
                    verdict = result
                    _log.info("[ReActAgent] finalize_rca called | root=%s | category=%s | confidence=%s",
                              verdict.data.get("root_cause", "?"),
                              verdict.data.get("fault_category", "?"),
                              verdict.data.get("confidence", "?"))

            if verdict is not None:
                break

            step += 1

        total_s = time.monotonic() - run_start
        _log.info("=== ReActAgent done | steps=%d, duration=%.1fs, concluded=%s ===",
                  step + 1, total_s, verdict is not None)

        return AgentResult(
            verdict            = verdict,
            steps_taken        = step + 1 if message.tool_calls else step,
            reached_conclusion = verdict is not None,
            total_duration_s   = round(total_s, 2),
            tool_calls         = tool_call_log,
            conversation       = messages,
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_args(args: dict) -> str:
    """Format tool args for verbose display."""
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 40:
            v = v[:37] + "..."
        elif isinstance(v, list) and len(v) > 5:
            v = v[:5] + ["..."]
        parts.append(f"{k}={v!r}")
    return ", ".join(parts)
