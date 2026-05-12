"""
omnirca/agents/fault_typer.py — FaultTyper sub-agent.

Role: Fault Classification and SOP Retrieval.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Takes DataDetective's sub-channel analysis and produces:
  fault_category   — best-matching fault archetype from the encyclopedia
  sop_steps        — the diagnostic SOP for that fault category
  pattern_notes    — reasoning about which sub-channel patterns matched

The FaultTyper only classifies — it does NOT make a root cause decision.

Tools available: search_fault_knowledge, classify_fault_pattern, check_sop
"""
from __future__ import annotations

import json
import re
from typing import Any

from omnirca.agents.base_agent import SubAgent
from omnirca.agent.tool_schema import TOOL_SCHEMAS, _TOOL_MAP


_FAULT_TOOL_NAMES = {"search_fault_knowledge", "classify_fault_pattern", "check_sop"}
_SCHEMAS          = [s for s in TOOL_SCHEMAS if s["function"]["name"] in _FAULT_TOOL_NAMES]
_TOOL_MAP_LOCAL   = {k: v for k, v in _TOOL_MAP.items() if k in _FAULT_TOOL_NAMES}


_SYSTEM_PROMPT = """\
You are FaultTyper, a specialist SRE agent for FAULT CLASSIFICATION.

Your job is to classify WHAT TYPE of fault occurred, given data signals from
DataDetective's syscall analysis. You do NOT decide the root cause service.

CLASSIFICATION APPROACH:
1. Formulate a `symptom_text` from the sub_channel_hints you received.
   Include: dominant syscall types elevated, count vs. duration patterns,
   error rate direction, latency direction.
2. Call search_fault_knowledge(symptom_text) → 2 best-matching fault archetypes.
3. Call classify_fault_pattern(signals_dict) with the quantitative signal dict.
4. Based on the top match, call check_sop(fault_category) to retrieve the SOP.

SUB-CHANNEL PATTERN GUIDE (use to formulate symptom_text):
• read_count + write_count elevated, errors high, duration extreme → data_race_condition / stale_cache
• mmap count rising, duration long, memory_slope > 0 → memory_leak
• recv_count spike, socket_errors, send_count dropping → thread_pool_exhaustion
• fsync_count + write_duration both extreme → disk_io_saturation
• read_count high (repeated auth lookups), system-wide secondary elevation → authentication_failure
• p99 latency extreme at one service, fan-out to many dependents → cascading_timeout
• write_errors escalating, CPU flat, lock waits → transaction_deadlock

OUTPUT FORMAT (after tool calls, write plain text):
FAULT_CATEGORY: fault_type_name
FAULT_CONFIDENCE: HIGH|MEDIUM|LOW
SOP_RETRIEVED: YES|NO
PATTERN_NOTES: brief explanation of which signals matched which fault type
ANALYSIS_COMPLETE
"""


class FaultTyper(SubAgent):
    AGENT_NAME    = "FaultTyper"
    SYSTEM_PROMPT = _SYSTEM_PROMPT
    TOOL_SCHEMAS  = _SCHEMAS
    TOOL_MAP      = _TOOL_MAP_LOCAL
    MAX_STEPS     = 6

    def _build_user_message(self, context: dict) -> str:
        sub_hints            = context.get("sub_channel_hints", "unknown")
        top_score            = context.get("top_score", 0.0)
        mem_slope            = context.get("memory_slope", None)
        top_svc              = context.get("top_service", "unknown")
        channel_stats_by_svc = context.get("channel_stats_by_service", {})
        # Prefer channel_stats for GE's confirmed root over DD's top (which may be a victim)
        top_candidate = context.get("top_candidate", top_svc)
        if top_candidate and top_candidate in channel_stats_by_svc:
            channel_stats = channel_stats_by_svc[top_candidate]
        else:
            channel_stats = context.get("channel_stats", {})
        chan_str = json.dumps(channel_stats, indent=2) if channel_stats else "not available"
        return (
            f"Fault window: {context['t_start']} → {context['t_end']}\n\n"
            f"DataDetective sub-channel analysis for confirmed root service '{top_candidate}':\n"
            f"  Sub-channel hints: {sub_hints}\n"
            f"  Composite score:   {top_score}\n"
            f"  Memory slope:      {mem_slope}\n"
            f"  Channel stats (quantitative per-syscall breakdown):\n{chan_str}\n\n"
            f"Classify the fault type and retrieve the appropriate SOP.\n"
            f"Step 1: call search_fault_knowledge with a symptom_text from the hints above.\n"
            f"Step 2: call classify_fault_pattern(signals={{\"channel_stats\": <channel_stats above>, "
            f"\"memory_slope_mb_per_s\": {mem_slope}}}). "
            f"The argument name is exactly 'signals' (not 'signals_dict').\n"
            f"Step 3: call check_sop for the winning fault category."
        )

    def _parse_findings(
        self,
        messages: list[dict],
        tool_calls: list[dict],
        final_text: str,
        context: dict,
    ) -> dict[str, Any]:
        findings: dict[str, Any] = {
            "fault_category":     "unknown",
            "fault_confidence":   "LOW",
            "sop_steps":          "",
            "pattern_notes":      "",
            "raw_summary":        final_text,
        }

        m = re.search(r"FAULT_CATEGORY:\s*(\S+)", final_text)
        if m:
            findings["fault_category"] = m.group(1).strip()

        m = re.search(r"FAULT_CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", final_text)
        if m:
            findings["fault_confidence"] = m.group(1).strip()

        m = re.search(r"PATTERN_NOTES:\s*(.+?)(?:\n|ANALYSIS_COMPLETE|$)", final_text, re.DOTALL)
        if m:
            findings["pattern_notes"] = m.group(1).strip()

        # Extract SOP text from check_sop tool call result
        for tc in tool_calls:
            if tc["tool"] == "check_sop":
                findings["sop_steps"] = tc.get("result", "")
                break

        # Fallback: extract fault category from search_fault_knowledge result
        if findings["fault_category"] == "unknown":
            for tc in tool_calls:
                if tc["tool"] == "search_fault_knowledge":
                    m2 = re.search(r"\btype=(\w+)\b", tc.get("result", ""))
                    if m2:
                        findings["fault_category"] = m2.group(1)
                        break

        return findings
