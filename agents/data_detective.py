"""
omnirca/agents/data_detective.py — DataDetective sub-agent.

Role: Anomaly Discovery across all 20 services.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The DataDetective is ALWAYS the first sub-agent called by MainAgent.
It performs the mandatory syscall-first triage, producing:
  1. ranked_services   — all anomalous services sorted by syscall z-score
  2. top_service       — the single most anomalous service
  3. sub_channel       — which syscall types are elevated (read/write/mmap/fsync…)
  4. anomaly_scores    — composite anomaly_score for top candidates

Tools available: syscall_multi_service_compare, syscall_sub_channel_analysis,
                 compute_anomaly_score, detect_memory_slope
"""
from __future__ import annotations

import re
from typing import Any

from omnirca.agents.base_agent import SubAgent, SubAgentReport
from omnirca.agent.tool_schema import TOOL_SCHEMAS, _TOOL_MAP
from omnirca.tools.base import ToolResult


# ── Tool subset (4 tools) ─────────────────────────────────────────────────────
_DETECTIVE_TOOL_NAMES = {
    "syscall_multi_service_compare",
    "syscall_sub_channel_analysis",
    "compute_anomaly_score",
    "detect_memory_slope",
}
_SCHEMAS  = [s for s in TOOL_SCHEMAS if s["function"]["name"] in _DETECTIVE_TOOL_NAMES]
_TOOL_MAP_LOCAL = {k: v for k, v in _TOOL_MAP.items() if k in _DETECTIVE_TOOL_NAMES}


_SYSTEM_PROMPT = """\
You are DataDetective, a specialist SRE agent responsible for ANOMALY DISCOVERY.

Your ONLY job is to identify which services are anomalous in the given fault window.

CRITICAL RULES:
1. Your FIRST action MUST be syscall_multi_service_compare(t_start, t_end).
   Syscalls are the ONLY reliable signal — do NOT start with metrics or logs.
2. After finding top anomalous services, call syscall_sub_channel_analysis on
   each of the TOP 2 anomalous services (skip "logging" — it is never root cause).
3. Call compute_anomaly_score on each top service to get the composite score.
4. If memory_leak is suspected (mmap count rising), call detect_memory_slope.
5. If syscall_multi_service_compare shows only "logging" as anomalous, look at
   the SECOND-ranked service — logging is always filtered in SRE topology rules.

OUTPUT FORMAT (after all tool calls, write a plain-text summary):
RANKED_SERVICES: [service_name, service_name, ...]     ← sorted by z-score desc
TOP_SERVICE: service_name
TOP_SCORE: float
SUB_CHANNEL_HINTS: text description of dominant syscall types elevated
MEMORY_SLOPE: float (or NONE if not measured)

Always end with "ANALYSIS_COMPLETE".
"""


class DataDetective(SubAgent):
    AGENT_NAME    = "DataDetective"
    SYSTEM_PROMPT = _SYSTEM_PROMPT
    TOOL_SCHEMAS  = _SCHEMAS
    TOOL_MAP      = _TOOL_MAP_LOCAL
    MAX_STEPS     = 8

    def _build_user_message(self, context: dict) -> str:
        return (
            f"Fault window: {context['t_start']} → {context['t_end']}\n\n"
            f"Run your anomaly discovery scan across all 20 services.\n"
            f"Start with syscall_multi_service_compare, then drill into the top services."
        )

    def _parse_findings(
        self,
        messages: list[dict],
        tool_calls: list[dict],
        final_text: str,
        context: dict,
    ) -> dict[str, Any]:
        findings: dict[str, Any] = {
            "ranked_services":   [],
            "top_service":       None,
            "top_score":         0.0,
            "sub_channel_hints": "",
            "memory_slope":      None,
            "channel_stats":     {},
            "raw_summary":       final_text,
        }

        # Extract structured fields from final_text
        m = re.search(r"RANKED_SERVICES:\s*\[([^\]]+)\]", final_text)
        if m:
            findings["ranked_services"] = [
                s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()
            ]

        m = re.search(r"TOP_SERVICE:\s*(\S+)", final_text)
        if m:
            findings["top_service"] = m.group(1).strip()

        m = re.search(r"TOP_SCORE:\s*([\d.]+)", final_text)
        if m:
            findings["top_score"] = float(m.group(1))

        m = re.search(r"SUB_CHANNEL_HINTS:\s*(.+?)(?:\n|$)", final_text)
        if m:
            findings["sub_channel_hints"] = m.group(1).strip()

        m = re.search(r"MEMORY_SLOPE:\s*([\d.\-]+|NONE)", final_text)
        if m:
            val = m.group(1).strip()
            if val == "NONE" or val == "-" or val == "":
                findings["memory_slope"] = None
            else:
                try:
                    findings["memory_slope"] = float(val)
                except ValueError:
                    findings["memory_slope"] = None

        # Extract channel_stats from all syscall_sub_channel_analysis calls (per-service)
        channel_stats_by_service: dict[str, dict] = {}
        for tc in tool_calls:
            if tc["tool"] == "syscall_sub_channel_analysis":
                svc  = tc["args"].get("service", "")
                data = tc.get("result_data", {})
                if svc and data and "channel_stats" in data:
                    channel_stats_by_service[svc] = data["channel_stats"]

        findings["channel_stats_by_service"] = channel_stats_by_service
        # Default channel_stats = first service analysed
        if channel_stats_by_service:
            findings["channel_stats"] = next(iter(channel_stats_by_service.values()))

        # Fallback: extract top_service from tool call trace
        if not findings["top_service"] and tool_calls:
            for tc in tool_calls:
                if tc["tool"] == "syscall_sub_channel_analysis":
                    svc = tc["args"].get("service")
                    if svc and svc != "logging":
                        findings["top_service"] = svc
                        break

        return findings
