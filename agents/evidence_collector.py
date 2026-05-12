"""
omnirca/agents/evidence_collector.py — EvidenceCollector sub-agent.

Role: Multi-Modal Evidence Corroboration.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Takes the top-2 root-cause candidates from GraphExplorer and builds a
structured evidence package for each by following the SOP steps.

Enforces the multi-modal evidence requirement: to pass a candidate forward,
at least syscalls + one other modality must confirm. This prevents syscall-only
false positives.

Produces:
  evidence_dict    — {service: [evidence_strings]} for each candidate
  best_candidate   — service with strongest multi-modal evidence
  modalities_hit   — {"syscalls": True, "traces": bool, "memory": bool, ...}

Tools available: query_syscalls, query_traces, compare_trace_latency,
                 compute_anomaly_score, detect_error_burst, cross_correlate_services
"""
from __future__ import annotations

import re
from typing import Any

from omnirca.agents.base_agent import SubAgent
from omnirca.agent.tool_schema import TOOL_SCHEMAS, _TOOL_MAP


_EVIDENCE_TOOL_NAMES = {
    "query_syscalls",
    "query_traces",
    "compare_trace_latency",
    "compute_anomaly_score",
    "detect_error_burst",
    "cross_correlate_services",
}
_SCHEMAS       = [s for s in TOOL_SCHEMAS if s["function"]["name"] in _EVIDENCE_TOOL_NAMES]
_TOOL_MAP_LOCAL = {k: v for k, v in _TOOL_MAP.items() if k in _EVIDENCE_TOOL_NAMES}


_SYSTEM_PROMPT = """\
You are EvidenceCollector, a specialist SRE agent for MULTI-MODAL EVIDENCE GATHERING.

You receive up to 2 root-cause candidate services from GraphExplorer.
Your job is to build an evidence package for each candidate by collecting data
from MULTIPLE telemetry sources (not just syscalls).

EVIDENCE GATHERING RULES:
1. For EACH candidate (up to 2), call compute_anomaly_score to get composite data.
2. For EACH candidate, call query_syscalls to confirm the syscall anomaly is real.
3. If traces are available (only ~10 of 20 services have traces), call
   compare_trace_latency or query_traces for latency evidence.
4. If error bursts are expected (based on fault type), call detect_error_burst.
5. Do NOT call query_logs expecting ERROR entries — this dataset has NONE.

MULTI-MODAL REQUIREMENT:
A candidate must have BOTH:
  ✓ Syscall anomaly (required — the primary signal)
  ✓ At least ONE of: trace latency spike | memory slope | anomaly score > 200

OUTPUT FORMAT (after tool calls):
CANDIDATE_1: service_name
  EVIDENCE: [item1, item2, item3]
  MODALITIES: syscalls=yes, traces=yes/no, memory=yes/no, errors=yes/no
  STRONG: YES|NO

CANDIDATE_2: service_name  (if applicable)
  EVIDENCE: [item1, item2]
  MODALITIES: syscalls=yes, traces=yes/no, memory=yes/no, errors=yes/no
  STRONG: YES|NO

BEST_CANDIDATE: service_name
ANALYSIS_COMPLETE
"""


class EvidenceCollector(SubAgent):
    AGENT_NAME    = "EvidenceCollector"
    SYSTEM_PROMPT = _SYSTEM_PROMPT
    TOOL_SCHEMAS  = _SCHEMAS
    TOOL_MAP      = _TOOL_MAP_LOCAL
    MAX_STEPS     = 6

    def _build_user_message(self, context: dict) -> str:
        candidates = context.get("root_candidates", [])
        top        = context.get("top_candidate", context.get("top_service", "unknown"))
        fault_cat  = context.get("fault_category", "unknown")

        # Pull top 2 candidates safely
        if candidates:
            c1 = candidates[0][0] if isinstance(candidates[0], tuple) else candidates[0]
            c2 = (candidates[1][0] if len(candidates) > 1
                  and isinstance(candidates[1], tuple) else
                  candidates[1] if len(candidates) > 1 else None)
        else:
            c1, c2 = top, None

        candidate_str = c1
        if c2 and c2 != c1:
            candidate_str += f", {c2}"

        return (
            f"Fault window: {context['t_start']} → {context['t_end']}\n\n"
            f"Root-cause candidates from GraphExplorer: {candidate_str}\n"
            f"Suspected fault category: {fault_cat}\n\n"
            f"Collect multi-modal evidence for each candidate.\n"
            f"Confirm that at least syscalls + one other modality supports each candidate."
        )

    def _parse_findings(
        self,
        messages: list[dict],
        tool_calls: list[dict],
        final_text: str,
        context: dict,
    ) -> dict[str, Any]:
        findings: dict[str, Any] = {
            "evidence_dict":  {},
            "best_candidate": context.get("top_candidate", context.get("top_service")),
            "modalities_hit": {"syscalls": True},
            "raw_summary":    final_text,
        }

        # Parse CANDIDATE_1 / CANDIDATE_2 blocks
        for block_re, cand_key in [
            (r"CANDIDATE_1:\s*(\S+)(.*?)(?=CANDIDATE_2:|BEST_CANDIDATE:|ANALYSIS_COMPLETE|$)",
             "candidate_1"),
            (r"CANDIDATE_2:\s*(\S+)(.*?)(?=BEST_CANDIDATE:|ANALYSIS_COMPLETE|$)",
             "candidate_2"),
        ]:
            m = re.search(block_re, final_text, re.DOTALL)
            if m:
                svc   = m.group(1).strip()
                block = m.group(2)
                ev_m  = re.search(r"EVIDENCE:\s*\[([^\]]+)\]", block)
                evs   = []
                if ev_m:
                    evs = [e.strip().strip("'\"") for e in ev_m.group(1).split(",") if e.strip()]
                findings["evidence_dict"][svc] = evs

                # Extract modalities
                mod_m = re.search(
                    r"MODALITIES:\s*(.*?)(?:\n|STRONG:|$)", block)
                if mod_m and cand_key == "candidate_1":
                    mod_str = mod_m.group(1)
                    for mod in ["syscalls", "traces", "memory", "errors"]:
                        yes_m = re.search(rf"{mod}=(yes|no)", mod_str, re.I)
                        if yes_m:
                            findings["modalities_hit"][mod] = yes_m.group(1).lower() == "yes"

        m = re.search(r"BEST_CANDIDATE:\s*(\S+)", final_text)
        if m:
            findings["best_candidate"] = m.group(1).strip()

        # Build flat evidence list for the best candidate (used by JudgeAgent)
        best = findings["best_candidate"]
        if best and best in findings["evidence_dict"]:
            findings["evidence_list"] = findings["evidence_dict"][best]
        else:
            # Collect all evidence if best not explicitly parsed
            all_evs = []
            for evs in findings["evidence_dict"].values():
                all_evs.extend(evs)
            findings["evidence_list"] = all_evs

        return findings
