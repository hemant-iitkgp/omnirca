"""
omnirca/agents/temporal_analyst.py — TemporalAnalyst sub-agent.

Role: Causal Ordering Analysis.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Answers the causal direction question:
  "Which suspect service's anomaly STARTED FIRST?"

In multi-service incidents, the service whose anomaly onset was earliest is most
likely the root cause. All downstream degradation follows effect → cause reasoning.

This is a research contribution (temporal_bonus) not found in reference papers.

Produces:
  onset_order      — {service: onset_timestamp}
  earliest_service — the service with the earliest anomaly onset
  causal_lead_min  — how many minutes earlier the earliest anomaly appeared

Tools available: detect_causal_order, cross_correlate_services
"""
from __future__ import annotations

import re
from typing import Any

from omnirca.agents.base_agent import SubAgent
from omnirca.agent.tool_schema import TOOL_SCHEMAS, _TOOL_MAP


_TEMPORAL_TOOL_NAMES = {"detect_causal_order", "cross_correlate_services"}
_SCHEMAS             = [s for s in TOOL_SCHEMAS if s["function"]["name"] in _TEMPORAL_TOOL_NAMES]
_TOOL_MAP_LOCAL      = {k: v for k, v in _TOOL_MAP.items() if k in _TEMPORAL_TOOL_NAMES}


_SYSTEM_PROMPT = """\
You are TemporalAnalyst, a specialist SRE agent for CAUSAL ORDERING.

Your job is to determine WHICH service's anomaly started FIRST.
The service that shows degradation earliest in the timeline is the most likely
root cause — all others became anomalous as cascading victims.

INVESTIGATION STEPS:
1. Call detect_causal_order with the list of ALL anomalous services found by
   DataDetective (and any other suspects). Include all of them.
2. If the ordering is unclear (multiple services onset at the same minute),
   call cross_correlate_services to check if one service's signal leads the other.
3. Note: "logging" has no causal significance — exclude it.

CAUSAL PRINCIPLE:
• Root cause anomaly onset PRECEDES all downstream anomalies.
• If service A onset is ≥ 1 minute before service B, and A → B in the
  dependency graph, then A is strongly supported as root cause.
• Equal onset = can't distinguish purely from time; set TEMPORAL_VERDICT=AMBIGUOUS.

TIEBREAKER RULE (when AMBIGUOUS — multiple services share the same onset minute):
  DO NOT use alphabetical order as a tiebreaker. Instead, reason structurally:
  • A service that is CALLED BY other anomalous services is a DOWNSTREAM VICTIM.
    It absorbs errors from its callers — its anomaly is effect, not cause.
  • A service with NO anomalous upstream callers is a CANDIDATE ROOT CAUSE.
    Nothing upstream triggered it; it originated the fault.
  When onset is tied, report EARLIEST_SERVICE as the service that has no
  anomalous upstream callers in the affected set, if such a service exists.
  If all services have anomalous callers or none do, report AMBIGUOUS and
  leave the decision to GraphExplorer's topology analysis.

OUTPUT FORMAT (after tool calls):
ONSET_ORDER: [service1:timestamp, service2:timestamp, ...]   ← sorted earliest first
EARLIEST_SERVICE: service_name
CAUSAL_LEAD_MIN: float   ← minutes by which earliest leads second-earliest
TEMPORAL_VERDICT: STRONG_EVIDENCE|WEAK_EVIDENCE|AMBIGUOUS
ANALYSIS_COMPLETE
"""


class TemporalAnalyst(SubAgent):
    AGENT_NAME    = "TemporalAnalyst"
    SYSTEM_PROMPT = _SYSTEM_PROMPT
    TOOL_SCHEMAS  = _SCHEMAS
    TOOL_MAP      = _TOOL_MAP_LOCAL
    MAX_STEPS     = 4

    def _build_user_message(self, context: dict) -> str:
        ranked   = context.get("ranked_services", [])
        top_cand = context.get("top_candidate", context.get("top_service", "unknown"))

        # Build a merged service list from DataDetective + GraphExplorer findings
        all_services = list(ranked)
        if top_cand and top_cand not in all_services:
            all_services.insert(0, top_cand)
        # Remove logging — never meaningful for temporal analysis
        all_services = [s for s in all_services if s != "logging"][:6]

        return (
            f"Fault window: {context['t_start']} → {context['t_end']}\n\n"
            f"Anomalous services to analyze for temporal order:\n"
            f"  {all_services}\n\n"
            f"Call detect_causal_order with this list to find which service\n"
            f"showed anomaly onset EARLIEST. That service is the causal root."
        )

    def _parse_findings(
        self,
        messages: list[dict],
        tool_calls: list[dict],
        final_text: str,
        context: dict,
    ) -> dict[str, Any]:
        findings: dict[str, Any] = {
            "onset_order":       {},
            "earliest_service":  context.get("top_candidate", context.get("top_service")),
            "causal_lead_min":   0.0,
            "temporal_verdict":  "AMBIGUOUS",
            "raw_summary":       final_text,
        }

        m = re.search(r"ONSET_ORDER:\s*\[([^\]]+)\]", final_text)
        if m:
            pairs = m.group(1).split(",")
            for p in pairs:
                p = p.strip().strip("'\"")
                if ":" in p:
                    svc, ts = p.split(":", 1)
                    findings["onset_order"][svc.strip()] = ts.strip()

        m = re.search(r"EARLIEST_SERVICE:\s*(\S+)", final_text)
        if m:
            findings["earliest_service"] = m.group(1).strip()

        m = re.search(r"CAUSAL_LEAD_MIN:\s*([\d.]+)", final_text)
        if m:
            findings["causal_lead_min"] = float(m.group(1))

        m = re.search(r"TEMPORAL_VERDICT:\s*(STRONG_EVIDENCE|WEAK_EVIDENCE|AMBIGUOUS)", final_text)
        if m:
            findings["temporal_verdict"] = m.group(1)

        # Fallback: extract from detect_causal_order tool result
        if not findings["onset_order"] and tool_calls:
            for tc in tool_calls:
                if tc["tool"] == "detect_causal_order":
                    result_str = tc.get("result", "")
                    from omnirca.data_layer.loader import get_loader
                    known = sorted(get_loader().service_names, key=len, reverse=True)
                    svc_m = re.search(
                        r"\b(" + "|".join(re.escape(s) for s in known) + r")\b",
                        result_str,
                    )
                    if svc_m:
                        findings["earliest_service"] = svc_m.group(1)
                    break

        return findings
