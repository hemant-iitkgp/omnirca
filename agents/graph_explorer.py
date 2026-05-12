"""
omnirca/agents/graph_explorer.py — GraphExplorer sub-agent.

Role: Dependency & Propagation Analysis.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Takes DataDetective's ranked anomalous services and answers:
  "Which UPSTREAM service could have CAUSED the observed anomalies?"

Uses the fused topology (static architecture + dynamic trace data) to trace
propagation paths and score root-cause candidates.

Produces:
  root_candidates  — ordered list of (service, propagation_score)
  propagation_path — likely fault propagation chain
  dependency_notes — key topology observations

Tools available: get_service_dependencies, build_call_path, get_dynamic_graph,
                 get_propagation_candidates, get_fused_graph_summary,
                 detect_causal_order
"""
from __future__ import annotations

import re
from typing import Any

from omnirca.agents.base_agent import SubAgent
from omnirca.agent.tool_schema import TOOL_SCHEMAS, _TOOL_MAP


_GRAPH_TOOL_NAMES = {
    "get_service_dependencies",
    "build_call_path",
    "get_dynamic_graph",
    "get_propagation_candidates",
    "get_fused_graph_summary",
    "detect_causal_order",
}
_SCHEMAS       = [s for s in TOOL_SCHEMAS if s["function"]["name"] in _GRAPH_TOOL_NAMES]
_TOOL_MAP_LOCAL = {k: v for k, v in _TOOL_MAP.items() if k in _GRAPH_TOOL_NAMES}


_SYSTEM_PROMPT = """\
You are GraphExplorer, a specialist SRE agent for DEPENDENCY AND PROPAGATION ANALYSIS.

You receive a list of anomalous services from DataDetective. Your job is to trace
UPSTREAM through the dependency graph to find which service most likely CAUSED the
anomalies seen in others.

KEY TOPOLOGY RULES:
• auth_service is called by ALL services — its anomaly = everyone is affected.
  It is almost always a VICTIM, not a cause. Treat it last.
• logging is called by ALL services passively — ignore it entirely.
• cache_1 has ZERO incoming edges — it is isolated. If cache_1 is anomalous,
  it IS the root cause for that fault (nothing upstream can cause it).
• A typical call chain: frontend → api_gateway → backend_{N} → database/cache
• Edge direction: A → B means "A calls B". Predecessors of B = callers of B.

INVESTIGATION STEPS:
1. Call get_service_dependencies on EACH of the top 3 anomalous services from DD.
2. Call get_propagation_candidates on EACH of the top 3 DD-ranked services, not just #1.
   CRITICAL: DD's #1 service may be a downstream VICTIM, not the root.
   • If a service returns "no anomalous upstream candidates found", that service
     has NO anomalous callers — it is a STRONG root cause candidate.
   • The root is the service that:
     a) Has the HIGHEST z-score among services with no anomalous upstream callers, OR
     b) Has the earliest anomaly onset (from detect_causal_order).
3. Call detect_causal_order on all anomalous services to find the first-onset service.
4. Call build_call_path(root_candidate, victim_service) to confirm the causal chain.

ROOT CAUSE SELECTION RULE:
• DataDetective ranks services by z-score. The z-score is the PRIMARY signal.
• A service with "no anomalous upstream candidates" AND the highest z-score
  is the strongest root cause candidate — regardless of whether it is a
  backend, database, cache, or any other service type.
• Do NOT assume that database/cache services are more likely roots than backends.
  A backend CAN be the root cause if its anomaly propagates downstream to databases.
• Do NOT assume that databases are roots just because they appear in the anomalous list.
  Databases are often VICTIMS of degraded backends that send bad queries/requests.
• When multiple services have "no anomalous upstream callers", prefer the one
  with the HIGHEST z-score from DD's ranking.

OUTPUT FORMAT (after tool calls, write plain text):
ROOT_CANDIDATES: [service1:score, service2:score, ...]
PROPAGATION_PATH: [service1, service2, service3]
TOP_CANDIDATE: service_name
TOPOLOGY_NOTES: brief observation about graph structure relevant to this incident
ANALYSIS_COMPLETE
"""


class GraphExplorer(SubAgent):
    AGENT_NAME    = "GraphExplorer"
    SYSTEM_PROMPT = _SYSTEM_PROMPT
    TOOL_SCHEMAS  = _SCHEMAS
    TOOL_MAP      = _TOOL_MAP_LOCAL
    MAX_STEPS     = 8

    def _build_user_message(self, context: dict) -> str:
        ranked = context.get("ranked_services", [])
        top    = context.get("top_service", "unknown")
        # Top 3 candidates to investigate (GE must not assume #1 is the root)
        top3   = [svc for svc, _ in ranked[:3]] if ranked and isinstance(ranked[0], (list, tuple)) else ranked[:3]
        return (
            f"Fault window: {context['t_start']} → {context['t_end']}\n\n"
            f"DataDetective found these anomalous services (ranked by z-score):\n"
            f"  {ranked}\n"
            f"DD top service (highest z-score): {top}\n"
            f"Candidates to investigate: {top3}\n\n"
            f"IMPORTANT: Run get_propagation_candidates on EACH of the top candidates. "
            f"The service with the HIGHEST z-score and NO anomalous upstream callers "
            f"is the most likely root cause. Do NOT prefer databases over backends — "
            f"any service type can be the root. DD's z-score ranking is the primary signal."
        )

    def _parse_findings(
        self,
        messages: list[dict],
        tool_calls: list[dict],
        final_text: str,
        context: dict,
    ) -> dict[str, Any]:
        findings: dict[str, Any] = {
            "root_candidates":  [],
            "propagation_path": [],
            "top_candidate":    context.get("top_service"),
            "topology_notes":   "",
            "raw_summary":      final_text,
        }

        m = re.search(r"ROOT_CANDIDATES:\s*\[([^\]]+)\]", final_text)
        if m:
            raw_pairs = m.group(1).split(",")
            parsed = []
            for p in raw_pairs:
                p = p.strip().strip("'\"")
                if ":" in p:
                    svc, score = p.rsplit(":", 1)
                    try:
                        parsed.append((svc.strip(), float(score.strip())))
                    except ValueError:
                        parsed.append((p, 0.0))
                elif p:
                    parsed.append((p, 0.0))
            findings["root_candidates"] = parsed

        m = re.search(r"PROPAGATION_PATH:\s*\[([^\]]+)\]", final_text)
        if m:
            findings["propagation_path"] = [
                s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()
            ]

        m = re.search(r"TOP_CANDIDATE:\s*(\S+)", final_text)
        if m:
            findings["top_candidate"] = m.group(1).strip().strip("`")

        m = re.search(r"TOPOLOGY_NOTES:\s*(.+?)(?:\n|ANALYSIS_COMPLETE|$)", final_text, re.DOTALL)
        if m:
            findings["topology_notes"] = m.group(1).strip()

        # Fallback: use propagation_candidates tool result from trace
        if not findings["top_candidate"] and tool_calls:
            for tc in tool_calls:
                if tc["tool"] == "get_propagation_candidates":
                    result_str = tc.get("result", "")
                    # Pull first mentioned service from result
                    svc_m = re.search(
                        r"\b(backend_\d+|cache_\d+|database_\d+|frontend_\d+|auth_service)\b",
                        result_str)
                    if svc_m:
                        findings["top_candidate"] = svc_m.group(1)
                        break

        return findings
