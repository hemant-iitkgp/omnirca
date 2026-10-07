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

CAUSAL DIRECTION — THE MOST IMPORTANT RULE:
• Edge direction: A → B means "A calls B". Successors of B = the services B
  DEPENDS ON (callees, downstream). Predecessors of B = its CALLERS (upstream).
• A fault inside service X makes X slow or broken. Everything that CALLS X then
  waits on it and also looks degraded. So faults propagate UPSTREAM, from the
  broken service toward its callers.
• THEREFORE: callers of the faulty service are VICTIMS. The root cause is the
  anomalous service that does NOT itself depend on any other anomalous service
  — i.e. the deepest anomalous service, with no anomalous CALLEE beneath it.
• A service with FEW OR NO CALLERS is the system's entry point (frontend,
  gateway, load balancer). It sees every downstream problem and is therefore the
  most common VICTIM of all. Never conclude the entry point is the root cause
  merely because nothing upstream of it is anomalous — that is true by
  definition for an entry point and carries no evidence.
• Passive sinks that everything calls (logging/telemetry collectors) are never
  root causes.
• A service that is anomalous while all of its callees are healthy originated
  the fault. That is the strongest structural signal available.

INVESTIGATION STEPS:
1. Call get_service_dependencies on EACH of the top 3 anomalous services from DD.
   Look at each candidate's CALLEES: are any of them also in DD's anomalous list?
   • If a candidate depends on another anomalous service, the candidate is
     probably a victim of it — move down to that dependency.
   • If a candidate's callees are all healthy, the candidate is the root.
2. Call get_propagation_candidates on the top DD-ranked services to see the
   anomalous callers. Many anomalous callers = the service is being depended on
   and is a likely ROOT; a service with no callers at all is the entry point and
   is a likely VICTIM, not a root.
   • The root is the service that:
     a) Is anomalous, AND has no anomalous service among its own dependencies, OR
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
            # Strip markdown emphasis; if nothing sane is left, keep DD's pick
            # rather than letting a token like "**" propagate into the verdict.
            cand = m.group(1).strip().strip("`").strip("*").strip()
            if cand:
                findings["top_candidate"] = cand

        m = re.search(r"TOPOLOGY_NOTES:\s*(.+?)(?:\n|ANALYSIS_COMPLETE|$)", final_text, re.DOTALL)
        if m:
            findings["topology_notes"] = m.group(1).strip()

        # Fallback: use propagation_candidates tool result from trace
        if not findings["top_candidate"] and tool_calls:
            for tc in tool_calls:
                if tc["tool"] == "get_propagation_candidates":
                    result_str = tc.get("result", "")
                    # Pull first mentioned service from result
                    from omnirca.data_layer.loader import get_loader
                    known = sorted(get_loader().service_names, key=len, reverse=True)
                    svc_m = re.search(
                        r"\b(" + "|".join(re.escape(s) for s in known) + r")\b",
                        result_str)
                    if svc_m:
                        findings["top_candidate"] = svc_m.group(1)
                        break

        return findings
