"""
omnirca/agents/judge_agent.py — JudgeAgent sub-agent.

Role: Evidence Quality Assessment and Confidence Scoring.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The JudgeAgent receives all findings from all other sub-agents and evaluates:
  1. Is the evidence INTERNALLY CONSISTENT across modalities?
  2. Does the temporal ordering ALIGN with the graph structure?
  3. Does the signal pattern FIT the claimed fault category?
  4. Is there a CREDIBLE ALTERNATIVE that should be considered?

Produces:
  final_root_cause    — the agreed root cause service
  final_fault_category — the agreed fault type
  final_confidence     — HIGH | MEDIUM | LOW
  needs_more_evidence  — bool: if True, MainAgent should run another sub-agent pass
  justification        — LLM-written natural language verdict
  evidence_list        — consolidated evidence for finalize_rca

Tools available: explain_evidence
  (one tool only — JudgeAgent reasons from context, not raw data queries)
"""
from __future__ import annotations

import json
import re
from typing import Any

from omnirca.agents.base_agent import SubAgent
from omnirca.agent.tool_schema import TOOL_SCHEMAS, _TOOL_MAP


_JUDGE_TOOL_NAMES = {"explain_evidence"}
_SCHEMAS          = [s for s in TOOL_SCHEMAS if s["function"]["name"] in _JUDGE_TOOL_NAMES]
_TOOL_MAP_LOCAL   = {k: v for k, v in _TOOL_MAP.items() if k in _JUDGE_TOOL_NAMES}


_SYSTEM_PROMPT = """\
You are JudgeAgent, the final evaluator in a multi-agent RCA pipeline.

You receive a COMPLETE DOSSIER from 5 specialist agents:
  DataDetective   — identified the most anomalous services
  GraphExplorer   — traced upstream dependencies and propagation
  FaultTyper      — classified the fault category
  EvidenceCollector — gathered multi-modal corroboration
  TemporalAnalyst — established causal onset ordering

Your job is to SYNTHESIZE all findings and produce a final verdict.

EVALUATION CRITERIA:
1. CONSISTENCY: Do DataDetective, GraphExplorer, and TemporalAnalyst all point to
   the same service? Consistency across independent agents → HIGH confidence.
2. TEMPORAL ALIGNMENT: Does the earliest-onset service from TemporalAnalyst match
   GraphExplorer's top candidate?
   CRITICAL — when TEMPORAL_VERDICT is AMBIGUOUS (onset tied, causal_lead_min=0):
   • Temporal ordering provides NO information. Do NOT use TemporalAnalyst's
     EARLIEST_SERVICE field — it is structurally uninformative in this case.
   • Fall back entirely to GraphExplorer's topology: the service identified as
     having NO anomalous upstream callers is the structural root cause.
   • GraphExplorer's get_propagation_candidates result "no anomalous upstream
     candidates found" for a service = definitive structural root cause signal.
3. FAULT CATEGORY FIT: Does EvidenceCollector's evidence match FaultTyper's pattern?
4. MULTI-MODAL COVERAGE: Does EvidenceCollector show syscalls PLUS at least one
   other modality? If only syscalls confirmed → cap confidence at MEDIUM.
5. ALTERNATIVE HYPOTHESIS: Is there a plausible alternative root cause?

CONFIDENCE RULES:
• HIGH:   3+ agents agree, temporal order confirmed, multi-modal evidence strong
• MEDIUM: 2 agents agree, temporal unclear or only syscall evidence
• LOW:    Agents disagree or evidence is weak
• LOW:    TEMPORAL_VERDICT=AMBIGUOUS AND DataDetective top service differs from
          GraphExplorer top candidate — structural disagreement, no temporal proof

CONFLICT RESOLUTION — WHEN DD AND GE DISAGREE:
When DataDetective's top_service ≠ GraphExplorer's top_candidate, you MUST pick
one of those two. DO NOT select a third service that neither agent identified — that
is always wrong. The correct answer is always one of: DD's top_service OR GE's top_candidate.

RESOLUTION CRITERIA (apply in order):
1. Z-SCORE PRIORITY: DataDetective's ranking is based on z-score — the most
   objective signal. If DD's top_service has a HIGHER z-score than GE's candidate
   (check DD's ranked_services list), TRUST DD. The highest z-score service with
   no anomalous upstream callers is almost always the root cause.
2. TOPOLOGY TIEBREAKER: Trust GE's candidate ONLY when:
   a) GE's candidate has a CLEARLY higher propagation_score from multiple paths, AND
   b) DD's top_service is DOWNSTREAM of GE's candidate in the dependency graph
      (i.e., GE showed that its candidate CALLS DD's top service).
3. STRUCTURAL ROOT: A service with "no anomalous upstream candidates" from GE's
   propagation analysis is a structural root — it has no anomalous callers, so
   nothing upstream triggered it. This is STRONG evidence it originated the fault.
4. DEFAULT: When evidence is ambiguous, trust DD's top_service (highest z-score).
   DD does data-first analysis without topology bias.

• Do NOT assume databases/caches are always root causes over backends.
  Backends CAN be roots when they send degraded requests to downstream databases.
• If confidence is LOW due to disagreement, still pick one of DD or GE's candidates.
• NEVER output "**" or an empty/blank service name.

YOUR ACTION:
1. Reason through the 5 criteria above.
2. Call explain_evidence(service=final_root_cause, fault_category=..., evidence_list=[...])
   to generate the human-readable RCA summary.
3. After the tool call, output your structured verdict.

OUTPUT FORMAT:
FINAL_ROOT_CAUSE: service_name
FINAL_FAULT_CATEGORY: fault_type
FINAL_CONFIDENCE: HIGH|MEDIUM|LOW
NEEDS_MORE_EVIDENCE: YES|NO
JUSTIFICATION: [your reasoning across the 5 criteria]
ANALYSIS_COMPLETE
"""


class JudgeAgent(SubAgent):
    AGENT_NAME    = "JudgeAgent"
    SYSTEM_PROMPT = _SYSTEM_PROMPT
    TOOL_SCHEMAS  = _SCHEMAS
    TOOL_MAP      = _TOOL_MAP_LOCAL
    MAX_STEPS     = 4

    def _build_user_message(self, context: dict) -> str:
        # Assemble a compact dossier from all prior agent findings
        dossier = {
            "fault_window": {
                "t_start": context.get("t_start"),
                "t_end":   context.get("t_end"),
            },
            "DataDetective": {
                "ranked_services":   context.get("ranked_services", []),
                "top_service":       context.get("top_service"),
                "top_score":         context.get("top_score"),
                "sub_channel_hints": context.get("sub_channel_hints", ""),
                "memory_slope":      context.get("memory_slope"),
            },
            "GraphExplorer": {
                "top_candidate":    context.get("top_candidate"),
                "propagation_path": context.get("propagation_path", []),
                "topology_notes":   context.get("topology_notes", ""),
            },
            "FaultTyper": {
                "fault_category":   context.get("fault_category", "unknown"),
                "fault_confidence": context.get("fault_confidence", "LOW"),
                "pattern_notes":    context.get("pattern_notes", ""),
            },
            "EvidenceCollector": {
                "best_candidate":   context.get("best_candidate"),
                "modalities_hit":   context.get("modalities_hit", {}),
                "evidence_list":    context.get("evidence_list", []),
            },
            "TemporalAnalyst": {
                "earliest_service": context.get("earliest_service"),
                "causal_lead_min":  context.get("causal_lead_min", 0.0),
                "temporal_verdict": context.get("temporal_verdict", "AMBIGUOUS"),
            },
        }

        return (
            f"Complete investigation dossier:\n"
            f"{json.dumps(dossier, indent=2, default=str)}\n\n"
            f"Evaluate the evidence, call explain_evidence, and deliver your final verdict."
        )

    def _parse_findings(
        self,
        messages: list[dict],
        tool_calls: list[dict],
        final_text: str,
        context: dict,
    ) -> dict[str, Any]:
        # Fallback values from upstream agents
        fallback_service  = (
            context.get("best_candidate") or
            context.get("earliest_service") or
            context.get("top_candidate") or
            context.get("top_service", "unknown")
        )
        fallback_category = context.get("fault_category", "unknown")

        findings: dict[str, Any] = {
            "final_root_cause":     fallback_service,
            "final_fault_category": fallback_category,
            "final_confidence":     "MEDIUM",
            "needs_more_evidence":  False,
            "justification":        final_text,
            "evidence_list":        context.get("evidence_list", []),
            "explanation":          "",
            "raw_summary":          final_text,
        }

        m = re.search(r"FINAL_ROOT_CAUSE:\s*(\S+)", final_text)
        if m:
            # Strip markdown emphasis the model sometimes wraps the name in.
            # If nothing sane survives, keep the upstream fallback rather than
            # emitting a junk service name like "**".
            cand = m.group(1).strip().strip("`").strip("*").strip()
            if cand:
                findings["final_root_cause"] = cand

        m = re.search(r"FINAL_FAULT_CATEGORY:\s*(\S+)", final_text)
        if m:
            findings["final_fault_category"] = m.group(1).strip().strip("`")

        m = re.search(r"FINAL_CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", final_text)
        if m:
            findings["final_confidence"] = m.group(1).strip()

        m = re.search(r"NEEDS_MORE_EVIDENCE:\s*(YES|NO)", final_text)
        if m:
            findings["needs_more_evidence"] = m.group(1).strip() == "YES"

        m = re.search(r"JUSTIFICATION:\s*\[(.+?)\]", final_text, re.DOTALL)
        if m:
            findings["justification"] = m.group(1).strip()
        elif "JUSTIFICATION:" in final_text:
            # Non-bracket format
            just_m = re.search(r"JUSTIFICATION:\s*(.+?)(?:ANALYSIS_COMPLETE|$)",
                                final_text, re.DOTALL)
            if just_m:
                findings["justification"] = just_m.group(1).strip()

        # Extract explain_evidence output from tool call
        for tc in tool_calls:
            if tc["tool"] == "explain_evidence":
                findings["explanation"] = tc.get("result", "")
                break

        return findings
