"""
omnirca/agents/main_agent.py — MainAgent: the orchestrator.

Role: Orchestrate the 5 sub-agents, aggregate findings, and call finalize_rca.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Workflow (from plan.md Phase 5.2):
  1. DataDetective   — syscall triage of all 20 services
  2. GraphExplorer   — dependency/propagation upstream search
  3. FaultTyper      — classify fault type + retrieve SOP
  4. EvidenceCollector — multi-modal corroboration for top 2 candidates
  5. TemporalAnalyst  — causal onset ordering
  6. JudgeAgent       — synthesize → final verdict → explain_evidence

Then calls finalize_rca with the JudgeAgent's verdict.

Returns a MultiAgentResult dataclass with:
  root_cause, fault_category, confidence,
  evidence_list, propagation_path,
  agent_reports (all 6 sub-agent outputs),
  duration_s
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from omnirca.agents.base_agent import SubAgentReport
from omnirca.agents.data_detective import DataDetective
from omnirca.agents.graph_explorer import GraphExplorer
from omnirca.agents.fault_typer import FaultTyper
from omnirca.agents.evidence_collector import EvidenceCollector
from omnirca.agents.temporal_analyst import TemporalAnalyst
from omnirca.agents.judge_agent import JudgeAgent
from omnirca.agent.tool_schema import dispatch_tool
from omnirca.logger import get_logger

_log = get_logger("omnirca.main_agent")


@dataclass
class MultiAgentResult:
    """Structured output from the full multi-agent pipeline."""
    root_cause:      str
    fault_category:  str
    confidence:      str                           # HIGH | MEDIUM | LOW
    evidence_list:   list[str]
    propagation_path: list[str]
    composite_score: float

    # Per-agent diagnostics
    agent_reports:   dict[str, SubAgentReport] = field(default_factory=dict)
    judge_justification: str = ""
    duration_s:      float = 0.0
    needs_more_evidence: bool = False

    def summary(self) -> str:
        return (
            f"Root cause: {self.root_cause}  |  "
            f"Category: {self.fault_category}  |  "
            f"Confidence: {self.confidence}  |  "
            f"Duration: {self.duration_s:.1f}s"
        )

    def __repr__(self) -> str:
        return (
            f"<MultiAgentResult root_cause={self.root_cause!r} "
            f"confidence={self.confidence!r}>"
        )


class MainAgent:
    """
    Orchestrates all 5 specialist sub-agents and calls finalize_rca.

    Parameters
    ----------
    verbose : bool
        If True, print progress for each sub-agent as it runs.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def run(
        self,
        incident_query: str,
        t_start:        str,
        t_end:          str,
    ) -> MultiAgentResult:
        """
        Run the full multi-agent investigation pipeline.

        Parameters
        ----------
        incident_query : str
            Free-text symptom description (does NOT contain fault type or service name).
        t_start, t_end : str
            Fault window boundaries ("2024-01-01 HH:MM:SS").

        Returns
        -------
        MultiAgentResult with the final verdict and all sub-agent reports.
        """
        run_start = time.monotonic()

        _log.info("=== MainAgent started | window %s → %s ===", t_start, t_end)

        # Shared context dict — each agent adds its findings here
        ctx: dict[str, Any] = {
            "t_start":        t_start,
            "t_end":          t_end,
            "incident_query": incident_query,
        }

        agent_reports: dict[str, SubAgentReport] = {}

        # ── 1. DataDetective ────────────────────────────────────────────────
        self._log("DataDetective", "scanning all 20 services for syscall anomalies")
        try:
            det_report = DataDetective(verbose=self.verbose).run(ctx)
            agent_reports["DataDetective"] = det_report
            self._merge(ctx, det_report.findings)
        except Exception as e:
            _log.error("DataDetective FAILED: %s — using empty anomaly list", e)
            ctx.setdefault("ranked_services", [])
            ctx.setdefault("top_service", "unknown")

        # ── 2. GraphExplorer ─────────────────────────────────────────────────
        self._log("GraphExplorer", f"tracing upstream from {ctx.get('top_service')}")
        try:
            graph_report = GraphExplorer(verbose=self.verbose).run(ctx)
            agent_reports["GraphExplorer"] = graph_report
            self._merge(ctx, graph_report.findings)
        except Exception as e:
            _log.error("GraphExplorer FAILED: %s — using DD top as candidate", e)
            ctx.setdefault("top_candidate", ctx.get("top_service", "unknown"))

        # ── 3. FaultTyper ────────────────────────────────────────────────────
        self._log("FaultTyper", "classifying fault type")
        try:
            fault_report = FaultTyper(verbose=self.verbose).run(ctx)
            agent_reports["FaultTyper"] = fault_report
            self._merge(ctx, fault_report.findings)
        except Exception as e:
            _log.error("FaultTyper FAILED: %s — using unknown category", e)
            ctx.setdefault("fault_category", "unknown")

        # ── 4. EvidenceCollector ─────────────────────────────────────────────
        self._log("EvidenceCollector", f"gathering multi-modal evidence for candidates")
        try:
            evidence_report = EvidenceCollector(verbose=self.verbose).run(ctx)
            agent_reports["EvidenceCollector"] = evidence_report
            self._merge(ctx, evidence_report.findings)
        except Exception as e:
            _log.error("EvidenceCollector FAILED: %s — proceeding without extra evidence", e)

        # ── 5. TemporalAnalyst ───────────────────────────────────────────────
        self._log("TemporalAnalyst", "establishing causal onset order")
        try:
            temporal_report = TemporalAnalyst(verbose=self.verbose).run(ctx)
            agent_reports["TemporalAnalyst"] = temporal_report
            self._merge(ctx, temporal_report.findings)
        except Exception as e:
            _log.error("TemporalAnalyst FAILED: %s — proceeding without temporal data", e)

        # ── 6. JudgeAgent ────────────────────────────────────────────────────
        self._log("JudgeAgent", "synthesizing all findings → final verdict")
        try:
            judge_report = JudgeAgent(verbose=self.verbose).run(ctx)
            agent_reports["JudgeAgent"] = judge_report
            self._merge(ctx, judge_report.findings)
        except Exception as e:
            _log.error("JudgeAgent FAILED: %s — using best available root cause", e)
            ctx.setdefault("final_root_cause", ctx.get("top_candidate", ctx.get("top_service", "unknown")))

        # ── Extract final verdict from JudgeAgent findings ───────────────────
        root_cause      = ctx.get("final_root_cause",     ctx.get("top_service", "unknown"))
        fault_category  = ctx.get("final_fault_category", ctx.get("fault_category", "unknown"))
        confidence      = ctx.get("final_confidence",     "MEDIUM")
        evidence_list   = ctx.get("evidence_list",        [])
        propagation     = ctx.get("propagation_path",     [root_cause])
        composite_score = float(ctx.get("top_score",      0.0))
        needs_more      = ctx.get("needs_more_evidence",  False)
        justification   = ctx.get("justification",        "")

        # Ensure propagation_path is a proper list, not a string
        if isinstance(propagation, str):
            propagation = [propagation]

        # ── Call finalize_rca ─────────────────────────────────────────────────
        self._log("MainAgent", f"calling finalize_rca → root_cause={root_cause}")
        _log.info(
            "=== finalize_rca | root=%s | category=%s | confidence=%s ===",
            root_cause, fault_category, confidence,
        )
        finalize_result = dispatch_tool("finalize_rca", {
            "root_cause":       root_cause,
            "confidence":       confidence,
            "evidence_list":    evidence_list if evidence_list else [f"Syscall anomaly on {root_cause}"],
            "fault_category":   fault_category,
            "propagation_path": propagation,
            "affected_services": list(ctx.get("ranked_services", [root_cause])),
            "composite_score":  composite_score,
            "notes":            justification[:500] if justification else "",
        })

        total_s = time.monotonic() - run_start
        _log.info("=== MainAgent done | total=%.1fs | root=%s | category=%s | confidence=%s ===",
                  total_s, root_cause, fault_category, confidence)

        return MultiAgentResult(
            root_cause           = root_cause,
            fault_category       = fault_category,
            confidence           = confidence,
            evidence_list        = evidence_list,
            propagation_path     = propagation,
            composite_score      = composite_score,
            agent_reports        = agent_reports,
            judge_justification  = justification,
            duration_s           = round(total_s, 2),
            needs_more_evidence  = needs_more,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, agent: str, msg: str) -> None:
        if self.verbose:
            print(f"\n{'─' * 60}")
            print(f"  [{agent}] {msg}")
            print(f"{'─' * 60}")

    @staticmethod
    def _merge(ctx: dict, findings: dict) -> None:
        """
        Merge sub-agent findings into the shared context.
        Only add keys that are NOT already set (or that are None/empty),
        so later agents can override earlier ones where the key is the same.
        """
        for k, v in findings.items():
            if k == "raw_summary":
                continue  # never pollute context with raw text
            if v is None or v == "" or v == [] or v == {}:
                continue
            # Keys that should always be overwritten by later agents
            _OVERWRITE_KEYS = {
                "top_candidate",
                "best_candidate",
                "final_root_cause",
                "final_fault_category",
                "final_confidence",
                "evidence_list",
                "justification",
                "needs_more_evidence",
            }
            if k in _OVERWRITE_KEYS or k not in ctx or not ctx[k]:
                ctx[k] = v
