"""
omnirca/evaluation/test_harness.py — Blind evaluation harness (Phase 7).

Fairness guarantees implemented here:
  1. faults.csv is loaded ONCE at harness init — ground truth is NEVER
     passed to the agent.  Only the query_text (time window + one affected
     service + symptom) is passed.
  2. The agent's prediction is compared to ground truth ONLY AFTER the run
     completes.
  3. ranked_candidates are extracted from sub-agent reports (DataDetective
     ranked_services), not from ground truth.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Optional

from omnirca.evaluation.query_builder import QueryBuilder, FaultQuery
from omnirca.evaluation.metrics import EvaluationResult, MetricsReport, compute_metrics

# Default path to faults.csv (ground truth — harness only, never agent)
_DEFAULT_FAULTS_CSV = str(
    Path(__file__).parent.parent.parent / "datasets_complex" / "faults.csv"
)


class EvaluationHarness:
    """
    Orchestrates blind evaluation of the full agent system over all 10 faults.

    Usage:
        harness = EvaluationHarness()
        results = harness.run_all(verbose=True)
        report  = compute_metrics(results)
        print(report.summary())
        print(report.per_fault_table())
    """

    def __init__(
        self,
        faults_csv: str = _DEFAULT_FAULTS_CSV,
        config: str = "config_g",
    ) -> None:
        self.query_builder = QueryBuilder(faults_csv)
        self.config = config
        self._agent = None   # lazy init

    def _get_agent(self):
        """Lazy-init agent so import cost is paid only on first run."""
        if self._agent is None:
            from omnirca.agents.main_agent import MainAgent
            self._agent = MainAgent(verbose=False)
        return self._agent

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run_single(
        self,
        fault_id: int,
        verbose: bool = True,
    ) -> EvaluationResult:
        """
        Run the agent on a single fault and return an EvaluationResult.

        Parameters
        ----------
        fault_id : int    0–9
        verbose  : bool   Print progress and result.
        """
        fq: FaultQuery = self.query_builder.build(fault_id)

        if verbose:
            print(f"\n{'═'*70}")
            print(f"  FAULT {fault_id}: {fq.fault_type}  "
                  f"[ground truth: {fq.ground_truth_service} | "
                  f"{fq.ground_truth_category}]")
            print(f"{'═'*70}")
            print(f"  Window:             {fq.t_start} → {fq.t_end}")
            print(f"  Reported service:   {fq.reported_service}")
            print(f"  Symptom:            {fq.symptom[:90]}…")
            print(f"{'─'*70}")
            print("  Running agent…")

        t0 = time.monotonic()
        error: Optional[str] = None

        try:
            agent = self._get_agent()
            result = agent.run(
                incident_query = fq.query_text,
                t_start        = fq.t_start,
                t_end          = fq.t_end,
            )
            duration = time.monotonic() - t0

            # Extract ranked candidates from DataDetective findings
            ranked = _extract_ranked_candidates(result)

            pred_root = result.root_cause or "unknown"
            pred_cat  = result.fault_category or "unknown"

            # ── Compute correctness flags inline for immediate display ───────
            from omnirca.evaluation.metrics import _norm_cat
            t1_ok   = pred_root.lower().strip() == fq.ground_truth_service.lower().strip()
            cands_l = [c.lower().strip() for c in ranked]
            true_l  = fq.ground_truth_service.lower().strip()
            if true_l in cands_l:
                _rank = cands_l.index(true_l) + 1
            elif t1_ok:
                _rank = 1
            else:
                _rank = len(cands_l) + 2   # not in top-3
            cat_ok  = _norm_cat(pred_cat) == _norm_cat(fq.ground_truth_category)

            ev = EvaluationResult(
                fault_id             = fault_id,
                fault_type           = fq.fault_type,          # ground truth type label
                t_start              = fq.t_start,
                t_end                = fq.t_end,
                reported_service     = fq.reported_service,
                true_root_cause      = fq.ground_truth_service,
                true_category        = fq.ground_truth_category,
                true_category_group  = fq.ground_truth_category_group,
                predicted_root_cause = pred_root,
                predicted_category   = pred_cat,
                predicted_confidence = result.confidence or "LOW",
                ranked_candidates    = ranked,
                top1_correct         = t1_ok,
                top3_correct         = _rank <= 3,
                category_correct     = cat_ok,
                rank                 = _rank,
                duration_s           = duration,
                agent_steps          = sum(
                    r.steps_taken
                    for r in result.agent_reports.values()
                ),
                config               = self.config,
            )

        except Exception as exc:
            duration = time.monotonic() - t0
            error    = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            ev = EvaluationResult(
                fault_id             = fault_id,
                fault_type           = fq.fault_type,
                t_start              = fq.t_start,
                t_end                = fq.t_end,
                reported_service     = fq.reported_service,
                true_root_cause      = fq.ground_truth_service,
                true_category        = fq.ground_truth_category,
                true_category_group  = fq.ground_truth_category_group,
                predicted_root_cause = "ERROR",
                predicted_category   = "unknown",
                predicted_confidence = "LOW",
                ranked_candidates    = [],
                duration_s           = duration,
                agent_steps          = 0,
                config               = self.config,
                error                = error,
            )

        if verbose:
            _print_result(ev)

        return ev

    def run_all(
        self,
        fault_ids: Optional[list[int]] = None,
        verbose: bool = True,
    ) -> list[EvaluationResult]:
        """
        Run the agent on all (or selected) faults.

        Parameters
        ----------
        fault_ids : list[int] | None   Subset of fault IDs (0–9). None = all.
        verbose   : bool               Print per-fault progress.

        Returns
        -------
        List of EvaluationResult (one per fault, same order as fault_ids).
        """
        ids = fault_ids if fault_ids is not None else list(range(10))
        results: list[EvaluationResult] = []

        if verbose:
            print(f"\n{'█'*70}")
            print(f"  OmniRCA BLIND EVALUATION — {len(ids)} fault(s)")
            print(f"  Config: {self.config}")
            print(f"{'█'*70}")

        for fid in ids:
            ev = self.run_single(fid, verbose=verbose)
            results.append(ev)

        if verbose:
            report = compute_metrics(results, config=self.config)
            print(report.per_fault_table())
            print(report.summary())

        return results

    def run_all_and_report(
        self,
        fault_ids: Optional[list[int]] = None,
    ) -> MetricsReport:
        """Convenience: run and return the MetricsReport directly."""
        results = self.run_all(fault_ids=fault_ids, verbose=True)
        return compute_metrics(results, config=self.config)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_ranked_candidates(result: Any) -> list[str]:
    """
    Extract the ranked service candidates from a MultiAgentResult.

    Preference order:
      1. DataDetective.findings["ranked_services"]  (syscall z-score ranked list)
      2. GraphExplorer.findings["root_candidates"]  ((svc, score) tuples)
      3. [result.root_cause]                        (only the final answer)
    """
    reports = getattr(result, "agent_reports", {})

    # Try DataDetective ranked_services
    dd = reports.get("DataDetective")
    if dd:
        ranked = dd.findings.get("ranked_services", [])
        if ranked and isinstance(ranked, list) and len(ranked) > 0:
            # May be plain strings or (name, score) tuples
            cleaned = []
            for item in ranked:
                if isinstance(item, str):
                    cleaned.append(item)
                elif isinstance(item, (list, tuple)) and len(item) >= 1:
                    cleaned.append(str(item[0]))
            if cleaned:
                # Put predicted root cause first if not already first
                root = getattr(result, "root_cause", None)
                if root and root in cleaned and cleaned[0] != root:
                    cleaned.remove(root)
                    cleaned.insert(0, root)
                elif root and root not in cleaned:
                    cleaned.insert(0, root)
                return cleaned

    # Try GraphExplorer root_candidates
    ge = reports.get("GraphExplorer")
    if ge:
        cands = ge.findings.get("root_candidates", [])
        if cands:
            cleaned = []
            for item in cands:
                if isinstance(item, str):
                    cleaned.append(item)
                elif isinstance(item, (list, tuple)) and len(item) >= 1:
                    cleaned.append(str(item[0]))
            if cleaned:
                return cleaned

    # Fallback: just the final answer
    root = getattr(result, "root_cause", "unknown") or "unknown"
    return [root]


def _print_result(ev: EvaluationResult) -> None:
    """Print a formatted result block for one fault."""
    t1  = "✓  CORRECT" if ev.top1_correct else "✗  WRONG"
    cat = "✓" if ev.category_correct else "✗"
    print(f"  Predicted root cause: {ev.predicted_root_cause:<14}  "
          f"[true: {ev.true_root_cause}]  {t1}")
    print(f"  Predicted category:   {ev.predicted_category:<22}  "
          f"[true: {ev.true_category}]  {cat}")
    print(f"  Confidence:           {ev.predicted_confidence}")
    print(f"  Rank of true answer:  #{ev.rank}  "
          f"(top-3: {'✓' if ev.top3_correct else '✗'})")
    print(f"  Duration:             {ev.duration_s:.1f}s  |  "
          f"Agent steps: {ev.agent_steps}")
    if ev.error:
        print(f"  ERROR: {ev.error[:200]}")
    print(f"{'─'*70}")
