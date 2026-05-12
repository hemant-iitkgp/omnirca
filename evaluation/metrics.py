"""
omnirca/evaluation/metrics.py — Evaluation metrics for Phase 7 (plan.md §7.2).

Metrics computed:
  Top-1 Accuracy    — exact correct root-cause service as first prediction
  Top-3 Accuracy    — correct service appearing in top-3 ranked candidates
  MRR               — Mean Reciprocal Rank of correct service in candidate list
  Category Accuracy — exact fault category match (normalized)
  Group Accuracy    — functional vs non-functional classification accuracy
  ECE               — Expected Calibration Error (confidence vs actual accuracy)
  Avg Duration      — average wall-clock time per investigation

References: plan.md §7.2 targets
  Top-1 > 60%, Top-3 > 85%, MRR > 0.65, Category Accuracy > 50%, ECE < 0.15
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
# EvaluationResult — single fault run
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Result from running the agent on one fault."""
    fault_id:               int
    fault_type:             str               # ground truth type (from CSV)
    t_start:                str
    t_end:                  str
    reported_service:       str               # the ONE service mentioned in query

    # Ground truth
    true_root_cause:        str
    true_category:          str               # normalised
    true_category_group:    str               # functional | non-functional

    # Agent prediction
    predicted_root_cause:   str
    predicted_category:     str               # normalised if possible
    predicted_confidence:   str               # HIGH | MEDIUM | LOW
    ranked_candidates:      list[str]         # ordered list of service candidates

    # Correctness flags
    top1_correct:           bool = False
    top3_correct:           bool = False
    category_correct:       bool = False
    group_correct:          bool = False
    rank:                   int  = 999        # rank of true_root_cause in ranked_candidates

    # Metadata
    duration_s:             float = 0.0
    agent_steps:            int   = 0
    config:                 str   = "config_g"
    error:                  Optional[str] = None

    # Confidence as float for ECE
    @property
    def confidence_float(self) -> float:
        return {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.4}.get(
            self.predicted_confidence.upper(), 0.5
        )

    def __repr__(self) -> str:
        mark = "✓" if self.top1_correct else "✗"
        return (
            f"<EvaluationResult fault={self.fault_id}:{self.fault_type} "
            f"predicted={self.predicted_root_cause!r} "
            f"true={self.true_root_cause!r} "
            f"top1={mark} cat={'✓' if self.category_correct else '✗'}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MetricsReport — aggregate over all faults
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricsReport:
    """Aggregate evaluation metrics over a set of EvaluationResult objects."""
    config:              str
    num_faults:          int

    top1_accuracy:       float      # fraction correct
    top3_accuracy:       float
    mrr:                 float      # mean reciprocal rank
    category_accuracy:   float      # exact category match
    group_accuracy:      float      # functional vs non-functional
    ece:                 float      # expected calibration error
    avg_duration_s:      float

    # Per-fault breakdown
    per_fault:           list[EvaluationResult] = field(default_factory=list)

    # Plan targets
    TOP1_TARGET:    float = 0.60
    TOP3_TARGET:    float = 0.85
    MRR_TARGET:     float = 0.65
    CAT_TARGET:     float = 0.50
    ECE_TARGET:     float = 0.15

    def summary(self) -> str:
        lines = [
            f"",
            f"{'='*62}",
            f"  EVALUATION METRICS — {self.config.upper()}  ({self.num_faults} faults)",
            f"{'='*62}",
            f"  Top-1 Accuracy:     {self.top1_accuracy:.0%}  "
                f"(target > {self.TOP1_TARGET:.0%})  "
                f"{'✓' if self.top1_accuracy >= self.TOP1_TARGET else '✗'}",
            f"  Top-3 Accuracy:     {self.top3_accuracy:.0%}  "
                f"(target > {self.TOP3_TARGET:.0%})  "
                f"{'✓' if self.top3_accuracy >= self.TOP3_TARGET else '✗'}",
            f"  MRR:                {self.mrr:.3f}  "
                f"(target > {self.MRR_TARGET:.2f})  "
                f"{'✓' if self.mrr >= self.MRR_TARGET else '✗'}",
            f"  Category Accuracy:  {self.category_accuracy:.0%}  "
                f"(target > {self.CAT_TARGET:.0%})  "
                f"{'✓' if self.category_accuracy >= self.CAT_TARGET else '✗'}",
            f"  Group Accuracy:     {self.group_accuracy:.0%}  (func vs non-func)",
            f"  ECE:                {self.ece:.3f}  "
                f"(target < {self.ECE_TARGET:.2f})  "
                f"{'✓' if self.ece < self.ECE_TARGET else '✗'}",
            f"  Avg Duration:       {self.avg_duration_s:.1f}s",
            f"{'='*62}",
        ]
        return "\n".join(lines)

    def per_fault_table(self) -> str:
        lines = [
            f"",
            f"{'─'*82}",
            f"  {'ID':>3}  {'Fault Type':<25}  {'True':>12}  {'Predicted':>12}  {'T1':>2}  {'Cat':>3}  {'Conf':>4}",
            f"{'─'*82}",
        ]
        for r in self.per_fault:
            t1  = "✓" if r.top1_correct else "✗"
            cat = "✓" if r.category_correct else "✗"
            lines.append(
                f"  {r.fault_id:>3}  {r.fault_type:<25}  "
                f"{r.true_root_cause:>12}  {r.predicted_root_cause:>12}  "
                f"{t1:>2}  {cat:>3}  {r.predicted_confidence[:4]:>4}"
            )
        lines.append(f"{'─'*82}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<MetricsReport config={self.config!r} "
            f"top1={self.top1_accuracy:.0%} "
            f"mrr={self.mrr:.3f} "
            f"cat={self.category_accuracy:.0%}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# compute_metrics
# ─────────────────────────────────────────────────────────────────────────────

# Normalize agent-reported category names to the canonical set
_CAT_ALIASES: dict[str, str] = {
    "authentication_failure": "auth_failure",
    "cascading_timeouts":     "cascading_timeout",
    "thread_exhaustion":      "thread_pool_exhaustion",
    "thread_pool":            "thread_pool_exhaustion",
    "io_saturation":          "disk_io_saturation",
    "disk_saturation":        "disk_io_saturation",
    "race_condition":         "data_race_condition",
    "memory_growth":          "memory_leak",
    "api_mismatch":           "api_version_mismatch",
    "stale_data":             "stale_cache",
}

def _norm_cat(cat: str) -> str:
    c = cat.lower().strip().replace("-", "_").replace(" ", "_")
    return _CAT_ALIASES.get(c, c)


def compute_metrics(
    results: list[EvaluationResult],
    config:  str = "config_g",
) -> MetricsReport:
    """
    Compute all evaluation metrics from a list of EvaluationResult objects.
    Returns a MetricsReport with all aggregate statistics.
    """
    n = len(results)
    if n == 0:
        return MetricsReport(
            config=config, num_faults=0,
            top1_accuracy=0.0, top3_accuracy=0.0, mrr=0.0,
            category_accuracy=0.0, group_accuracy=0.0, ece=0.0,
            avg_duration_s=0.0,
        )

    # ── Compute per-result correctness ──────────────────────────────────────
    enriched: list[EvaluationResult] = []
    for r in results:
        r_new = EvaluationResult(
            **{k: v for k, v in r.__dict__.items()
               if k not in ("top1_correct", "top3_correct",
                            "category_correct", "group_correct", "rank")},
        )
        # Top-1
        r_new.top1_correct = (
            r.predicted_root_cause.lower().strip() ==
            r.true_root_cause.lower().strip()
        )

        # Rank in candidates
        cands_lower = [c.lower().strip() for c in r.ranked_candidates]
        true_lower  = r.true_root_cause.lower().strip()
        if true_lower in cands_lower:
            r_new.rank = cands_lower.index(true_lower) + 1
        elif r_new.top1_correct:
            r_new.rank = 1
        else:
            r_new.rank = len(cands_lower) + 1  # after all listed

        # Top-3
        r_new.top3_correct = r_new.rank <= 3

        # Category
        r_new.category_correct = (
            _norm_cat(r.predicted_category) == _norm_cat(r.true_category)
        )

        # Group (functional vs non-functional)
        pred_group = _infer_group(_norm_cat(r.predicted_category))
        r_new.group_correct = (pred_group == r.true_category_group)

        enriched.append(r_new)

    # ── Aggregate ───────────────────────────────────────────────────────────
    top1 = sum(r.top1_correct   for r in enriched) / n
    top3 = sum(r.top3_correct   for r in enriched) / n
    cat  = sum(r.category_correct for r in enriched) / n
    grp  = sum(r.group_correct  for r in enriched) / n
    mrr  = sum(1.0 / r.rank     for r in enriched) / n
    avg_dur = sum(r.duration_s  for r in enriched) / n

    # ECE — bin by confidence level, measure calibration gap
    ece = _compute_ece(enriched)

    return MetricsReport(
        config=config,
        num_faults=n,
        top1_accuracy=top1,
        top3_accuracy=top3,
        mrr=mrr,
        category_accuracy=cat,
        group_accuracy=grp,
        ece=ece,
        avg_duration_s=avg_dur,
        per_fault=enriched,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_GROUP_MAP: dict[str, str] = {
    "stale_cache":            "functional",
    "transaction_deadlock":   "functional",
    "api_version_mismatch":   "functional",
    "auth_failure":           "functional",
    "authentication_failure": "functional",
    "data_corruption":        "functional",
    "data_race_condition":    "functional",
    "disk_io_saturation":     "non-functional",
    "memory_leak":            "non-functional",
    "thread_pool_exhaustion": "non-functional",
    "cascading_timeout":      "non-functional",
}

def _infer_group(category: str) -> str:
    return _GROUP_MAP.get(category, "unknown")


def _compute_ece(results: list[EvaluationResult]) -> float:
    """
    Expected Calibration Error over three confidence bins:
      HIGH (0.9), MEDIUM (0.7), LOW (0.4).
    ECE = Σ |accuracy_in_bin - confidence_mean| × (bin_size / total)
    """
    bins: dict[str, list[bool]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for r in results:
        conf = r.predicted_confidence.upper()
        if conf in bins:
            bins[conf].append(r.top1_correct)

    conf_vals = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.4}
    n = len(results)
    ece = 0.0
    for conf_name, correct_list in bins.items():
        if not correct_list:
            continue
        acc_in_bin = sum(correct_list) / len(correct_list)
        conf_val   = conf_vals[conf_name]
        ece += abs(acc_in_bin - conf_val) * (len(correct_list) / n)
    return round(ece, 4)
