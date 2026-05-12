"""
omnirca/evaluation — Phase 7 Evaluation Framework.

Exports:
  QueryBuilder      — builds blind evaluation queries from fault metadata
  EvaluationResult  — result for a single fault run
  MetricsReport     — aggregate accuracy/calibration metrics
  compute_metrics   — computes all metrics from a list of EvaluationResults
  EvaluationHarness — orchestrates blind evaluation over all 10 faults
  AblationRunner    — runs the 7-configuration ablation study
"""
from omnirca.evaluation.query_builder import QueryBuilder, FaultQuery
from omnirca.evaluation.metrics import EvaluationResult, MetricsReport, compute_metrics
from omnirca.evaluation.test_harness import EvaluationHarness
from omnirca.evaluation.ablation import AblationRunner, ABLATION_CONFIGS

__all__ = [
    "QueryBuilder", "FaultQuery",
    "EvaluationResult", "MetricsReport", "compute_metrics",
    "EvaluationHarness",
    "AblationRunner", "ABLATION_CONFIGS",
]
