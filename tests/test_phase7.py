"""
Phase 7 tests — blind evaluation framework (plan.md §7).

Structure
─────────
TestEvaluationPackageImports (4)
    — evaluation package importable, all public exports present

TestQueryBuilderStructure (6)
    — FaultQuery fields, all 10 faults build, fairness (root cause not leaked)

TestMetricsCalculation (8)
    — EvaluationResult dataclass, MRR, Top-1/3, ECE, category aliases

TestHarnessStructure (5)
    — EvaluationHarness API, 10 faults in query builder, ground truth names

TestAblationStructure (5)
    — 7 configs defined, names A-G, runnable flags, AblationRunner API

TestPhase7Regression (2)
    — Phase 5 & 6 not damaged

Total: 30 structural tests (no LLM calls)
"""
import math
import os
import re
import sys
import unittest
from pathlib import Path

# ── Workspace root on sys.path ────────────────────────────────────────────────
_REPO_ROOT = str(Path(__file__).parent.parent.parent)

_FAULTS_CSV = os.path.join(_REPO_ROOT, "datasets_complex", "faults.csv")


# ═════════════════════════════════════════════════════════════════════════════
# Group 1 — Package-level imports
# ═════════════════════════════════════════════════════════════════════════════

class TestEvaluationPackageImports(unittest.TestCase):
    """Verify phase 7 evaluation package is importable and exposes its API."""

    def test_evaluation_package_importable(self):
        """omnirca.evaluation package imports without error."""
        import omnirca.evaluation  # noqa: F401 — import side-effect test

    def test_query_builder_class_exported(self):
        """QueryBuilder and FaultQuery are available at package level."""
        from omnirca.evaluation import QueryBuilder, FaultQuery  # noqa
        self.assertTrue(callable(QueryBuilder))
        self.assertTrue(hasattr(FaultQuery, "__dataclass_fields__"))

    def test_metrics_classes_exported(self):
        """EvaluationResult, MetricsReport, compute_metrics exported."""
        from omnirca.evaluation import (
            EvaluationResult, MetricsReport, compute_metrics
        )
        self.assertTrue(callable(compute_metrics))
        self.assertTrue(hasattr(EvaluationResult, "__dataclass_fields__"))
        self.assertTrue(hasattr(MetricsReport, "__dataclass_fields__"))

    def test_ablation_exports(self):
        """AblationRunner and ABLATION_CONFIGS are exported."""
        from omnirca.evaluation import AblationRunner, ABLATION_CONFIGS
        self.assertIsInstance(ABLATION_CONFIGS, dict)
        self.assertTrue(callable(AblationRunner))


# ═════════════════════════════════════════════════════════════════════════════
# Group 2 — QueryBuilder structure
# ═════════════════════════════════════════════════════════════════════════════

class TestQueryBuilderStructure(unittest.TestCase):
    """Structural tests for QueryBuilder — no LLM, no agent calls."""

    @classmethod
    def setUpClass(cls):
        from omnirca.evaluation.query_builder import QueryBuilder
        cls.qb = QueryBuilder(_FAULTS_CSV)
        cls.queries = cls.qb.build_all()

    def test_build_all_returns_ten_queries(self):
        """QueryBuilder.build_all() returns exactly 10 FaultQuery objects."""
        self.assertEqual(len(self.queries), 10)

    def test_fault_query_has_required_fields(self):
        """FaultQuery dataclass has all required fields."""
        fq = self.queries[0]
        required = (
            "fault_id", "t_start", "t_end", "reported_service",
            "symptom", "query_text", "ground_truth_service",
            "ground_truth_category", "ground_truth_category_group",
        )
        for field in required:
            self.assertTrue(
                hasattr(fq, field),
                f"FaultQuery missing field: {field!r}",
            )

    def test_query_text_contains_time_window(self):
        """Every query_text includes the t_start time window."""
        for fq in self.queries:
            self.assertIn(
                fq.t_start, fq.query_text,
                f"Fault {fq.fault_id}: t_start not in query_text",
            )

    def test_query_text_contains_reported_service(self):
        """Every query_text mentions the reported affected service."""
        for fq in self.queries:
            self.assertIn(
                fq.reported_service, fq.query_text,
                f"Fault {fq.fault_id}: reported_service not in query_text",
            )

    def test_fairness_root_cause_not_in_query_for_victim_faults(self):
        """
        For faults where the REPORTED service is a victim (not the root cause),
        the root cause service name MUST NOT appear in query_text.
        Faults 0, 1, 2, 5 have victim != root.
        """
        victim_faults = {0: "cache_0", 1: "database_2", 2: "backend_6", 5: "database_2"}
        for fid, root_name in victim_faults.items():
            fq = self.queries[fid]
            self.assertNotIn(
                root_name, fq.query_text,
                f"Fault {fid}: root cause service '{root_name}' leaked into query_text!",
            )

    def test_ground_truth_service_not_empty(self):
        """ground_truth_service is a non-empty string for all 10 faults."""
        for fq in self.queries:
            self.assertTrue(
                len(fq.ground_truth_service.strip()) > 0,
                f"Fault {fq.fault_id}: ground_truth_service is empty",
            )


# ═════════════════════════════════════════════════════════════════════════════
# Group 3 — Metrics calculation
# ═════════════════════════════════════════════════════════════════════════════

class TestMetricsCalculation(unittest.TestCase):
    """Unit-test metric calculations with synthetic EvaluationResult objects."""

    @staticmethod
    def _make_result(
        fault_id=0,
        true_root="cache_0",
        predicted_root="cache_0",
        true_cat="stale_cache",
        pred_cat="stale_cache",
        true_group="functional",
        confidence="HIGH",
        ranked=None,
        duration=1.5,
        error=None,
    ):
        from omnirca.evaluation.metrics import EvaluationResult
        return EvaluationResult(
            fault_id             = fault_id,
            fault_type           = true_cat,
            t_start              = "2024-01-01 03:23:36",
            t_end                = "2024-01-01 03:37:35",
            reported_service     = "auth_service",
            true_root_cause      = true_root,
            true_category        = true_cat,
            true_category_group  = true_group,
            predicted_root_cause = predicted_root,
            predicted_category   = pred_cat,
            predicted_confidence = confidence,
            ranked_candidates    = ranked if ranked is not None else [predicted_root],
            duration_s           = duration,
            error                = error,
        )

    def test_evaluation_result_instantiates(self):
        """EvaluationResult dataclass instantiates with all required fields."""
        r = self._make_result()
        self.assertEqual(r.fault_id, 0)
        self.assertEqual(r.true_root_cause, "cache_0")
        self.assertEqual(r.predicted_root_cause, "cache_0")

    def test_confidence_float_property(self):
        """confidence_float converts HIGH/MEDIUM/LOW to correct probabilities."""
        from omnirca.evaluation.metrics import EvaluationResult
        for conf, expected in [("HIGH", 0.9), ("MEDIUM", 0.7), ("LOW", 0.4)]:
            r = self._make_result(confidence=conf)
            self.assertAlmostEqual(r.confidence_float, expected)

    def test_compute_metrics_perfect_score(self):
        """All-correct results → Top-1=100%, MRR=1.0."""
        from omnirca.evaluation.metrics import compute_metrics
        results = [
            self._make_result(fault_id=i, true_root=f"svc_{i}", predicted_root=f"svc_{i}",
                              ranked=[f"svc_{i}"])
            for i in range(5)
        ]
        report = compute_metrics(results)
        self.assertAlmostEqual(report.top1_accuracy, 1.0)
        self.assertAlmostEqual(report.mrr, 1.0)

    def test_compute_metrics_zero_score(self):
        """All-wrong results → Top-1=0%.  Top-3=0% when true svc absent from 5-deep list."""
        from omnirca.evaluation.metrics import compute_metrics
        # 5 wrong candidates — true_svc absent, fallback rank = 6 > 3 → top3=False
        results = [
            self._make_result(
                true_root      = "true_svc",
                predicted_root = "wrong_1",
                ranked         = ["wrong_1", "wrong_2", "wrong_3", "wrong_4", "wrong_5"],
            )
        ]
        report = compute_metrics(results)
        self.assertAlmostEqual(report.top1_accuracy, 0.0)
        self.assertAlmostEqual(report.top3_accuracy, 0.0)

    def test_mrr_rank2_gives_half(self):
        """Ground truth at rank 2 → MRR contribution of 0.5."""
        from omnirca.evaluation.metrics import compute_metrics
        results = [
            self._make_result(
                true_root  = "true_svc",
                predicted_root = "wrong_svc",
                ranked     = ["wrong_svc", "true_svc", "third_svc"],
            )
        ]
        report = compute_metrics(results)
        # rank=2 so MRR=0.5 and top3=True
        self.assertAlmostEqual(report.mrr, 0.5)
        self.assertAlmostEqual(report.top3_accuracy, 1.0)

    def test_category_alias_normalisation(self):
        """'authentication_failure' (CSV label) equals 'auth_failure' (SOP key)."""
        from omnirca.evaluation.metrics import compute_metrics
        results = [
            self._make_result(
                true_cat   = "authentication_failure",
                pred_cat   = "auth_failure",       # agent may output either
                true_group = "functional",
            )
        ]
        report = compute_metrics(results)
        self.assertAlmostEqual(report.category_accuracy, 1.0,
                               msg="auth_failure alias not normalised correctly")

    def test_compute_metrics_empty_returns_zero_report(self):
        """Empty results list → MetricsReport with all zeros."""
        from omnirca.evaluation.metrics import compute_metrics
        report = compute_metrics([])
        self.assertEqual(report.num_faults, 0)
        self.assertAlmostEqual(report.top1_accuracy, 0.0)

    def test_metrics_report_summary_string(self):
        """MetricsReport.summary() returns a multi-line string with key labels."""
        from omnirca.evaluation.metrics import compute_metrics
        results = [self._make_result()]
        report = compute_metrics(results)
        summary = report.summary()
        self.assertIn("Top-1", summary)
        self.assertIn("MRR", summary)
        self.assertIn("ECE", summary)


# ═════════════════════════════════════════════════════════════════════════════
# Group 4 — EvaluationHarness structure
# ═════════════════════════════════════════════════════════════════════════════

class TestHarnessStructure(unittest.TestCase):
    """Structural tests for EvaluationHarness — no LLM calls."""

    @classmethod
    def setUpClass(cls):
        from omnirca.evaluation.test_harness import EvaluationHarness
        cls.harness = EvaluationHarness(faults_csv=_FAULTS_CSV)

    def test_harness_instantiates(self):
        """EvaluationHarness constructs without error."""
        from omnirca.evaluation.test_harness import EvaluationHarness
        h = EvaluationHarness(faults_csv=_FAULTS_CSV)
        self.assertIsNotNone(h)

    def test_harness_query_builder_has_10_faults(self):
        """Harness query_builder contains entries for fault IDs 0-9."""
        queries = self.harness.query_builder.build_all()
        ids = {fq.fault_id for fq in queries}
        self.assertEqual(ids, set(range(10)))

    def test_ground_truth_service_names_are_valid(self):
        """Ground truth root cause services match the known service name list."""
        known_services = {
            "frontend_0", "frontend_1", "api_gateway",
            "backend_0",  "backend_1",  "backend_2",  "backend_3",
            "backend_4",  "backend_5",  "backend_6",  "backend_7",
            "database_0", "database_1", "database_2",
            "cache_0",    "cache_1",
            "message_queue_0", "message_queue_1",
            "auth_service", "logging",
        }
        for fq in self.harness.query_builder.build_all():
            self.assertIn(
                fq.ground_truth_service, known_services,
                f"Fault {fq.fault_id}: unknown ground truth service "
                f"'{fq.ground_truth_service}'",
            )

    def test_run_single_method_exists(self):
        """EvaluationHarness has a callable run_single method."""
        self.assertTrue(callable(getattr(self.harness, "run_single", None)))

    def test_run_all_method_exists(self):
        """EvaluationHarness has a callable run_all method."""
        self.assertTrue(callable(getattr(self.harness, "run_all", None)))


# ═════════════════════════════════════════════════════════════════════════════
# Group 5 — Ablation structure
# ═════════════════════════════════════════════════════════════════════════════

class TestAblationStructure(unittest.TestCase):
    """Structural tests for the ablation framework — no LLM calls."""

    def test_ablation_configs_importable(self):
        """ABLATION_CONFIGS dict is importable and non-empty."""
        from omnirca.evaluation.ablation import ABLATION_CONFIGS
        self.assertIsInstance(ABLATION_CONFIGS, dict)
        self.assertGreater(len(ABLATION_CONFIGS), 0)

    def test_exactly_seven_configs(self):
        """Exactly 7 ablation configurations are defined (A-G)."""
        from omnirca.evaluation.ablation import ABLATION_CONFIGS
        self.assertEqual(len(ABLATION_CONFIGS), 7,
                         f"Expected 7, got {len(ABLATION_CONFIGS)}: "
                         f"{list(ABLATION_CONFIGS.keys())}")

    def test_config_names_a_through_g(self):
        """Config keys are config_a through config_g."""
        from omnirca.evaluation.ablation import ABLATION_CONFIGS
        expected = {f"config_{c}" for c in "abcdefg"}
        self.assertEqual(set(ABLATION_CONFIGS.keys()), expected)

    def test_config_g_is_runnable(self):
        """config_g (full system) must be marked runnable=True."""
        from omnirca.evaluation.ablation import ABLATION_CONFIGS
        self.assertTrue(ABLATION_CONFIGS["config_g"].runnable,
                        "config_g should be runnable=True")

    def test_ablation_runner_instantiates(self):
        """AblationRunner constructs without error."""
        from omnirca.evaluation.ablation import AblationRunner
        runner = AblationRunner(faults_csv=_FAULTS_CSV)
        self.assertIsNotNone(runner)


# ═════════════════════════════════════════════════════════════════════════════
# Group 6 — Regression: Phase 5 & 6 not damaged
# ═════════════════════════════════════════════════════════════════════════════

class TestPhase7Regression(unittest.TestCase):
    """Verify that Phase 7 files didn't break Phase 5 or Phase 6 imports."""

    def test_phase6_sop_library_intact(self):
        """SopLibrary still importable and has ≥11 built-in SOPs."""
        from omnirca.sops.sop_library import SopLibrary
        lib = SopLibrary()
        cats = lib.list_categories()
        self.assertGreaterEqual(len(cats), 11,
                                f"Expected ≥11 SOPs, got {len(cats)}: {cats}")

    def test_phase5_main_agent_class_importable(self):
        """MainAgent class is importable (Phase 5 multi-agent core)."""
        from omnirca.agents.main_agent import MainAgent
        self.assertTrue(callable(MainAgent))


# ═════════════════════════════════════════════════════════════════════════════
# Entry-point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
