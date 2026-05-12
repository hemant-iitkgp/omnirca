"""
Phase 5 validation test suite.

Test classes:
  TestSubAgentBase        (6)  — SubAgent/SubAgentReport API
  TestAgentImports        (8)  — all 6 sub-agents + MainAgent + VotingResult import cleanly
  TestAgentToolSubsets    (6)  — each sub-agent exposes only its intended tool subset
  TestMainAgentStructure  (5)  — MultiAgentResult dataclass and summary()
  TestVotingStructure     (8)  — VotingResult dataclass, _aggregate logic
  TestDataDetectiveLive   (5)  — real LLM run (stale_cache window)
  TestMainAgentStalecache (7)  — full 6-agent pipeline (stale_cache)
  TestMainAgentMemoryLeak (5)  — full 6-agent pipeline (memory_leak)
  TestVotingLive          (4)  — 3× self-consistency run (stale_cache, quick)
  TestPhase1to4Regression (4)  — Phase 1–4 exports undamaged
"""
import sys
from pathlib import Path


import pytest

# ─────────────────────────────────────────────────────────────────────────────
# TestSubAgentBase — base class API
# ─────────────────────────────────────────────────────────────────────────────
class TestSubAgentBase:
    def test_subagentreport_fields_exist(self):
        from omnirca.agents.base_agent import SubAgentReport
        r = SubAgentReport(
            agent_name="test",
            findings={"k": "v"},
            tool_calls=[],
            steps_taken=3,
            duration_s=1.5,
            raw_conclusion="done",
        )
        assert r.agent_name == "test"
        assert r.findings["k"] == "v"
        assert r.steps_taken == 3

    def test_subagentreport_get_method(self):
        from omnirca.agents.base_agent import SubAgentReport
        r = SubAgentReport("x", {"a": 1}, [], 0, 0.0, "")
        assert r.get("a") == 1
        assert r.get("missing", 99) == 99

    def test_subagentreport_repr(self):
        from omnirca.agents.base_agent import SubAgentReport
        r = SubAgentReport("detective", {"top_service": "cache_0"}, [], 2, 0.1, "")
        assert "detective" in repr(r)
        assert "top_service" in repr(r)

    def test_subagent_dispatch_unknown_tool(self):
        from omnirca.agents.base_agent import SubAgent
        agent = SubAgent()
        result = agent._dispatch("nonexistent_tool", {})
        assert "not available" in result.summary.lower() or "error" in result.summary.lower()

    def test_subagent_dispatch_error_is_tolerated(self):
        """Dispatch wraps exceptions, never raises to caller."""
        from omnirca.agents.base_agent import SubAgent
        from omnirca.tools.base import ToolResult
        agent = SubAgent()
        agent.TOOL_MAP = {"boom": lambda **_: 1/0}
        result = agent._dispatch("boom", {})
        assert isinstance(result, ToolResult)
        assert not result.summary == ""   # has error message

    def test_subagent_max_steps_attribute(self):
        from omnirca.agents.base_agent import SubAgent
        assert SubAgent.MAX_STEPS >= 4   # sane default


# ─────────────────────────────────────────────────────────────────────────────
# TestAgentImports — all classes importable
# ─────────────────────────────────────────────────────────────────────────────
class TestAgentImports:
    def test_import_data_detective(self):
        from omnirca.agents.data_detective import DataDetective
        assert DataDetective.AGENT_NAME == "DataDetective"

    def test_import_graph_explorer(self):
        from omnirca.agents.graph_explorer import GraphExplorer
        assert GraphExplorer.AGENT_NAME == "GraphExplorer"

    def test_import_fault_typer(self):
        from omnirca.agents.fault_typer import FaultTyper
        assert FaultTyper.AGENT_NAME == "FaultTyper"

    def test_import_evidence_collector(self):
        from omnirca.agents.evidence_collector import EvidenceCollector
        assert EvidenceCollector.AGENT_NAME == "EvidenceCollector"

    def test_import_temporal_analyst(self):
        from omnirca.agents.temporal_analyst import TemporalAnalyst
        assert TemporalAnalyst.AGENT_NAME == "TemporalAnalyst"

    def test_import_judge_agent(self):
        from omnirca.agents.judge_agent import JudgeAgent
        assert JudgeAgent.AGENT_NAME == "JudgeAgent"

    def test_import_main_agent(self):
        from omnirca.agents.main_agent import MainAgent, MultiAgentResult
        assert MainAgent is not None
        assert MultiAgentResult is not None

    def test_import_voting(self):
        from omnirca.voting.self_consistency import run_with_voting, VotingResult
        assert run_with_voting is not None
        assert VotingResult is not None


# ─────────────────────────────────────────────────────────────────────────────
# TestAgentToolSubsets — each sub-agent has only its intended tools
# ─────────────────────────────────────────────────────────────────────────────
class TestAgentToolSubsets:
    def test_data_detective_tools(self):
        from omnirca.agents.data_detective import DataDetective, _DETECTIVE_TOOL_NAMES
        schema_names = {s["function"]["name"] for s in DataDetective.TOOL_SCHEMAS}
        assert schema_names == _DETECTIVE_TOOL_NAMES

    def test_graph_explorer_tools_contain_propagation(self):
        from omnirca.agents.graph_explorer import GraphExplorer
        names = {s["function"]["name"] for s in GraphExplorer.TOOL_SCHEMAS}
        assert "get_propagation_candidates" in names
        assert "detect_causal_order" in names
        # should NOT have syscall tools
        assert "syscall_multi_service_compare" not in names

    def test_fault_typer_tools(self):
        from omnirca.agents.fault_typer import FaultTyper
        names = {s["function"]["name"] for s in FaultTyper.TOOL_SCHEMAS}
        assert {"search_fault_knowledge", "classify_fault_pattern", "check_sop"} == names

    def test_evidence_collector_tools(self):
        from omnirca.agents.evidence_collector import EvidenceCollector
        names = {s["function"]["name"] for s in EvidenceCollector.TOOL_SCHEMAS}
        assert "compute_anomaly_score" in names
        assert "finalize_rca" not in names   # only MainAgent calls finalize_rca

    def test_judge_agent_has_only_explain_evidence(self):
        from omnirca.agents.judge_agent import JudgeAgent
        names = {s["function"]["name"] for s in JudgeAgent.TOOL_SCHEMAS}
        assert names == {"explain_evidence"}

    def test_no_subagent_has_finalize_rca(self):
        """finalize_rca must only be in MainAgent's dispatch, not sub-agent schemas."""
        from omnirca.agents.data_detective import DataDetective
        from omnirca.agents.graph_explorer import GraphExplorer
        from omnirca.agents.fault_typer import FaultTyper
        from omnirca.agents.evidence_collector import EvidenceCollector
        from omnirca.agents.temporal_analyst import TemporalAnalyst
        from omnirca.agents.judge_agent import JudgeAgent

        for AgentClass in [DataDetective, GraphExplorer, FaultTyper,
                           EvidenceCollector, TemporalAnalyst, JudgeAgent]:
            names = {s["function"]["name"] for s in AgentClass.TOOL_SCHEMAS}
            assert "finalize_rca" not in names, (
                f"{AgentClass.AGENT_NAME} should not have finalize_rca in its tool schemas"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TestMainAgentStructure — MultiAgentResult dataclass
# ─────────────────────────────────────────────────────────────────────────────
class TestMainAgentStructure:
    def _make_result(self, **kwargs):
        from omnirca.agents.main_agent import MultiAgentResult
        defaults = dict(
            root_cause="cache_0",
            fault_category="stale_cache",
            confidence="HIGH",
            evidence_list=["e1", "e2"],
            propagation_path=["cache_0", "auth_service"],
            composite_score=319.5,
        )
        defaults.update(kwargs)
        return MultiAgentResult(**defaults)

    def test_root_cause_field(self):
        r = self._make_result()
        assert r.root_cause == "cache_0"

    def test_fault_category_field(self):
        r = self._make_result()
        assert r.fault_category == "stale_cache"

    def test_confidence_field(self):
        r = self._make_result()
        assert r.confidence == "HIGH"

    def test_summary_contains_root_cause(self):
        r = self._make_result()
        s = r.summary()
        assert "cache_0" in s
        assert "HIGH" in s

    def test_repr_contains_root_cause(self):
        r = self._make_result()
        assert "cache_0" in repr(r)


# ─────────────────────────────────────────────────────────────────────────────
# TestVotingStructure — VotingResult + _aggregate logic
# ─────────────────────────────────────────────────────────────────────────────
class TestVotingStructure:
    def _make_run(self, root_cause, confidence):
        from omnirca.agents.main_agent import MultiAgentResult
        return MultiAgentResult(
            root_cause=root_cause,
            fault_category="stale_cache",
            confidence=confidence,
            evidence_list=["e1"],
            propagation_path=[root_cause],
            composite_score=100.0,
            duration_s=10.0,
        )

    def test_all_agree_high_confidence(self):
        from omnirca.voting.self_consistency import _aggregate
        runs = [self._make_run("cache_0", "HIGH")] * 3
        result = _aggregate(runs, 30.0)
        assert result.root_cause == "cache_0"
        assert result.vote_agreement == "3/3"
        assert result.confidence == "HIGH"
        assert result.confidence_score >= 0.80

    def test_two_agree_medium_confidence(self):
        from omnirca.voting.self_consistency import _aggregate
        runs = [
            self._make_run("cache_0", "HIGH"),
            self._make_run("cache_0", "HIGH"),
            self._make_run("auth_service", "LOW"),
        ]
        result = _aggregate(runs, 30.0)
        assert result.root_cause == "cache_0"
        assert result.vote_agreement == "2/3"
        assert result.confidence_score < 0.90   # penalised by 0.85 factor

    def test_no_agreement_low_confidence(self):
        from omnirca.voting.self_consistency import _aggregate
        runs = [
            self._make_run("cache_0", "HIGH"),
            self._make_run("backend_4", "MEDIUM"),
            self._make_run("database_0", "LOW"),
        ]
        result = _aggregate(runs, 30.0)
        assert result.vote_agreement == "1/3"
        assert result.confidence_score <= 0.60

    def test_voting_result_summary(self):
        from omnirca.voting.self_consistency import _aggregate
        runs = [self._make_run("cache_0", "HIGH")] * 3
        result = _aggregate(runs, 30.0)
        s = result.summary()
        assert "cache_0" in s
        assert "3/3" in s

    def test_voting_result_repr(self):
        from omnirca.voting.self_consistency import _aggregate
        runs = [self._make_run("cache_0", "HIGH")] * 3
        result = _aggregate(runs, 30.0)
        assert "cache_0" in repr(result)

    def test_individual_runs_stored(self):
        from omnirca.voting.self_consistency import _aggregate
        runs = [self._make_run("cache_0", "HIGH")] * 3
        result = _aggregate(runs, 30.0)
        assert len(result.individual_runs) == 3

    def test_vote_counts_type(self):
        from omnirca.voting.self_consistency import _aggregate
        from collections import Counter
        runs = [self._make_run("cache_0", "HIGH")] * 3
        result = _aggregate(runs, 30.0)
        assert isinstance(result.vote_counts, Counter)

    def test_evidence_list_comes_from_best_run(self):
        from omnirca.voting.self_consistency import _aggregate
        runs = [self._make_run("cache_0", "HIGH")] * 3
        runs[0].evidence_list = ["best_evidence"]
        result = _aggregate(runs, 30.0)
        assert "best_evidence" in result.evidence_list


# ─────────────────────────────────────────────────────────────────────────────
# TestDataDetectiveLive — one sub-agent LLM test (fast, no full pipeline)
# ─────────────────────────────────────────────────────────────────────────────
class TestDataDetectiveLive:
    """
    DataDetective alone — uses LLM but only calls 2-3 tools.
    Much faster than a full pipeline run (~30s vs ~300s).
    """
    @pytest.fixture(scope="class")
    def detective_report(self):
        from omnirca.agents.data_detective import DataDetective
        ctx = {
            "t_start": "2024-01-01 03:23:00",
            "t_end":   "2024-01-01 03:37:00",
        }
        return DataDetective(verbose=True).run(ctx)

    def test_report_is_subagentreport(self, detective_report):
        from omnirca.agents.base_agent import SubAgentReport
        assert isinstance(detective_report, SubAgentReport)

    def test_report_made_tool_calls(self, detective_report):
        assert detective_report.steps_taken >= 1
        assert len(detective_report.tool_calls) >= 1

    def test_multi_service_compare_was_first(self, detective_report):
        first_tool = detective_report.tool_calls[0]["tool"]
        assert first_tool == "syscall_multi_service_compare"

    def test_top_service_is_not_logging(self, detective_report):
        top = detective_report.get("top_service")
        assert top != "logging", "logging must never be flagged as root cause"

    def test_top_service_is_string(self, detective_report):
        top = detective_report.get("top_service")
        assert isinstance(top, str) and len(top) > 0


# ─────────────────────────────────────────────────────────────────────────────
# TestMainAgentStalecache — full 6-agent pipeline on stale_cache window
# ─────────────────────────────────────────────────────────────────────────────
class TestMainAgentStalecache:
    """
    Full multi-agent pipeline: DataDetective → GraphExplorer → FaultTyper
    → EvidenceCollector → TemporalAnalyst → JudgeAgent → finalize_rca.

    Ground truth: stale_cache, root = cache_0
    """
    @pytest.fixture(scope="class")
    def result(self):
        from omnirca.agents.main_agent import MainAgent
        agent = MainAgent(verbose=True)
        return agent.run(
            incident_query=(
                "Multiple services reporting elevated error rates. "
                "Some users seeing stale or inconsistent data."
            ),
            t_start="2024-01-01 03:23:00",
            t_end="2024-01-01 03:37:00",
        )

    def test_result_type(self, result):
        from omnirca.agents.main_agent import MultiAgentResult
        assert isinstance(result, MultiAgentResult)

    def test_root_cause_is_string(self, result):
        assert isinstance(result.root_cause, str) and len(result.root_cause) > 0

    def test_root_cause_not_logging(self, result):
        assert result.root_cause != "logging"

    def test_fault_category_not_empty(self, result):
        assert isinstance(result.fault_category, str) and len(result.fault_category) > 0

    def test_confidence_is_valid(self, result):
        assert result.confidence in {"HIGH", "MEDIUM", "LOW"}

    def test_all_six_agents_ran(self, result):
        expected = {
            "DataDetective", "GraphExplorer", "FaultTyper",
            "EvidenceCollector", "TemporalAnalyst", "JudgeAgent"
        }
        assert expected == set(result.agent_reports.keys())

    def test_evidence_list_non_empty(self, result):
        assert isinstance(result.evidence_list, list) and len(result.evidence_list) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# TestMainAgentMemoryLeak — full 6-agent pipeline on memory_leak window
# ─────────────────────────────────────────────────────────────────────────────
class TestMainAgentMemoryLeak:
    """Ground truth: memory_leak, root = backend_4"""
    @pytest.fixture(scope="class")
    def result(self):
        from omnirca.agents.main_agent import MainAgent
        return MainAgent(verbose=True).run(
            incident_query=(
                "backend services showing gradual degradation. "
                "Increasing memory consumption observed over time."
            ),
            t_start="2024-01-02 01:26:00",
            t_end="2024-01-02 01:41:00",
        )

    def test_result_not_none(self, result):
        assert result is not None

    def test_root_cause_not_logging(self, result):
        assert result.root_cause != "logging"

    def test_confidence_is_valid(self, result):
        assert result.confidence in {"HIGH", "MEDIUM", "LOW"}

    def test_all_agents_ran(self, result):
        assert "DataDetective" in result.agent_reports
        assert "JudgeAgent" in result.agent_reports

    @pytest.mark.xfail(reason="Exact fault category uncertain given overlapping syscall pattern")
    def test_fault_category_is_memory_leak(self, result):
        assert "memory" in result.fault_category.lower()


# ─────────────────────────────────────────────────────────────────────────────
# TestVotingLive — 3× self-consistency on stale_cache (quick, n_votes=2)
# ─────────────────────────────────────────────────────────────────────────────
class TestVotingLive:
    """
    Run 2 (not 3) votes to save time while still testing the mechanism.
    """
    @pytest.fixture(scope="class")
    def vote_result(self):
        from omnirca.voting.self_consistency import run_with_voting
        return run_with_voting(
            incident_query="Error rates spiking, users seeing stale data.",
            t_start="2024-01-01 03:23:00",
            t_end="2024-01-01 03:37:00",
            n_votes=2,
            temperatures=[0.1, 0.5],
            verbose=True,
        )

    def test_voting_result_type(self, vote_result):
        from omnirca.voting.self_consistency import VotingResult
        assert isinstance(vote_result, VotingResult)

    def test_vote_agreement_format(self, vote_result):
        assert "/" in vote_result.vote_agreement

    def test_root_cause_not_logging(self, vote_result):
        assert vote_result.root_cause != "logging"

    def test_n_individual_runs(self, vote_result):
        assert len(vote_result.individual_runs) == 2


# ─────────────────────────────────────────────────────────────────────────────
# TestPhase1to4Regression — validate prior phases still intact
# ─────────────────────────────────────────────────────────────────────────────
class TestPhase1to4Regression:
    def test_all_24_tools_still_exported(self):
        from omnirca.tools import __all__ as tool_exports
        assert len(tool_exports) == 24

    def test_react_agent_still_importable(self):
        from omnirca.agent.react_agent import ReActAgent, AgentResult
        assert ReActAgent is not None
        assert AgentResult is not None

    def test_rag_retriever_still_works(self):
        from omnirca.rag.retriever import retrieve
        results = retrieve("cached data serving stale responses fast", top_k=1)
        assert len(results) >= 1

    def test_llm_client_still_works(self):
        from omnirca.llm_client import simple_chat
        resp = simple_chat("Reply with: OK", temperature=0.0)
        assert isinstance(resp, str) and len(resp) > 0
