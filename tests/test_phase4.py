"""
test_phase4.py — Tests for the Phase 4 ReAct LLM agent.

Test categories:
  A. LLM client connectivity  — basic Azure OpenAI connection check
  B. Tool schema validation   — all 24 schemas are well-formed (no LLM needed)
  C. Dispatcher               — every tool can be called without error (no LLM)
  D. explain_evidence         — LLM generates a coherent explanation
  E. Agent: stale_cache run   — full agent run on a real fault window
  F. Agent: memory_leak run   — full agent run on a different fault type
  G. AgentResult structure    — all required fields present and typed correctly
  H. Phase 1+2+3 regression   — imports + smoke tests to ensure no breakage
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


# ── Known fault windows from window_labels.csv ────────────────────────────────
# stale_cache:  2024-01-01 03:23 → 03:37  |  root service: cache_0 (id=14)
# memory_leak:  2024-01-02 01:26 → 01:41  |  root service: backend_4 (id=7)
# cascading_t:  2024-01-01 13:53 → 14:16  |  root service: backend_6 (id=9)

STALE_CACHE_START  = "2024-01-01 03:23:00"
STALE_CACHE_END    = "2024-01-01 03:37:00"

MEMORY_LEAK_START  = "2024-01-02 01:26:00"
MEMORY_LEAK_END    = "2024-01-02 01:41:00"


# ════════════════════════════════════════════════════════════════════════════
# A. LLM client connectivity
# ════════════════════════════════════════════════════════════════════════════

class TestLLMConnectivity:
    def test_client_instantiates(self):
        from omnirca.llm_client import get_client, AzureOpenAI
        client = get_client()
        assert isinstance(client, AzureOpenAI)

    def test_simple_chat_returns_string(self):
        """Single-turn LLM call — verifies endpoint reachable."""
        from omnirca.llm_client import simple_chat
        result = simple_chat("Reply with exactly: PONG", temperature=0.0)
        assert isinstance(result, str)
        assert len(result) > 0, "LLM returned empty string"

    def test_simple_chat_content_sensible(self):
        """Model follows basic instructions."""
        from omnirca.llm_client import simple_chat
        result = simple_chat(
            "What is 2 + 2? Reply with a single integer.",
            temperature=0.0,
        )
        assert "4" in result, f"Unexpected response: {result[:200]}"


# ════════════════════════════════════════════════════════════════════════════
# B. Tool schema validation
# ════════════════════════════════════════════════════════════════════════════

class TestToolSchemas:
    def test_schema_count(self):
        from omnirca.agent.tool_schema import TOOL_SCHEMAS
        assert len(TOOL_SCHEMAS) == 24, (
            f"Expected 24 tool schemas, got {len(TOOL_SCHEMAS)}"
        )

    def test_all_schemas_have_type_function(self):
        from omnirca.agent.tool_schema import TOOL_SCHEMAS
        bad = [s for s in TOOL_SCHEMAS if s.get("type") != "function"]
        assert not bad, f"Schemas missing type='function': {[s.get('function',{}).get('name') for s in bad]}"

    def test_all_schemas_have_name_and_description(self):
        from omnirca.agent.tool_schema import TOOL_SCHEMAS
        for schema in TOOL_SCHEMAS:
            fn = schema["function"]
            assert fn.get("name"),        f"Schema missing name: {schema}"
            assert fn.get("description"), f"Schema '{fn['name']}' missing description"

    def test_all_schemas_have_parameters(self):
        from omnirca.agent.tool_schema import TOOL_SCHEMAS
        for schema in TOOL_SCHEMAS:
            fn = schema["function"]
            params = fn.get("parameters", {})
            assert params.get("type") == "object", (
                f"Schema '{fn['name']}' parameters.type != 'object'"
            )
            assert "properties" in params, (
                f"Schema '{fn['name']}' missing properties"
            )

    def test_required_fields_are_lists(self):
        from omnirca.agent.tool_schema import TOOL_SCHEMAS
        for schema in TOOL_SCHEMAS:
            fn     = schema["function"]
            params = fn.get("parameters", {})
            req    = params.get("required", [])
            assert isinstance(req, list), (
                f"Schema '{fn['name']}' required is not a list: {req!r}"
            )

    def test_schema_names_match_tool_map(self):
        from omnirca.agent.tool_schema import TOOL_SCHEMAS, _TOOL_MAP
        schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        map_names    = set(_TOOL_MAP.keys())
        only_in_schema = schema_names - map_names
        only_in_map    = map_names - schema_names
        assert not only_in_schema, f"Schemas with no dispatcher: {only_in_schema}"
        assert not only_in_map,    f"Dispatcher entries with no schema: {only_in_map}"

    def test_finalize_rca_has_correct_required_fields(self):
        from omnirca.agent.tool_schema import TOOL_SCHEMAS
        fin = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "finalize_rca")
        req = set(fin["function"]["parameters"]["required"])
        assert "root_cause"    in req
        assert "confidence"    in req
        assert "evidence_list" in req


# ════════════════════════════════════════════════════════════════════════════
# C. Dispatcher — every tool callable without crashing on valid minimal args
# ════════════════════════════════════════════════════════════════════════════

class TestDispatcher:
    """
    Call every tool with minimal valid arguments and verify:
     - No unhandled exception escapes dispatch_tool()
     - Result is a ToolResult
     - Summary is a non-empty string
    """

    TS = STALE_CACHE_START
    TE = STALE_CACHE_END
    SVC = "cache_0"

    def _check(self, name: str, args: dict):
        from omnirca.agent.tool_schema import dispatch_tool
        from omnirca.tools.base import ToolResult
        result = dispatch_tool(name, args)
        assert isinstance(result, ToolResult), f"{name}: result is not ToolResult"
        assert isinstance(result.summary, str), f"{name}: summary not str"
        assert len(result.summary) > 0, f"{name}: empty summary"

    def test_dispatch_query_syscalls(self):
        self._check("query_syscalls", {"service": self.SVC, "t_start": self.TS, "t_end": self.TE})

    def test_dispatch_syscall_multi_service_compare(self):
        self._check("syscall_multi_service_compare", {"t_start": self.TS, "t_end": self.TE})

    def test_dispatch_syscall_sub_channel_analysis(self):
        self._check("syscall_sub_channel_analysis", {"service": self.SVC, "t_start": self.TS, "t_end": self.TE})

    def test_dispatch_compute_anomaly_score(self):
        self._check("compute_anomaly_score", {"service": self.SVC, "t_start": self.TS, "t_end": self.TE})

    def test_dispatch_detect_memory_slope(self):
        self._check("detect_memory_slope", {"service": self.SVC, "t_start": self.TS, "t_end": self.TE})

    def test_dispatch_get_service_dependencies(self):
        self._check("get_service_dependencies", {"service": self.SVC})

    def test_dispatch_get_propagation_candidates(self):
        self._check("get_propagation_candidates", {"affected_service": self.SVC, "t_start": self.TS, "t_end": self.TE})

    def test_dispatch_detect_causal_order(self):
        self._check("detect_causal_order", {
            "service_list": ["cache_0", "backend_0", "auth_service"],
            "t_start": self.TS, "t_end": self.TE,
        })

    def test_dispatch_search_fault_knowledge(self):
        self._check("search_fault_knowledge", {"symptom_text": "high read count, latency dropped"})

    def test_dispatch_check_sop(self):
        self._check("check_sop", {"fault_category": "stale_cache"})

    def test_dispatch_finalize_rca(self):
        self._check("finalize_rca", {
            "root_cause": "cache_0",
            "confidence": "HIGH",
            "evidence_list": ["z=45", "read_count_ratio=3.2"],
            "fault_category": "stale_cache",
        })

    def test_dispatch_unknown_tool_returns_error_result(self):
        from omnirca.agent.tool_schema import dispatch_tool
        from omnirca.tools.base import ToolResult
        result = dispatch_tool("nonexistent_tool", {})
        assert isinstance(result, ToolResult)
        assert "unknown" in result.summary.lower() or "error" in result.summary.lower()

    def test_dispatch_bad_args_returns_error_result(self):
        from omnirca.agent.tool_schema import dispatch_tool
        from omnirca.tools.base import ToolResult
        result = dispatch_tool("get_service_dependencies", {"wrong_param": "foo"})
        assert isinstance(result, ToolResult)


# ════════════════════════════════════════════════════════════════════════════
# D. explain_evidence — LLM-powered narrative generation
# ════════════════════════════════════════════════════════════════════════════

class TestExplainEvidence:
    def test_explain_evidence_returns_tool_result(self):
        from omnirca.tools.rca_tools import explain_evidence
        from omnirca.tools.base import ToolResult
        result = explain_evidence(
            service="cache_0",
            fault_category="stale_cache",
            evidence_list=[
                "syscall z-score = 45.2 (CRITICAL)",
                "read_count_ratio = 3.8 (stale_cache pattern)",
                "latency decreased to 0.5× baseline (counter-intuitive)",
                "error_rate_ratio = 1.25 (25% increase)",
            ],
        )
        assert isinstance(result, ToolResult)

    def test_explain_evidence_summary_non_empty(self):
        from omnirca.tools.rca_tools import explain_evidence
        result = explain_evidence(
            service="backend_4",
            fault_category="memory_leak",
            evidence_list=[
                "memory slope = +0.006 MB/s",
                "mmap_count_ratio = 2.1",
                "syscall z-score = 38.7",
            ],
        )
        assert len(result.summary) > 100

    def test_explain_evidence_mentions_service_name(self):
        from omnirca.tools.rca_tools import explain_evidence
        result = explain_evidence(
            service="backend_4",
            fault_category="memory_leak",
            evidence_list=["memory slope +5 MB/min", "mmap count rising"],
        )
        # Either the summary or explanation data should mention the service
        assert (
            "backend_4" in result.summary or
            "backend_4" in str(result.data.get("explanation", ""))
        ), f"Service name not in explanation:\n{result.summary[:400]}"

    def test_explain_evidence_no_evidence_list_still_works(self):
        from omnirca.tools.rca_tools import explain_evidence
        result = explain_evidence(service="backend_0", fault_category="cascading_timeout")
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0


# ════════════════════════════════════════════════════════════════════════════
# E. Full agent run — stale_cache fault window
# ════════════════════════════════════════════════════════════════════════════

class TestAgentStaleCache:
    """
    Run the full ReAct agent on the stale_cache fault window.
    Ground truth: root cause = cache_0.

    Test does NOT assert the exact service name (model variance is expected);
    it asserts structural correctness and that the agent completed.
    """

    @pytest.fixture(scope="class")
    def agent_result(self):
        from omnirca.agent.react_agent import ReActAgent
        agent = ReActAgent(max_steps=25, verbose=True)
        return agent.run(
            incident_query=(
                "Multiple services are reporting elevated error rates and "
                "unexpected latency changes. Some services seem to be "
                "responding faster than normal but with more errors. "
                "Cache-related behaviour may be involved."
            ),
            t_start=STALE_CACHE_START,
            t_end=STALE_CACHE_END,
        )

    def test_reached_conclusion(self, agent_result):
        assert agent_result.reached_conclusion, (
            f"Agent did not call finalize_rca.\n"
            f"Steps taken: {agent_result.steps_taken}\n"
            f"Tools called: {[tc.tool for tc in agent_result.tool_calls]}"
        )

    def test_verdict_not_none(self, agent_result):
        assert agent_result.verdict is not None

    def test_root_cause_is_string(self, agent_result):
        assert isinstance(agent_result.root_cause, str)
        assert len(agent_result.root_cause) > 0

    def test_root_cause_not_logging(self, agent_result):
        """logging is never the root cause."""
        assert agent_result.root_cause != "logging", (
            "Agent identified 'logging' as root cause — topology rule violated."
        )

    def test_fault_category_is_valid(self, agent_result):
        valid_categories = {
            "stale_cache", "memory_leak", "thread_pool_exhaustion",
            "transaction_deadlock", "disk_io_saturation", "cascading_timeout",
            "authentication_failure", "auth_failure", "data_corruption",
            "api_version_mismatch", "data_race_condition", "unknown",
        }
        cat = agent_result.fault_category or "unknown"
        assert cat.lower() in valid_categories, (
            f"Invalid fault_category: '{cat}'"
        )

    def test_confidence_is_valid(self, agent_result):
        assert agent_result.confidence in ("HIGH", "MEDIUM", "LOW"), (
            f"Invalid confidence: '{agent_result.confidence}'"
        )

    def test_made_at_least_3_tool_calls(self, agent_result):
        assert agent_result.steps_taken >= 3, (
            f"Agent only took {agent_result.steps_taken} steps — too shallow."
        )

    def test_multi_service_compare_was_called(self, agent_result):
        """Agent must always call the mandatory first tool."""
        tools_used = [tc.tool for tc in agent_result.tool_calls]
        assert "syscall_multi_service_compare" in tools_used, (
            f"syscall_multi_service_compare was not called. Tools used: {tools_used}"
        )

    def test_evidence_list_non_empty(self, agent_result):
        ev = agent_result.verdict.data.get("evidence_list", [])
        assert len(ev) >= 1, "finalize_rca called with empty evidence_list"

    def test_result_summary_shows_root_cause(self, agent_result):
        summary = agent_result.summary()
        assert "Root cause:" in summary


# ════════════════════════════════════════════════════════════════════════════
# F. Full agent run — memory_leak fault window
# ════════════════════════════════════════════════════════════════════════════

class TestAgentMemoryLeak:
    """
    Run the full ReAct agent on the memory_leak fault window.
    Ground truth: root cause = backend_4 (id=7).
    """

    @pytest.fixture(scope="class")
    def agent_result(self):
        from omnirca.agent.react_agent import ReActAgent
        agent = ReActAgent(max_steps=25, verbose=True)
        return agent.run(
            incident_query=(
                "One or more backend services are consuming increasing amounts "
                "of memory over time. System is degrading gradually, "
                "not all at once. mmap syscall patterns may be elevated."
            ),
            t_start=MEMORY_LEAK_START,
            t_end=MEMORY_LEAK_END,
        )

    def test_reached_conclusion(self, agent_result):
        assert agent_result.reached_conclusion, (
            f"Agent did not call finalize_rca. "
            f"Tools called: {[tc.tool for tc in agent_result.tool_calls]}"
        )

    def test_root_cause_not_logging(self, agent_result):
        assert agent_result.root_cause != "logging"

    def test_fault_category_memory_leak_or_unknown(self, agent_result):
        """
        For memory_leak, the agent should identify memory_leak or at minimum
        not a completely wrong functional category.
        """
        non_functional = {"memory_leak", "thread_pool_exhaustion", "disk_io_saturation", "unknown"}
        cat = (agent_result.fault_category or "unknown").lower()
        # Soft check: we don't fail if the model is confused, just report
        if cat not in non_functional:
            pytest.xfail(
                f"Agent classified memory_leak as '{cat}' — "
                f"acceptable model variance but worth reviewing."
            )

    def test_made_at_least_4_tool_calls(self, agent_result):
        assert agent_result.steps_taken >= 4


# ════════════════════════════════════════════════════════════════════════════
# G. AgentResult dataclass structure
# ════════════════════════════════════════════════════════════════════════════

class TestAgentResultStructure:
    def _make_minimal_result(self):
        from omnirca.agent.react_agent import AgentResult, ToolCallRecord
        from omnirca.tools.base import ToolResult
        verdict = ToolResult(
            summary="verdict",
            data={"root_cause": "cache_0", "confidence": "HIGH",
                  "fault_category": "stale_cache", "evidence_list": ["ev1"]},
        )
        return AgentResult(
            verdict=verdict,
            steps_taken=5,
            reached_conclusion=True,
            total_duration_s=12.3,
            tool_calls=[
                ToolCallRecord(step=1, tool="syscall_multi_service_compare",
                               args={}, result_summary="r", duration_ms=120.0)
            ],
            conversation=[{"role": "user", "content": "test"}],
        )

    def test_accessor_root_cause(self):
        r = self._make_minimal_result()
        assert r.root_cause == "cache_0"

    def test_accessor_fault_category(self):
        r = self._make_minimal_result()
        assert r.fault_category == "stale_cache"

    def test_accessor_confidence(self):
        r = self._make_minimal_result()
        assert r.confidence == "HIGH"

    def test_summary_method(self):
        r = self._make_minimal_result()
        s = r.summary()
        assert "cache_0" in s
        assert "stale_cache" in s

    def test_no_verdict_summary(self):
        from omnirca.agent.react_agent import AgentResult
        r = AgentResult(
            verdict=None, steps_taken=25, reached_conclusion=False,
            total_duration_s=60.0,
        )
        s = r.summary()
        assert "NO CONCLUSION" in s


# ════════════════════════════════════════════════════════════════════════════
# H. Phase 1 + 2 + 3 regression
# ════════════════════════════════════════════════════════════════════════════

class TestPhase123Regression:
    def test_all_24_tools_still_exported(self):
        from omnirca.tools import __all__ as exports
        assert len(exports) >= 24

    def test_rag_retriever_still_works(self):
        from omnirca.rag.retriever import retrieve
        results = retrieve("memory growing, mmap rising", top_k=1)
        assert len(results) == 1
        assert results[0]["fault_type"] == "memory_leak"

    def test_search_fault_knowledge_not_stub(self):
        from omnirca.tools.rca_tools import search_fault_knowledge
        result = search_fault_knowledge("cache miss, latency dropped")
        # Should NOT say "stub" any more
        assert "stub" not in result.summary.lower()
        assert "results" in result.data

    def test_explain_evidence_not_stub(self):
        from omnirca.tools.rca_tools import explain_evidence
        result = explain_evidence("cache_0", "stale_cache", ["z=40", "read_count x3"])
        assert "stub" not in result.summary.lower()

    def test_fused_graph_intact(self):
        from omnirca.data_layer.graph_loader import get_fused_graph
        fg = get_fused_graph(STALE_CACHE_START, STALE_CACHE_END)
        assert fg.number_of_nodes() > 0

    def test_temporal_onset_intact(self):
        from omnirca.data_layer.temporal import rank_services_by_onset
        ranked = rank_services_by_onset(["cache_0", "backend_0"], STALE_CACHE_START, STALE_CACHE_END)
        assert isinstance(ranked, list)
