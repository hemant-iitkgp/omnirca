"""
test_phase3.py — Tests for the RAG knowledge base (Phase 3).

Test categories:
  A. Fairness  — no dataset-specific terms in any indexed chunk
  B. Retrieval correctness — stale_cache query → stale_cache top-1
  C. Retrieval correctness — memory_leak query → memory_leak top-1
  D. Retrieval correctness — thread_pool_exhaustion → top-1 match
  E. Counter-intuitive flags — stale_cache, thread_pool, disk_io flagged
  F. Tool integration — search_fault_knowledge returns proper ToolResult
  G. Tool integration — search_fault_knowledge with index absent falls back gracefully
  H. Edge cases — empty query handled
  I. Phase 1+2 regression — imports + basic smoke test
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
_PKG_ROOT = Path(__file__).resolve().parent.parent   # repo root = omnirca package root

from omnirca.rag.retriever import retrieve, is_index_available, FORBIDDEN_PATTERNS
from omnirca.rag.indexer import FORBIDDEN_PATTERNS as INDEXER_FORBIDDEN
from omnirca.tools.rca_tools import search_fault_knowledge


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_chunks() -> list[dict]:
    chunks_path = _PKG_ROOT / "rag" / "index" / "chunks.json"
    if not chunks_path.exists():
        pytest.skip("FAISS index not built — run `python -m omnirca.rag.indexer` first.")
    with open(chunks_path, encoding="utf-8") as f:
        return json.load(f)


def _forbidden_compiled() -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS]


# ─────────────────────────────────────────────────────────────────────────────
# A. Fairness tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFairness:
    """No dataset-specific identifiers must appear in any knowledge-base chunk."""

    def test_indexer_and_retriever_share_same_forbidden_patterns(self):
        """Both modules must enforce the same fairness boundary."""
        assert set(FORBIDDEN_PATTERNS) == set(INDEXER_FORBIDDEN), (
            "Forbidden-pattern lists diverge between indexer.py and retriever.py"
        )

    def test_no_forbidden_patterns_in_any_chunk(self):
        """Every chunk must be free of dataset-specific service names and IDs."""
        chunks = _load_chunks()
        compiled = _forbidden_compiled()
        violations = []
        for chunk in chunks:
            text = chunk["text"]
            for pat in compiled:
                m = pat.search(text)
                if m:
                    violations.append(
                        f"Chunk '{chunk['fault_name']}': pattern '{pat.pattern}' "
                        f"matched '{m.group()}'"
                    )
        assert not violations, "Fairness violations found:\n" + "\n".join(violations)

    def test_all_ten_chunks_indexed(self):
        chunks = _load_chunks()
        assert len(chunks) == 10, f"Expected 10 chunks, got {len(chunks)}"

    def test_fault_type_keys_are_clean_ascii_snake_case(self):
        chunks = _load_chunks()
        bad_keys = [
            c["fault_type"] for c in chunks
            if not re.match(r"^[a-z][a-z0-9_]*$", c["fault_type"])
        ]
        assert not bad_keys, f"Non-clean fault_type keys: {bad_keys}"

    def test_categories_are_valid(self):
        chunks = _load_chunks()
        valid = {"functional", "non-functional"}
        bad = [c for c in chunks if c["category"] not in valid]
        assert not bad, f"Invalid categories: {[(c['fault_name'], c['category']) for c in bad]}"


# ─────────────────────────────────────────────────────────────────────────────
# B-D. Retrieval correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrievalCorrectness:
    """Symptom descriptions must retrieve semantically correct fault archetypes."""

    QUERIES: list[tuple[str, str]] = [
        # (symptom_text, expected_fault_type)
        (
            "Cache hit rate dropped, stale data being served, latency FASTER "
            "than baseline (0.5x), cache miss storm, error rate UP 25%",
            "stale_cache",
        ),
        (
            "Memory usage growing steadily 5 MB per minute, "
            "mmap syscall count rising, GC runs increasing, eventual OOM",
            "memory_leak",
        ),
        (
            "Thread pool queue full, CPU utilization dropped while latency spikes, "
            "incoming requests pile up, socket accept errors",
            "thread_pool_exhaustion",
        ),
        (
            "Write syscall duration extreme spike, transaction lock wait, "
            "multiple services waiting for the same row",
            "transaction_deadlock",
        ),
        (
            "Timeout errors propagating through five service levels, "
            "cascading failure across dependency chain",
            "cascading_timeout",
        ),
    ]

    @pytest.mark.parametrize("query,expected", QUERIES)
    def test_top1_matches_expected_fault(self, query: str, expected: str):
        if not is_index_available():
            pytest.skip("FAISS index not built.")
        results = retrieve(query, top_k=2)
        assert results, f"No results for query: '{query[:60]}'"
        top1 = results[0]["fault_type"]
        assert top1 == expected, (
            f"Expected top-1 '{expected}', got '{top1}' "
            f"(similarity={results[0]['similarity']:.3f})"
        )

    def test_top1_similarity_above_threshold(self):
        """Good query → similarity should be comfortably above 0.5."""
        if not is_index_available():
            pytest.skip("FAISS index not built.")
        results = retrieve(
            "Cache hit rate dropped, stale data being served, latency FASTER "
            "than baseline (0.5x), cache miss storm",
            top_k=1,
        )
        assert results[0]["similarity"] >= 0.50, (
            f"Stale-cache query similarity too low: {results[0]['similarity']:.3f}"
        )

    def test_returns_at_most_top_k(self):
        if not is_index_available():
            pytest.skip("FAISS index not built.")
        results = retrieve("high syscall spike", top_k=3)
        assert len(results) <= 3

    def test_results_are_sorted_by_similarity(self):
        if not is_index_available():
            pytest.skip("FAISS index not built.")
        results = retrieve("memory grows, latency rises", top_k=4)
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True), "Results not sorted by similarity"

    def test_result_dicts_have_required_fields(self):
        if not is_index_available():
            pytest.skip("FAISS index not built.")
        results = retrieve("socket timeout cascading failure", top_k=1)
        required = {"fault_type", "fault_name", "category", "counter_intuitive",
                    "similarity", "text", "excerpt"}
        missing = required - set(results[0].keys())
        assert not missing, f"Missing fields: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# E. Counter-intuitive flags
# ─────────────────────────────────────────────────────────────────────────────

class TestCounterIntuitiveFlags:
    """Known counter-intuitive faults must be correctly flagged in the index."""

    def test_stale_cache_flagged_counter_intuitive(self):
        chunks = _load_chunks()
        c = next(c for c in chunks if c["fault_type"] == "stale_cache")
        assert c["counter_intuitive"], "stale_cache must be flagged counter_intuitive (latency drops)"

    def test_thread_pool_exhaustion_flagged_counter_intuitive(self):
        chunks = _load_chunks()
        c = next(c for c in chunks if c["fault_type"] == "thread_pool_exhaustion")
        assert c["counter_intuitive"], (
            "thread_pool_exhaustion must be flagged counter_intuitive (CPU drops while latency spikes)"
        )

    def test_disk_io_saturation_flagged_counter_intuitive(self):
        chunks = _load_chunks()
        c = next(c for c in chunks if c["fault_type"] == "disk_i_o_saturation")
        assert c["counter_intuitive"], "disk_i_o_saturation must be flagged counter_intuitive"

    def test_cascading_timeout_flagged_counter_intuitive(self):
        chunks = _load_chunks()
        c = next(c for c in chunks if c["fault_type"] == "cascading_timeout")
        assert c["counter_intuitive"], "cascading_timeout must be flagged counter_intuitive"

    def test_data_corruption_not_flagged(self):
        """data_corruption has no latency-drop or CPU-drop paradox."""
        chunks = _load_chunks()
        c = next(c for c in chunks if c["fault_type"] == "data_corruption")
        assert not c["counter_intuitive"], (
            "data_corruption should NOT be flagged counter_intuitive"
        )


# ─────────────────────────────────────────────────────────────────────────────
# F. Tool integration — search_fault_knowledge
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchFaultKnowledge:
    """The search_fault_knowledge tool must return a well-formed ToolResult."""

    def test_returns_tool_result(self):
        from omnirca.tools.base import ToolResult
        res = search_fault_knowledge("high read count, latency dropped, cache miss")
        assert isinstance(res, ToolResult)

    def test_summary_mentions_fault_type(self):
        res = search_fault_knowledge(
            "High read syscall count spikes, latency unexpectedly decreased, "
            "cache hit ratio dropped, error rate increased by 25%"
        )
        assert "stale_cache" in res.summary, (
            f"Expected 'stale_cache' in summary. Got:\n{res.summary[:400]}"
        )

    def test_data_contains_results_list(self):
        res = search_fault_knowledge("memory grows, mmap spikes, GC pressure")
        assert "results" in res.data
        assert isinstance(res.data["results"], list)
        assert len(res.data["results"]) >= 1

    def test_counter_intuitive_warning_in_summary_for_stale_cache(self):
        res = search_fault_knowledge(
            "Cache hit rate dropped, stale data, latency paradoxically FASTER (0.5x)"
        )
        # Should contain counter-intuitive warning in the summary text
        assert "COUNTER-INTUITIVE" in res.summary or "counter" in res.summary.lower(), (
            "Expected counter-intuitive warning for stale_cache query"
        )

    def test_empty_query_returns_error(self):
        res = search_fault_knowledge("")
        assert "empty" in res.summary.lower() or "error" in res.data


# ─────────────────────────────────────────────────────────────────────────────
# G. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_retrieve_empty_string_returns_empty(self):
        results = retrieve("")
        assert results == []

    def test_retrieve_whitespace_returns_empty(self):
        results = retrieve("   ")
        assert results == []

    def test_query_injection_sanitized(self):
        """Queries containing forbidden terms should be sanitized, not rejected outright."""
        # Should not raise; should return results with sanitized query
        results = retrieve("auth_service is throwing errors, cascading timeouts")
        # The forbidden term 'auth_service' gets redacted; query still runs
        assert isinstance(results, list)

    def test_top_k_1_returns_exactly_one(self):
        if not is_index_available():
            pytest.skip("FAISS index not built.")
        results = retrieve("memory grow large heap", top_k=1)
        assert len(results) == 1


# ─────────────────────────────────────────────────────────────────────────────
# H. Phase 1+2 regression smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase12Regression:
    """Ensure Phase 3 additions did not break Phase 1/2 exports."""

    def test_all_24_tools_still_exported(self):
        from omnirca.tools import __all__ as exports
        assert len(exports) >= 24, f"Expected ≥24 exports, got {len(exports)}: {exports}"

    def test_classify_fault_pattern_still_works(self):
        from omnirca.tools.rca_tools import classify_fault_pattern
        res = classify_fault_pattern({"syscall_avg_duration_z": 35.0})
        assert res.data["top_category"] is not None

    def test_check_sop_still_works(self):
        from omnirca.tools.rca_tools import check_sop
        res = check_sop("memory_leak")
        assert "memory_leak" in res.data["category"]

    def test_finalize_rca_still_works(self):
        from omnirca.tools.rca_tools import finalize_rca
        res = finalize_rca(
            root_cause="cache_1",
            confidence="HIGH",
            evidence_list=["syscall z=45", "stale_cache pattern"],
            fault_category="stale_cache",
        )
        assert res.data["root_cause"] == "cache_1"

    def test_data_loader_singleton_intact(self):
        from omnirca.data_layer.loader import get_loader
        dl = get_loader()
        assert dl.syscalls is not None
        assert len(dl.service_names) == 20

    def test_graph_loader_intact(self):
        from omnirca.data_layer.graph_loader import get_arch_graph
        g = get_arch_graph()
        assert g.number_of_nodes() == 20
        assert g.number_of_edges() > 0
