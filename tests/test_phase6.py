"""
Phase 6 validation test suite.

Test classes:
  TestSopLibraryImports       (4)  — package imports and class accessibility
  TestSopLibraryContent       (8)  — all known SOPs present, correct structure
  TestSopFairnessRule         (3)  — no numbered service names or fault IDs
  TestSopLibraryLookup        (6)  — per-category key_signals and steps validated
  TestCheckSopIntegration     (5)  — check_sop() delegates to sop_library
  TestAutoSopGeneratorStruct  (4)  — AutoSopGenerator API (no LLM)
  TestAutoSopGeneratorLive    (5)  — LLM-backed generation for novel categories
  TestPhase6Regression        (2)  — Phases 1–5 undamaged

Total: 37 tests
"""
import sys
import re
from pathlib import Path


import pytest

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

# Numbered service names that must never appear in SOP text (fairness rule)
_FORBIDDEN_SERVICE_PATTERNS = [
    r"\bcache_\d+\b",
    r"\bbackend_\d+\b",
    r"\bfrontend_\d+\b",
    r"\bdatabase_\d+\b",
    r"\bmessage_queue_\d+\b",
    r"\bfault_\d+\b",
    r"\bfault_id\b",
]

def _contains_forbidden(text: str) -> list[str]:
    """Return list of forbidden patterns found in text."""
    hits = []
    for pat in _FORBIDDEN_SERVICE_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            hits.append(pat)
    return hits


_KNOWN_FAULT_CATEGORIES = [
    "stale_cache",
    "memory_leak",
    "thread_pool_exhaustion",
    "transaction_deadlock",
    "disk_io_saturation",
    "cascading_timeout",
    "auth_failure",
    "data_corruption",
    "api_version_mismatch",
    "data_race_condition",
    "unknown",
]

_FOUR_GROUPS = {"data_integrity", "resource_exhaustion", "cascading_deadlock", "auth", "unknown"}


# ─────────────────────────────────────────────────────────────────────────────
# TestSopLibraryImports — package accessibility
# ─────────────────────────────────────────────────────────────────────────────
class TestSopLibraryImports:
    def test_sop_library_importable(self):
        from omnirca.sops.sop_library import SopLibrary
        assert SopLibrary is not None

    def test_sop_dataclass_importable(self):
        from omnirca.sops.sop_library import SOP
        assert SOP is not None

    def test_package_exports_sop_library(self):
        from omnirca.sops import SopLibrary, SOP
        assert SopLibrary is not None
        assert SOP is not None

    def test_auto_sop_generator_importable(self):
        from omnirca.sops import AutoSopGenerator
        assert AutoSopGenerator is not None


# ─────────────────────────────────────────────────────────────────────────────
# TestSopLibraryContent — all 11 known SOPs present with correct structure
# ─────────────────────────────────────────────────────────────────────────────
class TestSopLibraryContent:
    def test_all_known_categories_accessible(self):
        from omnirca.sops.sop_library import SopLibrary
        for cat in _KNOWN_FAULT_CATEGORIES:
            sop = SopLibrary.get(cat)
            assert sop is not None, f"Missing SOP for '{cat}'"

    def test_sop_has_all_required_fields(self):
        from omnirca.sops.sop_library import SopLibrary, SOP
        for cat in _KNOWN_FAULT_CATEGORIES:
            sop = SopLibrary.get(cat)
            assert isinstance(sop, SOP), f"{cat} not a SOP instance"
            assert sop.fault_category == cat
            assert len(sop.description) > 10, f"{cat}: description too short"
            assert len(sop.key_signals) > 0, f"{cat}: no key_signals"
            assert len(sop.diagnostic_steps) >= 3, f"{cat}: fewer than 3 steps"
            assert len(sop.recommended_tools) > 0, f"{cat}: no recommended_tools"
            assert len(sop.text) > 50, f"{cat}: text too short"

    def test_list_categories_returns_all_known(self):
        from omnirca.sops.sop_library import SopLibrary
        cats = SopLibrary.list_categories()
        for cat in _KNOWN_FAULT_CATEGORIES:
            assert cat in cats, f"'{cat}' missing from list_categories()"

    def test_category_groups_cover_four_plan_groups(self):
        from omnirca.sops.sop_library import SopLibrary
        groups = set(SopLibrary.list_groups())
        assert "data_integrity" in groups
        assert "resource_exhaustion" in groups
        assert "cascading_deadlock" in groups
        assert "auth" in groups

    def test_by_group_data_integrity_has_four_sops(self):
        from omnirca.sops.sop_library import SopLibrary
        di_sops = SopLibrary.by_group("data_integrity")
        assert len(di_sops) == 4, (
            f"Expected 4 data_integrity SOPs, got {len(di_sops)}: "
            f"{[s.fault_category for s in di_sops]}"
        )

    def test_by_group_resource_exhaustion_has_three_sops(self):
        from omnirca.sops.sop_library import SopLibrary
        re_sops = SopLibrary.by_group("resource_exhaustion")
        assert len(re_sops) == 3

    def test_sop_repr_contains_category(self):
        from omnirca.sops.sop_library import SopLibrary
        sop = SopLibrary.get("memory_leak")
        assert "memory_leak" in repr(sop)

    def test_sop_summary_is_non_empty(self):
        from omnirca.sops.sop_library import SopLibrary
        for cat in _KNOWN_FAULT_CATEGORIES:
            s = SopLibrary.get(cat).summary()
            assert len(s) > 15, f"{cat}: summary() too short: {s!r}"


# ─────────────────────────────────────────────────────────────────────────────
# TestSopFairnessRule — pattern-based only, no service names, no fault IDs
# ─────────────────────────────────────────────────────────────────────────────
class TestSopFairnessRule:
    def test_no_sop_text_contains_numbered_service_name(self):
        """SOP texts must not reference specific service instances like cache_0."""
        from omnirca.sops.sop_library import SopLibrary
        violations = []
        for cat in _KNOWN_FAULT_CATEGORIES:
            sop = SopLibrary.get(cat)
            hits = _contains_forbidden(sop.text)
            if hits:
                violations.append(f"  {cat}: matched patterns {hits}")
        assert not violations, "Fairness violations found:\n" + "\n".join(violations)

    def test_no_sop_key_signals_contain_service_names(self):
        from omnirca.sops.sop_library import SopLibrary
        for cat in _KNOWN_FAULT_CATEGORIES:
            sop = SopLibrary.get(cat)
            for sig in sop.key_signals:
                hits = _contains_forbidden(sig)
                assert not hits, f"{cat} key_signal contains forbidden pattern {hits}: {sig!r}"

    def test_no_sop_steps_contain_service_names(self):
        from omnirca.sops.sop_library import SopLibrary
        for cat in _KNOWN_FAULT_CATEGORIES:
            sop = SopLibrary.get(cat)
            for step in sop.diagnostic_steps:
                hits = _contains_forbidden(step)
                assert not hits, f"{cat} step contains forbidden pattern {hits}: {step!r}"


# ─────────────────────────────────────────────────────────────────────────────
# TestSopLibraryLookup — per-category content validation
# ─────────────────────────────────────────────────────────────────────────────
class TestSopLibraryLookup:
    def test_stale_cache_sop_mentions_read_count(self):
        from omnirca.sops.sop_library import SopLibrary
        sop = SopLibrary.get("stale_cache")
        combined = " ".join(sop.key_signals + sop.diagnostic_steps + [sop.text])
        assert "read_count" in combined.lower() or "read" in combined.lower()

    def test_memory_leak_sop_mentions_memory_slope(self):
        from omnirca.sops.sop_library import SopLibrary
        sop = SopLibrary.get("memory_leak")
        combined = " ".join(sop.diagnostic_steps + [sop.text])
        assert "detect_memory_slope" in combined or "memory_slope" in combined.lower()

    def test_cascading_timeout_sop_mentions_temporal_ordering(self):
        from omnirca.sops.sop_library import SopLibrary
        sop = SopLibrary.get("cascading_timeout")
        combined = " ".join(sop.diagnostic_steps + [sop.text])
        assert "detect_causal_order" in combined or "temporal" in combined.lower() or "earliest" in combined.lower()

    def test_auth_failure_sop_mentions_fan_in(self):
        from omnirca.sops.sop_library import SopLibrary
        sop = SopLibrary.get("auth_failure")
        combined = sop.text.lower()
        assert "fan-in" in combined or "fan_in" in combined or "callers" in combined

    def test_unknown_sop_has_finalize_rca_step(self):
        from omnirca.sops.sop_library import SopLibrary
        sop = SopLibrary.get("unknown")
        combined = " ".join(sop.diagnostic_steps)
        assert "finalize_rca" in combined

    def test_get_returns_none_for_truly_unknown_category(self):
        from omnirca.sops.sop_library import SopLibrary
        result = SopLibrary.get("this_category_does_not_exist_xyz123")
        assert result is None

    def test_is_known_true_for_builtin(self):
        from omnirca.sops.sop_library import SopLibrary
        assert SopLibrary.is_known("memory_leak") is True

    def test_is_known_false_for_novel_category(self):
        from omnirca.sops.sop_library import SopLibrary
        assert SopLibrary.is_known("network_partition") is False


# ─────────────────────────────────────────────────────────────────────────────
# TestCheckSopIntegration — check_sop() uses sop_library
# ─────────────────────────────────────────────────────────────────────────────
class TestCheckSopIntegration:
    def test_check_sop_known_category_returns_toolresult(self):
        from omnirca.tools.rca_tools import check_sop
        from omnirca.tools.base import ToolResult
        result = check_sop("memory_leak")
        assert isinstance(result, ToolResult)

    def test_check_sop_returns_category_in_data(self):
        from omnirca.tools.rca_tools import check_sop
        result = check_sop("cascading_timeout")
        assert result.data.get("category") == "cascading_timeout"

    def test_check_sop_result_has_diagnostic_steps(self):
        from omnirca.tools.rca_tools import check_sop
        result = check_sop("thread_pool_exhaustion")
        steps = result.data.get("diagnostic_steps")
        assert steps is not None and len(steps) >= 3

    def test_check_sop_result_has_recommended_tools(self):
        from omnirca.tools.rca_tools import check_sop
        result = check_sop("stale_cache")
        tools = result.data.get("recommended_tools")
        assert tools is not None and len(tools) >= 1

    def test_check_sop_not_auto_generated_for_known(self):
        from omnirca.tools.rca_tools import check_sop
        result = check_sop("auth_failure")
        assert result.data.get("is_auto_generated") is False


# ─────────────────────────────────────────────────────────────────────────────
# TestAutoSopGeneratorStruct — API shape tests (no LLM)
# ─────────────────────────────────────────────────────────────────────────────
class TestAutoSopGeneratorStruct:
    def test_generator_has_get_or_generate_method(self):
        from omnirca.sops.sop_generator import AutoSopGenerator
        assert callable(getattr(AutoSopGenerator, "get_or_generate", None))

    def test_generator_has_get_cached_method(self):
        from omnirca.sops.sop_generator import AutoSopGenerator
        assert callable(getattr(AutoSopGenerator, "get_cached", None))

    def test_generator_has_clear_cache_method(self):
        from omnirca.sops.sop_generator import AutoSopGenerator
        assert callable(getattr(AutoSopGenerator, "clear_cache", None))

    def test_get_cached_returns_none_for_unknown(self):
        from omnirca.sops.sop_generator import AutoSopGenerator
        AutoSopGenerator.clear_cache()
        result = AutoSopGenerator.get_cached("this_is_not_cached_xyz")
        assert result is None

    def test_builtin_category_returns_library_sop_not_generated(self):
        """get_or_generate on a known category must return the built-in SOP, not call LLM."""
        from omnirca.sops.sop_generator import AutoSopGenerator
        sop = AutoSopGenerator.get_or_generate("disk_io_saturation")
        assert sop is not None
        assert sop.is_auto_generated is False   # must be the built-in


# ─────────────────────────────────────────────────────────────────────────────
# TestAutoSopGeneratorLive — LLM-backed generation
# ─────────────────────────────────────────────────────────────────────────────
class TestAutoSopGeneratorLive:
    """
    Tests that require a live LLM call.  The novel category used here is
    'network_partition' — not present in any built-in SOP.
    """
    _NOVEL_CATEGORY = "network_partition"
    _SIGNALS = {
        "syscall_avg_duration_z": 8.4,
        "recv_count_ratio": 0.1,
        "send_count_ratio": 0.1,
        "socket_errors_ratio": 5.2,
    }

    @pytest.fixture(autouse=True)
    def clear_cache_before_test(self):
        from omnirca.sops.sop_generator import AutoSopGenerator
        AutoSopGenerator.clear_cache()
        yield
        AutoSopGenerator.clear_cache()

    def test_generate_returns_sop_for_novel_category(self):
        from omnirca.sops.sop_generator import AutoSopGenerator
        from omnirca.sops.sop_library import SOP
        sop = AutoSopGenerator.get_or_generate(
            self._NOVEL_CATEGORY, signals=self._SIGNALS
        )
        assert isinstance(sop, SOP)
        assert sop.fault_category == self._NOVEL_CATEGORY

    def test_generated_sop_is_auto_generated_flag(self):
        from omnirca.sops.sop_generator import AutoSopGenerator
        sop = AutoSopGenerator.get_or_generate(
            self._NOVEL_CATEGORY, signals=self._SIGNALS
        )
        assert sop.is_auto_generated is True

    def test_generated_sop_has_diagnostic_steps(self):
        from omnirca.sops.sop_generator import AutoSopGenerator
        sop = AutoSopGenerator.get_or_generate(
            self._NOVEL_CATEGORY, signals=self._SIGNALS
        )
        assert len(sop.diagnostic_steps) >= 3

    def test_generated_sop_is_cached_on_second_call(self):
        from omnirca.sops.sop_generator import AutoSopGenerator
        sop1 = AutoSopGenerator.get_or_generate(
            self._NOVEL_CATEGORY, signals=self._SIGNALS
        )
        sop2 = AutoSopGenerator.get_or_generate(
            self._NOVEL_CATEGORY, signals=self._SIGNALS
        )
        # Exact same object when retrieved from cache
        assert sop1 is sop2

    def test_generated_sop_no_numbered_service_names(self):
        """Auto-generated SOPs must obey the fairness rule."""
        from omnirca.sops.sop_generator import AutoSopGenerator
        sop = AutoSopGenerator.get_or_generate(
            self._NOVEL_CATEGORY, signals=self._SIGNALS
        )
        hits = _contains_forbidden(sop.text)
        assert not hits, f"Generated SOP contains forbidden patterns: {hits}"

    def test_check_sop_triggers_auto_generation_for_novel_category(self):
        """check_sop() with an unknown category should produce an auto-generated SOP."""
        from omnirca.tools.rca_tools import check_sop
        # Use a different novel name to avoid any residual cache
        result = check_sop("config_drift")
        assert result is not None
        assert result.data.get("is_auto_generated") is True


# ─────────────────────────────────────────────────────────────────────────────
# TestPhase6Regression — prior phases undamaged
# ─────────────────────────────────────────────────────────────────────────────
class TestPhase6Regression:
    def test_check_sop_still_works_for_all_known_categories(self):
        """Regression: ensure Phase 3 usage of check_sop() still returns valid text."""
        from omnirca.tools.rca_tools import check_sop
        for cat in _KNOWN_FAULT_CATEGORIES:
            result = check_sop(cat)
            assert len(result.summary) > 30, f"check_sop('{cat}') returned empty summary"
            assert "error" not in result.data, f"check_sop('{cat}') returned error: {result.data}"

    def test_fault_typer_still_imports_cleanly(self):
        from omnirca.agents.fault_typer import FaultTyper
        assert FaultTyper.AGENT_NAME == "FaultTyper"
        assert len(FaultTyper.TOOL_SCHEMAS) == 3
