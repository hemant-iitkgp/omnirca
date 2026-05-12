"""
Phase 2 validation test suite.

Tests the graph fusion engine and temporal onset detection against Fault 0:
  fault_id=0, category=stale_cache, root_cause=cache_0 (id=14)
  window: 2024-01-01 03:23:36 → 2024-01-01 03:37:35

Key Phase 2 milestones:
  M1 — build_fused_graph returns a valid DiGraph with ≥68 edges
  M2 — detect_anomaly_onset(cache_0) returns a timestamp inside the fault window
  M3 — detect_causal_order filters NEVER_ROOT_CAUSE and ranks detected services first
  M4 — get_propagation_candidates uses fused graph (verifies new fields in output)
  M5 — get_fused_graph_summary reports static/dynamic/both edge breakdown
  M6 — build_call_path with fused graph returns paths
  M7 — Phase 1 test imports still work (no regressions)
"""
import sys
import traceback
from pathlib import Path


import pandas as pd

FAULT_0_START = "2024-01-01 03:23:36"
FAULT_0_END   = "2024-01-01 03:37:35"
FAULT_0_SVC   = "cache_0"

PASS  = "✓"
FAIL  = "✗"
WIDTH = 60


def _header(title: str):
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")


def _result(label: str, passed: bool, detail: str = ""):
    icon   = PASS if passed else FAIL
    status = "PASS" if passed else "FAIL"
    print(f"  {icon} [{status}] {label:<57} {detail}")


def run_all():
    _header("OmniRCA Phase 2 — Validation Test Suite")

    failures = 0
    total    = 0

    # ── Imports ───────────────────────────────────────────────────────────────
    _header("0. Import Phase 2 modules")
    try:
        from omnirca.data_layer.loader      import get_loader
        from omnirca.data_layer.graph_loader import get_arch_graph, build_fused_graph, get_fused_graph
        from omnirca.data_layer.temporal     import detect_anomaly_onset, rank_services_by_onset
        from omnirca import tools
        loader = get_loader()
        total += 1
        _result("Import data_layer.temporal", True)
        total += 1
        _result("Import data_layer.graph_loader (fused)", True)
    except Exception as e:
        _result("Import Phase 2 modules", False, str(e))
        traceback.print_exc()
        print("\nFATAL: cannot import Phase 2 modules — aborting.")
        sys.exit(1)

    # ── tool count ────────────────────────────────────────────────────────────
    _header("1. Tool count (should be 24)")
    total += 1
    count = len(tools.__all__)
    ok    = count == 24
    if not ok:
        failures += 1
    _result(f"tools.__all__ has 24 entries", ok, f"found {count}")

    # ── M1: build_fused_graph ─────────────────────────────────────────────────
    _header("2. M1 — build_fused_graph structure")

    total += 1
    try:
        G = build_fused_graph(FAULT_0_START, FAULT_0_END)
        ok = G is not None and G.number_of_nodes() == 20
        _result("fused graph has 20 nodes", ok, f"{G.number_of_nodes()} nodes")
        if not ok:
            failures += 1
    except Exception as e:
        _result("build_fused_graph executes", False, str(e))
        traceback.print_exc()
        failures += 1
        G = None

    total += 1
    try:
        ok = G is not None and G.number_of_edges() >= 68
        _result("fused graph has ≥68 edges (all static + any dynamic)",
                ok, f"{G.number_of_edges()} edges")
        if not ok:
            failures += 1
    except Exception as e:
        _result("fused edge count ≥68", False, str(e))
        failures += 1

    total += 1
    try:
        # All edges should have fused_weight attribute
        sample = list(G.edges(data=True))[:5]
        ok = all("fused_weight" in d for _, _, d in sample)
        _result("edges carry fused_weight attribute", ok)
        if not ok:
            failures += 1
    except Exception as e:
        _result("fused_weight attribute present", False, str(e))
        failures += 1

    total += 1
    try:
        # cache_0 should appear as a node (service-name graph)
        ok = FAULT_0_SVC in G.nodes()
        _result("fused graph uses service-name nodes (cache_0 present)", ok)
        if not ok:
            failures += 1
    except Exception as e:
        _result("service-name nodes", False, str(e))
        failures += 1

    total += 1
    try:
        # Static-only edges should have fused_weight = 0.5
        static_only = [(u, v, d) for u, v, d in G.edges(data=True)
                       if d.get("static_weight", 0) > 0 and d.get("dynamic_weight", 0) == 0]
        ok = len(static_only) > 0 and all(
            abs(d["fused_weight"] - 0.5) < 0.01 for _, _, d in static_only[:10]
        )
        _result("static-only edges have fused_weight ≈ 0.5", ok,
                f"{len(static_only)} static-only edges")
        if not ok:
            failures += 1
    except Exception as e:
        _result("static-only fused_weight", False, str(e))
        failures += 1

    total += 1
    try:
        # get_fused_graph returns same object on second call (cached)
        G2  = get_fused_graph(FAULT_0_START, FAULT_0_END)
        ok  = G2 is not None and G2.number_of_edges() == G.number_of_edges()
        _result("get_fused_graph returns cached object", ok)
        if not ok:
            failures += 1
    except Exception as e:
        _result("get_fused_graph caching", False, str(e))
        failures += 1

    # ── M2: detect_anomaly_onset ──────────────────────────────────────────────
    _header("3. M2 — detect_anomaly_onset")

    total += 1
    try:
        onset = detect_anomaly_onset(FAULT_0_SVC, FAULT_0_START, FAULT_0_END)
        ok    = onset is not None
        detail = str(onset) if onset else "None"
        _result(f"detect_anomaly_onset(cache_0) is not None", ok, detail)
        if not ok:
            failures += 1
    except Exception as e:
        _result("detect_anomaly_onset(cache_0)", False, str(e))
        traceback.print_exc()
        failures += 1
        onset = None

    total += 1
    try:
        ok = (onset is not None and
              pd.Timestamp(FAULT_0_START) <= onset <= pd.Timestamp(FAULT_0_END))
        _result("onset timestamp is within fault window", ok,
                str(onset) if onset else "N/A")
        if not ok:
            failures += 1
    except Exception as e:
        _result("onset within fault window", False, str(e))
        failures += 1

    total += 1
    try:
        # A normal window should return None (or very late onset)
        # Use a random non-fault window (05:00–05:15 on day 1)
        normal_onset = detect_anomaly_onset(FAULT_0_SVC,
                                            "2024-01-01 05:00:00",
                                            "2024-01-01 05:15:00")
        ok = normal_onset is None
        _result("detect_anomaly_onset returns None in normal window", ok,
                str(normal_onset) if normal_onset else "None ✓")
        if not ok:
            failures += 1
    except Exception as e:
        _result("normal window returns None", False, str(e))
        failures += 1

    total += 1
    try:
        onset_auth = detect_anomaly_onset("auth_service", FAULT_0_START, FAULT_0_END)
        ok = onset_auth is not None
        _result("auth_service also anomalous in fault window", ok,
                str(onset_auth) if onset_auth else "not detected")
        if not ok:
            failures += 1
    except Exception as e:
        _result("detect_anomaly_onset(auth_service)", False, str(e))
        failures += 1

    # ── M3: detect_causal_order ───────────────────────────────────────────────
    _header("4. M3 — detect_causal_order")

    test_svcs = ["cache_0", "logging", "auth_service", "backend_0", "backend_5"]

    total += 1
    try:
        r   = tools.detect_causal_order(test_svcs, FAULT_0_START, FAULT_0_END)
        ok  = r is not None and isinstance(r.data, list)
        _result("detect_causal_order returns ToolResult with list data", ok,
                r.summary.split("\n")[0])
        if not ok:
            failures += 1
    except Exception as e:
        _result("detect_causal_order executes", False, str(e))
        traceback.print_exc()
        failures += 1
        r = None

    total += 1
    try:
        # logging must be excluded (NEVER_ROOT_CAUSE)
        names_in_result = [item["service"] for item in r.data] if r else []
        ok = "logging" not in names_in_result
        _result("logging excluded from detect_causal_order", ok,
                f"services: {names_in_result}")
        if not ok:
            failures += 1
    except Exception as e:
        _result("logging excluded", False, str(e))
        failures += 1

    total += 1
    try:
        detected = [item for item in r.data if item["detected"]] if r else []
        ok = len(detected) > 0
        _result("at least one service has detected onset", ok,
                f"{len(detected)} detected")
        if not ok:
            failures += 1
    except Exception as e:
        _result("some services detected", False, str(e))
        failures += 1

    total += 1
    try:
        # rank_services_by_onset raw API
        ranked = rank_services_by_onset(
            [FAULT_0_SVC, "backend_0", "backend_5"],
            FAULT_0_START, FAULT_0_END,
        )
        cache0_entry = next((r for r in ranked if r["service"] == FAULT_0_SVC), None)
        ok = cache0_entry is not None and cache0_entry["detected"]
        _result("cache_0 onset detected in rank_services_by_onset", ok,
                f"lead={cache0_entry['lead_minutes']:.1f}min" if cache0_entry else "not found")
        if not ok:
            failures += 1
    except Exception as e:
        _result("rank_services_by_onset API", False, str(e))
        traceback.print_exc()
        failures += 1

    # ── M4: get_propagation_candidates (fused) ────────────────────────────────
    _header("5. M4 — get_propagation_candidates (fused + temporal)")

    total += 1
    try:
        r = tools.get_propagation_candidates(FAULT_0_SVC, FAULT_0_START, FAULT_0_END)
        ok = r is not None
        _result("get_propagation_candidates executes without error", ok,
                r.summary.split("\n")[0])
        if not ok:
            failures += 1
    except Exception as e:
        _result("get_propagation_candidates executes", False, str(e))
        traceback.print_exc()
        failures += 1
        r = None

    total += 1
    try:
        # New Phase 2 output must include path_prob and temporal_bonus fields
        has_new_fields = (
            r is not None and isinstance(r.data, list) and
            (len(r.data) == 0 or
             ("path_prob" in r.data[0] and "temporal_bonus" in r.data[0]))
        )
        _result("output includes path_prob + temporal_bonus fields", has_new_fields,
                f"{len(r.data)} candidates" if r else "no result")
        if not has_new_fields:
            failures += 1
    except Exception as e:
        _result("Phase 2 output fields", False, str(e))
        failures += 1

    total += 1
    try:
        # No candidate should be in NEVER_ROOT_CAUSE
        from omnirca.config import NEVER_ROOT_CAUSE
        bad = [c["service"] for c in r.data if c["service"] in NEVER_ROOT_CAUSE] if r else []
        ok  = len(bad) == 0
        _result("no NEVER_ROOT_CAUSE service in candidates", ok,
                f"bad: {bad}" if bad else "clean")
        if not ok:
            failures += 1
    except Exception as e:
        _result("NEVER_ROOT_CAUSE exclusion", False, str(e))
        failures += 1

    # ── M5: get_fused_graph_summary ───────────────────────────────────────────
    _header("6. M5 — get_fused_graph_summary")

    total += 1
    try:
        r = tools.get_fused_graph_summary(FAULT_0_START, FAULT_0_END)
        ok = r is not None and "Nodes: 20" in r.summary
        _result("get_fused_graph_summary reports 20 nodes", ok,
                r.summary.split("\n")[1] if r else "")
        if not ok:
            failures += 1
    except Exception as e:
        _result("get_fused_graph_summary executes", False, str(e))
        traceback.print_exc()
        failures += 1

    total += 1
    try:
        ok = r is not None and r.data.get("edge_count", 0) >= 68
        _result("fused summary reports ≥68 edges", ok,
                f"edge_count={r.data.get('edge_count')}" if r else "")
        if not ok:
            failures += 1
    except Exception as e:
        _result("fused summary edge_count ≥68", False, str(e))
        failures += 1

    # ── M6: build_call_path with fused graph ─────────────────────────────────
    _header("7. M6 — build_call_path with fused graph")

    total += 1
    try:
        r = tools.build_call_path("frontend_0", FAULT_0_SVC,
                                  FAULT_0_START, FAULT_0_END)
        ok = r is not None
        _result("build_call_path(frontend_0→cache_0, fused) executes", ok,
                r.summary.split("\n")[0])
        if not ok:
            failures += 1
    except Exception as e:
        _result("build_call_path fused executes", False, str(e))
        traceback.print_exc()
        failures += 1

    total += 1
    try:
        r_static = tools.build_call_path("frontend_0", FAULT_0_SVC)
        ok = r_static is not None
        _result("build_call_path(frontend_0→cache_0, static) still works", ok,
                r_static.summary.split("\n")[0])
        if not ok:
            failures += 1
    except Exception as e:
        _result("build_call_path static (backward compat)", False, str(e))
        traceback.print_exc()
        failures += 1

    # ── M7: Phase 1 regression ────────────────────────────────────────────────
    _header("8. M7 — Phase 1 regression (key tools)")

    for tool_name, args in [
        ("query_syscalls",            (FAULT_0_SVC, FAULT_0_START, FAULT_0_END)),
        ("compute_anomaly_score",     (FAULT_0_SVC, FAULT_0_START, FAULT_0_END)),
        ("get_service_dependencies",  (FAULT_0_SVC,)),
        ("get_dynamic_graph",         (FAULT_0_START, FAULT_0_END)),
    ]:
        total += 1
        try:
            fn = getattr(tools, tool_name)
            res = fn(*args)
            ok  = res is not None
            _result(f"{tool_name} still works (Phase 1 compat)", ok)
            if not ok:
                failures += 1
        except Exception as e:
            _result(f"{tool_name} Phase 1 compat", False, str(e)[:60])
            failures += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    _header(f"RESULTS: {total - failures}/{total} passed")
    if failures == 0:
        print("  ✓ All Phase 2 tests PASSED.\n")
    else:
        print(f"  ✗ {failures} test(s) FAILED.\n")

    return failures


if __name__ == "__main__":
    sys.exit(run_all())
