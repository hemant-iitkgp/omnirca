"""
Phase 1 validation test suite.

Validates all 22 tools against Fault 0 (stale_cache, cache_0, id=14).
Known ground truth from assess.py:
  fault_id=0, category=stale_cache, root_cause=cache_0 (id=14)
  window: 2024-01-01 03:23:36 → 2024-01-01 03:37:35
  Expected: cache_0 has z_avg_duration ≈ 31.67, z_p99 ≈ 48.87
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
SKIP  = "~"
WIDTH = 60


def _header(title: str):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def _result(label: str, passed: bool, detail: str = ""):
    icon = PASS if passed else FAIL
    status = "PASS" if passed else "FAIL"
    print(f"  {icon} [{status}] {label:<55} {detail}")


def run_all():
    _header("OmniRCA Phase 1 — Validation Test Suite")

    failures = 0
    total    = 0

    # ── Import all tools ─────────────────────────────────────────────────────
    _header("0. Import & DataLoader warm-up")
    try:
        from omnirca.data_layer.loader import get_loader
        from omnirca.data_layer.graph_loader import get_arch_graph
        from omnirca.data_layer.kv_store import get_store
        from omnirca import tools
        loader = get_loader()
        _result("Import omnirca.tools (all 22 tools)", True)
        total += 1
    except Exception as e:
        _result("Import omnirca.tools", False, str(e))
        traceback.print_exc()
        print("\nFATAL: cannot import tools — aborting.")
        sys.exit(1)

    # ── DataLoader sanity ─────────────────────────────────────────────────────
    _header("1. DataLoader sanity checks")

    checks = [
        ("metrics rows",   len(loader.metrics)  == 3_456_000, f"{len(loader.metrics):,}"),
        ("logs rows",      len(loader.logs)      == 34_560,    f"{len(loader.logs):,}"),
        ("syscalls rows",  len(loader.syscalls)  == 57_600,    f"{len(loader.syscalls):,}"),
        ("traces rows",    len(loader.traces)    >= 5_000,     f"{len(loader.traces):,}"),
        ("20 services",    len(loader.service_names) == 20,    str(len(loader.service_names))),
        ("is_anomaly stripped (syscalls)",
                           "is_anomaly" not in loader.syscalls.columns, ""),
        ("is_anomaly stripped (metrics)",
                           "is_anomaly" not in loader.metrics.columns,  ""),
        ("traces has start_time col",
                           "start_time" in loader.traces.columns,         ""),
        ("resolve cache_0", loader.resolve_service("cache_0")[0] == 14,   ""),
        ("resolve by id 14", loader.resolve_service(14)[1] == "cache_0",  ""),
    ]
    for label, passed, detail in checks:
        total += 1
        if not passed:
            failures += 1
        _result(label, passed, detail)

    # ── Graph loader ──────────────────────────────────────────────────────────
    _header("2. Architecture graph loader")
    total += 1
    try:
        G = get_arch_graph()
        ok_nodes = G.number_of_nodes() == 20
        ok_edges = G.number_of_edges() == 68
        _result("graph loads (20 nodes, 68 edges)",
                ok_nodes and ok_edges,
                f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        if not (ok_nodes and ok_edges):
            failures += 1
    except Exception as e:
        _result("graph loads", False, str(e))
        failures += 1

    # ── KV Store ─────────────────────────────────────────────────────────────
    _header("3. KV Store")
    total += 1
    try:
        store = get_store()
        key = store.make_key("test", service="cache_0")
        store.save(key, {"value": 42})
        retrieved = store.get(key)
        ok = isinstance(retrieved, dict) and retrieved.get("value") == 42
        _result("KV store: save and retrieve", ok, key)
        if not ok:
            failures += 1
    except Exception as e:
        _result("KV store", False, str(e))
        failures += 1

    # ── Syscall tools (PRIMARY) ───────────────────────────────────────────────
    _header("4. Syscall tools — PRIMARY LOCALIZATION")

    # 4a query_syscalls
    total += 1
    try:
        r = tools.query_syscalls(FAULT_0_SVC, FAULT_0_START, FAULT_0_END)
        ok = len(r.data) > 0 and "avg_duration_us" in r.data.columns
        _result("query_syscalls(cache_0, fault_0_window)", ok,
                f"{len(r.data)} rows")
        if not ok: failures += 1
    except Exception as e:
        _result("query_syscalls", False, str(e)); failures += 1

    # 4b syscall_multi_service_compare — must rank cache_0 #1
    total += 1
    try:
        r = tools.syscall_multi_service_compare(FAULT_0_START, FAULT_0_END)
        print()
        print(r.summary)
        print()
        df = r.data
        top_svc    = df.iloc[0]["service_name"]
        top_z      = df.iloc[0]["z_avg_duration"]
        from omnirca.config import NEVER_ROOT_CAUSE
        # logging is in NEVER_ROOT_CAUSE so the top *eligible* service must be cache_0
        eligible = df[~df["service_name"].isin(NEVER_ROOT_CAUSE)]
        top_eligible_svc = eligible.iloc[0]["service_name"]
        top_eligible_z   = eligible.iloc[0]["z_avg_duration"]
        cache0_row = df[df["service_name"] == FAULT_0_SVC]
        cache0_z   = float(cache0_row["z_avg_duration"].iloc[0]) if not cache0_row.empty else 0.0
        ok = top_eligible_svc == FAULT_0_SVC and top_eligible_z > 20 and cache0_z > 20
        _result("syscall_multi_service_compare → cache_0 top eligible service", ok,
                f"top_eligible={top_eligible_svc} z={top_eligible_z:.2f} (logging excluded: z={top_z:.2f})")
        if not ok: failures += 1
    except Exception as e:
        _result("syscall_multi_service_compare", False, str(e))
        traceback.print_exc()
        failures += 1

    # 4c syscall_sub_channel_analysis
    total += 1
    try:
        r = tools.syscall_sub_channel_analysis(FAULT_0_SVC, FAULT_0_START, FAULT_0_END)
        print()
        print(r.summary)
        print()
        ok = "channel_stats" in r.data and len(r.data["channel_stats"]) > 0
        _result("syscall_sub_channel_analysis(cache_0)", ok,
                f"hint: {r.data.get('fault_hint', '')[:60]}")
        if not ok: failures += 1
    except Exception as e:
        _result("syscall_sub_channel_analysis", False, str(e)); failures += 1

    # ── Metric tools ─────────────────────────────────────────────────────────
    _header("5. Metric tools")

    for label, fn, args in [
        ("query_metrics(cache_0)", tools.query_metrics,
         (FAULT_0_SVC, FAULT_0_START, FAULT_0_END)),
        ("compute_anomaly_score(cache_0)", tools.compute_anomaly_score,
         (FAULT_0_SVC, FAULT_0_START, FAULT_0_END)),
        ("detect_memory_slope(cache_0)", tools.detect_memory_slope,
         (FAULT_0_SVC, FAULT_0_START, FAULT_0_END)),
    ]:
        total += 1
        try:
            r = fn(*args)
            ok = r is not None and len(r.summary) > 10
            _result(label, ok, r.summary.split("\n")[0])
            if not ok: failures += 1
        except Exception as e:
            _result(label, False, str(e)); failures += 1

    # Make sure compute_anomaly_score raises the score (not zero)
    total += 1
    try:
        r = tools.compute_anomaly_score(FAULT_0_SVC, FAULT_0_START, FAULT_0_END)
        score = r.data.get("total_score", 0)
        ok = score > 50   # syscall weights 4× and z≈31 should yield >100
        _result("compute_anomaly_score > 50 for cache_0", ok, f"score={score:.2f}")
        if not ok: failures += 1
    except Exception as e:
        _result("compute_anomaly_score threshold check", False, str(e)); failures += 1

    # ── Log tools ────────────────────────────────────────────────────────────
    _header("6. Log tools")

    for label, fn, args in [
        ("query_logs(cache_0, ERROR)", tools.query_logs,
         (FAULT_0_SVC, FAULT_0_START, FAULT_0_END, "ERROR")),
        ("detect_error_burst(cache_0)", tools.detect_error_burst,
         (FAULT_0_SVC, FAULT_0_START, FAULT_0_END)),
    ]:
        total += 1
        try:
            r = fn(*args)
            ok = r is not None
            _result(label, ok, r.summary.split("\n")[0])
            if not ok: failures += 1
        except Exception as e:
            _result(label, False, str(e)); failures += 1

    # ── Trace tools ─────────────────────────────────────────────────────────
    _header("7. Trace tools (partial coverage expected)")

    for label, fn, args in [
        ("query_traces(cache_0)", tools.query_traces,
         (FAULT_0_SVC, FAULT_0_START, FAULT_0_END)),
        ("trace_fan_out(cache_0)", tools.trace_fan_out,
         (FAULT_0_SVC, FAULT_0_START, FAULT_0_END)),
        ("compare_trace_latency(cache_0)", tools.compare_trace_latency,
         (FAULT_0_SVC, FAULT_0_START, FAULT_0_END)),
    ]:
        total += 1
        try:
            r = fn(*args)
            ok = r is not None   # may return empty — that's fine
            _result(label, ok, r.summary.split("\n")[0][:70])
            if not ok: failures += 1
        except Exception as e:
            _result(label, False, str(e)); failures += 1

    # ── Graph tools ──────────────────────────────────────────────────────────
    _header("8. Graph tools")

    total += 1
    try:
        r = tools.get_service_dependencies(FAULT_0_SVC)
        ok = "callers" in r.data and "callees" in r.data
        _result("get_service_dependencies(cache_0)", ok,
                f"callers={len(r.data['callers'])} callees={len(r.data['callees'])}")
        if not ok: failures += 1
    except Exception as e:
        _result("get_service_dependencies", False, str(e)); failures += 1

    total += 1
    try:
        r = tools.build_call_path("frontend_0", FAULT_0_SVC)
        ok = r is not None
        paths = r.data.get("paths", [])
        _result("build_call_path(frontend_0 → cache_0)", ok,
                f"{len(paths)} path(s)")
        if not ok: failures += 1
    except Exception as e:
        _result("build_call_path", False, str(e)); failures += 1

    total += 1
    try:
        r = tools.get_dynamic_graph(FAULT_0_START, FAULT_0_END)
        ok = r is not None
        _result("get_dynamic_graph(fault_0_window)", ok,
                r.summary.split("\n")[0])
        if not ok: failures += 1
    except Exception as e:
        _result("get_dynamic_graph", False, str(e)); failures += 1

    total += 1
    try:
        r = tools.get_propagation_candidates(FAULT_0_SVC, FAULT_0_START, FAULT_0_END)
        ok = r is not None
        _result("get_propagation_candidates(cache_0)", ok,
                r.summary.split("\n")[0])
        if not ok: failures += 1
    except Exception as e:
        _result("get_propagation_candidates", False, str(e)); failures += 1

    total += 1
    try:
        services = loader.service_names[:4]
        r = tools.cross_correlate_services(services, "avg_duration_us",
                                           FAULT_0_START, FAULT_0_END)
        ok = r is not None
        _result("cross_correlate_services(4 svcs, avg_duration_us)", ok,
                r.summary.split("\n")[0])
        if not ok: failures += 1
    except Exception as e:
        _result("cross_correlate_services", False, str(e)); failures += 1

    # ── RCA tools ────────────────────────────────────────────────────────────
    _header("9. RCA tools")

    total += 1
    try:
        r = tools.search_fault_knowledge("slow cache reads")
        ok = "stub" in r.summary.lower() or r is not None
        _result("search_fault_knowledge (stub returns gracefully)", ok)
        if not ok: failures += 1
    except Exception as e:
        _result("search_fault_knowledge", False, str(e)); failures += 1

    total += 1
    try:
        fake_signals = {
            "syscall_avg_duration_z": 31.7,
            "channel_stats": {
                "read":  {"count_ratio": 2.5, "errors_ratio": 1.0, "duration_z": 29.0},
                "write": {"count_ratio": 0.7, "errors_ratio": 1.0, "duration_z": 2.0},
                "fsync": {"count_ratio": 1.0, "errors_ratio": 1.0, "duration_z": 0.5},
                "mmap":  {"count_ratio": 1.0, "errors_ratio": 1.0, "duration_z": 0.5},
                "socket":{"count_ratio": 1.0, "errors_ratio": 1.0, "duration_z": 0.5},
                "send":  {"count_ratio": 0.9, "errors_ratio": 1.0, "duration_z": 0.5},
                "recv":  {"count_ratio": 1.0, "errors_ratio": 1.0, "duration_z": 0.5},
                "open":  {"count_ratio": 1.0, "errors_ratio": 1.0, "duration_z": 0.5},
                "close": {"count_ratio": 1.0, "errors_ratio": 1.0, "duration_z": 0.5},
            },
        }
        r = tools.classify_fault_pattern(fake_signals)
        predicted = r.data.get("top_category", "")
        ok = predicted == "stale_cache"   # high read, low write → stale_cache
        _result("classify_fault_pattern → stale_cache (for cache pattern)", ok,
                f"predicted={predicted}")
        if not ok: failures += 1
    except Exception as e:
        _result("classify_fault_pattern", False, str(e)); failures += 1

    total += 1
    try:
        r = tools.check_sop("stale_cache")
        ok = "stale_cache" in r.summary.lower() and "SOP" in r.summary
        _result("check_sop('stale_cache') returns SOP text", ok)
        if not ok: failures += 1
    except Exception as e:
        _result("check_sop", False, str(e)); failures += 1

    total += 1
    try:
        r = tools.explain_evidence("cache_0", "stale_cache")
        ok = r is not None and len(r.summary) > 20
        _result("explain_evidence (returns ToolResult with content)", ok)
        if not ok: failures += 1
    except Exception as e:
        _result("explain_evidence", False, str(e)); failures += 1

    total += 1
    try:
        r = tools.finalize_rca(
            root_cause        = "cache_0",
            confidence        = "HIGH",
            evidence_list     = ["cache_0 z=31.67 avg_duration_us",
                                  "read_count_ratio=2.5, write_count_ratio=0.7",
                                  "stale_cache sub-channel pattern confirmed"],
            fault_category    = "stale_cache",
            propagation_path  = ["cache_0"],
            affected_services = ["cache_0", "backend_0"],
            composite_score   = 134.5,
        )
        ok = "FINAL VERDICT" in r.summary and "cache_0" in r.summary
        _result("finalize_rca produces verdict report", ok)
        if not ok: failures += 1
    except Exception as e:
        _result("finalize_rca", False, str(e)); failures += 1

    # ── Tool count sanity ─────────────────────────────────────────────────────
    _header("10. Tool count — 24 tools exported (22 Phase 1 + 2 Phase 2 graph tools)")
    from omnirca.tools import __all__ as tool_names
    total += 1
    ok = len(tool_names) == 24
    _result(f"24 tools in omnirca.tools.__all__", ok, f"found {len(tool_names)}")
    if not ok: failures += 1

    # ── read_kv utility ───────────────────────────────────────────────────────
    _header("11. read_kv utility tool")
    total += 1
    try:
        from omnirca.data_layer.kv_store import get_store
        store = get_store()
        k = store.make_key("test_readkv", ts="xyz")
        store.save(k, {"answer": 99})
        r = tools.read_kv(f"kv://{k}")
        ok = r.data is not None and r.data.get("answer") == 99
        _result("read_kv retrieves KV data by URI", ok)
        if not ok: failures += 1
    except Exception as e:
        _result("read_kv", False, str(e)); failures += 1

    # ── Final summary ─────────────────────────────────────────────────────────
    _header("Results")
    passed = total - failures
    print(f"\n  Passed : {passed} / {total}")
    print(f"  Failed : {failures} / {total}")
    if failures == 0:
        print("\n  🎉  ALL TESTS PASSED — Phase 1 milestone achieved.\n")
    else:
        print(f"\n  ⚠   {failures} test(s) failed — see details above.\n")

    return failures


if __name__ == "__main__":
    sys.exit(run_all())
