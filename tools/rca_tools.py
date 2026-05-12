"""
RCA tools — fault classification, SOPs, and final report generation.

Phase 3 status:
  - classify_fault_pattern : FULL implementation (heuristic rules, no LLM needed).
  - check_sop             : FULL — delegates to SopLibrary (Phase 6); auto-generates
                            SOPs for novel categories via AutoSopGenerator.
  - finalize_rca          : FULL implementation (formats the final verdict dict).
  - search_fault_knowledge: FULL — RAG-indexed FAULT_ENCYCLOPEDIA (Phase 3).
  - explain_evidence      : FULL — LLM-backed (Phase 4).

Phase 6 change:
  check_sop() now delegates to omnirca.sops.sop_library.SopLibrary.
  The _SOPS dict is kept as a legacy fallback only; sop_library is the
  authoritative source.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
import json
import textwrap

from .base import ToolResult
from omnirca.rag.retriever import retrieve as _rag_retrieve, is_index_available as _rag_ready

# Phase 6 — lazy imports to avoid circular imports at module load time
def _get_sop_library():
    from omnirca.sops.sop_library import SopLibrary
    return SopLibrary

def _get_auto_sop_generator():
    from omnirca.sops.sop_generator import AutoSopGenerator
    return AutoSopGenerator

# Lazy import for LLM client — avoids import cycles if rca_tools is imported
# before the agent package is initialised.
def _get_llm_simple_chat():
    from omnirca.llm_client import simple_chat
    return simple_chat

# ─────────────────────────────────────────────────────────────────────────────
# Standard Operating Procedures (SOPs) — embedded from plan §2
# Four categories: functional, non-functional × two sub-types each.
# ─────────────────────────────────────────────────────────────────────────────

_SOPS: Dict[str, str] = {

    "stale_cache": textwrap.dedent("""\
        SOP: stale_cache
        ────────────────
        Symptom: cache miss storm → slow reads from upstream data store.
        Sub-channel signature: read_count HIGH, write_count LOW, open_count stable.
        Suggested steps:
          1. Confirm with syscall_sub_channel_analysis: read_count_ratio > 2, write_count_ratio < 0.8.
          2. Check propagation: downstream services with high latency_ms z-score (cache callers).
          3. Check metrics: memory_mb stable or declining (no leak).
          4. Check logs: WARNING entries about cache-miss or TTL expiry.
          5. Root cause confirmation: service is a cache_* service with read-dominated syscall pattern.
          6. Remediation (out-of-scope for agent): invalidate/repopulate cache, increase TTL, or add
             read-through fallback.
    """),

    "memory_leak": textwrap.dedent("""\
        SOP: memory_leak
        ────────────────
        Symptom: gradual memory growth → OOM or GC pressure → latency degradation.
        Sub-channel signature: mmap_count increasing over window (count_ratio > 1.3).
        Suggested steps:
          1. Confirm with detect_memory_slope: slope_mb_per_min > 0 sustained.
          2. Confirm with syscall_sub_channel_analysis: mmap_count_ratio > 1.3.
          3. Check propagation: is this service also a dependency of affected high-latency services?
          4. Check logs: OOM or GC log entries (WARNING/ERROR level).
          5. Check metrics: cpu_percent elevated (GC overhead).
          6. Root cause confirmation: memory_slope verdict is LEAK CANDIDATE and mmap elevated.
          7. Remediation (out-of-scope): restart service, tune heap size, fix leak in application code.
    """),

    "thread_pool_exhaustion": textwrap.dedent("""\
        SOP: thread_pool_exhaustion
        ────────────────────────────
        Symptom: all worker threads blocked → queuing delays → recv spikes, send drops.
        Sub-channel signature: recv_count × HIGH, socket_errors × HIGH, send_count × LOW.
        Suggested steps:
          1. Confirm with syscall_sub_channel_analysis: recv_count_ratio > 2,
             socket errors_ratio > 2, send_count_ratio < 1.
          2. Check propagation: services that call this service will show latency spikes.
          3. Check traces: trace_fan_out — callers see high p99 duration.
          4. Check logs: WARN/ERROR about thread pool queue length or rejected tasks.
          5. Check metrics: request_rate normal or rising (load not dropping yet).
          6. Root cause: recv spike + socket errors together = thread exhaustion pattern.
          7. Remediation: increase thread pool size, add circuit breaker, scale horizontally.
    """),

    "transaction_deadlock": textwrap.dedent("""\
        SOP: transaction_deadlock
        ──────────────────────────
        Symptom: circular lock waits → write latency extreme, read moderate.
        Sub-channel signature: write_avg_duration z-score >> read_avg_duration z-score.
        Suggested steps:
          1. Confirm with syscall_sub_channel_analysis: z(write) > 5 AND z(write) > z(read).
          2. Check if service is a data-store service (db_*, cache_*).
          3. Check logs: WARN/ERROR about lock timeouts or deadlock detection.
          4. Check concurrent services: do multiple services hit this data-store simultaneously?
          5. Cross-correlate callers: cross_correlate_services with high write_avg_duration.
          6. Root cause: write_duration spike without proportional read spike = deadlock not I/O saturation.
          7. Remediation: reduce transaction scope, add row-level locking, use optimistic concurrency.
    """),

    "disk_io_saturation": textwrap.dedent("""\
        SOP: disk_io_saturation
        ────────────────────────
        Symptom: storage I/O bandwidth exhausted → all read/write/fsync slow.
        Sub-channel signature: fsync_avg_duration HIGH, write_avg_duration HIGH, read_count elevated.
        Suggested steps:
          1. Confirm with syscall_sub_channel_analysis: z(fsync) + z(write) both > 3 AND read_count_ratio > 1.5.
          2. Check if this is the only service on the storage node (or co-located noisy neighbour).
          3. Check metrics: latency_ms elevated (I/O wait shows up as latency).
          4. Check propagation: downstream services will see slow query responses.
          5. Root cause: fsync + write both extreme distinguishes this from transaction_deadlock.
          6. Remediation: move to faster storage tier, throttle write batch sizes, add write buffering.
    """),

    "cascading_timeout": textwrap.dedent("""\
        SOP: cascading_timeout
        ───────────────────────
        Symptom: upstream slow → downstream timeouts propagate fan-out.
        Sub-channel signature: socket_errors HIGH, recv_avg_duration spike.
        Suggested steps:
          1. Identify the root slow service with syscall_multi_service_compare (highest z-score).
          2. Confirm with syscall_sub_channel_analysis on root: socket_errors ratio > 3, z(recv) > 3.
          3. Trace propagation: build_call_path from root to all affected high-z services.
          4. Verify temporal order: root anomaly onset must precede cascade (compare t_start across services).
          5. Check logs: WARN/ERROR about connection timeout or circuit breaker open.
          6. Root cause: the earliest-onset service with highest syscall z-score is the root.
          7. Remediation: add/tune circuit breakers, set shorter timeouts, add retry with backoff.
    """),

    "auth_failure": textwrap.dedent("""\
        SOP: auth_failure
        ──────────────────
        Symptom: authentication service under load or misconfigured → read storm, no writes.
        Sub-channel signature: read_count spike, write/send flat (auth is read-heavy).
        Suggested steps:
          1. Check syscall_multi_service_compare: auth_service (id=18) appears in top 3.
          2. Confirm with syscall_sub_channel_analysis: read_count_ratio > 1.5, write unchanged.
          3. Check propagation: ALL services call auth — if auth is root, system-wide latency rises.
          4. Check traces: compare_trace_latency on auth_service — p99 elevated.
          5. Check logs: ERROR entries about authentication failure or token validation.
          6. Root cause distinguisher: auth_service uniquely has 18 callers — widespread impact = auth fault.
          7. Remediation: check auth config, refresh credentials/tokens, scale auth service.
    """),

    "data_corruption": textwrap.dedent("""\
        SOP: data_corruption
        ─────────────────────
        Symptom: corrupted writes → read errors, retry storms, write error spike.
        Sub-channel signature: write_errors ratio HIGH, write_avg_duration spike.
        Suggested steps:
          1. Confirm with syscall_sub_channel_analysis: err_ratio(write) > 3 AND z(write) > 3.
          2. Check logs: ERROR entries about checksum failure, parse error, or data validation.
          3. Check propagation: downstream readers will see read errors → error_rate spike.
          4. Check metrics: error_rate in syscalls elevated for both writer AND its callee readers.
          5. Root cause: write errors + duration spike = corruption at write time (not read time).
          6. Remediation: validate input schemas, add checksums, roll back to last clean state.
    """),

    "api_version_mismatch": textwrap.dedent("""\
        SOP: api_version_mismatch
        ──────────────────────────
        Symptom: incompatible API versions → open/connect floods as clients retry.
        Sub-channel signature: open_count elevated, duration moderate.
        Suggested steps:
          1. Confirm with syscall_sub_channel_analysis: open_count_ratio > 2.
          2. Check logs: ERROR entries about version mismatch, 400/406 status codes.
          3. Check propagation: services that call this service will see errors, not latency.
          4. Check metrics: error_rate elevated for callee services.
          5. Root cause: open_count spike = connection establishment churn = protocol mismatch.
          6. Remediation: pin API version, add backwards compat layer, update client stubs.
    """),

    "data_race_condition": textwrap.dedent("""\
        SOP: data_race_condition
        ─────────────────────────
        Symptom: concurrent read/write contention → sporadic short intense bursts.
        Sub-channel signature: z(read) + z(write) both elevated; fsync_z low (not a disk fault).
        Suggested steps:
          1. Confirm with syscall_sub_channel_analysis: z(read) > 3 AND z(write) > 3 but z(fsync) < 2.
          2. Duration pattern: bursts are short (minutes), not sustained (unlike memory_leak).
          3. Check logs: WARN/ERROR about concurrent modification or optimistic lock failure.
          4. Check metrics: cpu_percent spikes in short bursts (contention → spin-wait).
          5. Root cause: read+write both spiking without fsync rules out disk — it's an in-memory race.
          6. Remediation: add synchronization primitives, use atomic operations, redesign to avoid shared state.
    """),

    "unknown": textwrap.dedent("""\
        SOP: unknown fault
        ───────────────────
        No strong sub-channel match was found. Follow this generic investigation path:
          1. syscall_multi_service_compare — identify top anomalous services.
          2. syscall_sub_channel_analysis — get sub-channel breakdown for top service.
          3. compute_anomaly_score — compute composite score for top 3 services.
          4. get_service_dependencies — map the topology around the anomalous service.
          5. get_propagation_candidates — find potential root causes upstream.
          6. Check logs and traces for any supporting evidence.
          7. finalize_rca with confidence=LOW if pattern is ambiguous.
    """),
}


# ── Tool 1: search_fault_knowledge (RAG — Phase 3) ───────────────────────────

def search_fault_knowledge(symptom_text: str, top_k: int = 2) -> ToolResult:
    """
    Search the RAG-indexed FAULT_ENCYCLOPEDIA.md for fault archetypes matching
    the described symptoms.  Uses sentence-transformer embeddings (all-MiniLM-L6-v2)
    and a FAISS cosine-similarity index over 10 fault-type chunks.

    Parameters
    ----------
    symptom_text : str
        Free-text description of the observed symptoms (e.g., "high read syscall
        count, latency decreased, error rate increased").
    top_k : int
        Number of matching fault archetypes to return (default 2).

    Returns
    -------
    ToolResult with top-K matching fault archetypes ranked by similarity.
    Each result includes: fault_type, fault_name, category,
    counter_intuitive flag, similarity score, and excerpt.
    """
    if not symptom_text or not symptom_text.strip():
        return ToolResult(
            summary="search_fault_knowledge: empty query — provide symptom description.",
            data={"error": "empty_query"},
        )

    if not _rag_ready():
        return ToolResult(
            summary=(
                "search_fault_knowledge: RAG index not available.\n"
                "  → Run `python -m omnirca.rag.indexer` to build it.\n"
                "  → Fallback: use check_sop(fault_category) with a known category."
            ),
            data={"status": "index_missing"},
        )

    results = _rag_retrieve(symptom_text, top_k=top_k)

    if not results:
        return ToolResult(
            summary="search_fault_knowledge: no matching fault archetypes found.",
            data={"query": symptom_text, "results": []},
        )

    lines = [f"search_fault_knowledge — top {len(results)} match(es) for: '{symptom_text[:80]}…'\n"]
    for i, r in enumerate(results, 1):
        ci_warn = "  ⚠ COUNTER-INTUITIVE: symptoms may look benign — see full text.\n" if r.get("counter_intuitive") else ""
        lines.append(
            f"[{i}] {r['fault_name']}  (type={r['fault_type']}, "
            f"category={r['category']}, similarity={r['similarity']:.3f})\n"
            f"{ci_warn}"
            f"    Excerpt: {r['excerpt'][:200]}…\n"
        )

    lines.append(
        "  → Call check_sop('<fault_type>') for the full investigation procedure."
    )

    return ToolResult(
        summary="\n".join(lines),
        data={
            "query":   symptom_text,
            "results": results,
        },
    )


# ── Tool 2: classify_fault_pattern (heuristic) ───────────────────────────────

def classify_fault_pattern(signals: Dict[str, Any] = None) -> ToolResult:
    """
    Classify the fault pattern from a signals dict using heuristic rules.
    The `signals` dict should contain keys like those returned by
    syscall_sub_channel_analysis and compute_anomaly_score, e.g.:
      {
        "syscall_avg_duration_z": 31.7,
        "syscall_p99_duration_z": 48.9,
        "syscall_error_rate_ratio": 1.2,
        "channel_stats": { "read": {"count_ratio": 2.2, "errors_ratio": 1.0, "duration_z": 28.5},
                            "write":{"count_ratio": 0.7, ...}, ... },
        "memory_slope_mb_per_s": 0.0001,
        ...
      }

    Returns: top fault categories ranked by score, recommended SOP key, confidence.
    `signals` is optional — if not provided or not a dict, all heuristic scores are
    zero and the tool falls back to returning 'unknown' with LOW confidence.
    """
    if signals is None or not isinstance(signals, dict):
        signals = {}
    cs = signals.get("channel_stats", {})

    def z(t):   return cs.get(t, {}).get("duration_z", 0.0)
    def cnt(t): return cs.get(t, {}).get("count_ratio", 1.0)
    def err(t): return cs.get(t, {}).get("errors_ratio", 1.0)

    mem_slope = signals.get("memory_slope_mb_per_s", 0.0) or 0.0
    global_z  = signals.get("syscall_avg_duration_z", 0.0) or 0.0

    scores: Dict[str, float] = {}

    # ── Primary discriminant: errors_ratio by channel ────────────────────────
    # duration_z is often uniform across all channels during a fault window;
    # the key differentiator is WHICH channels have elevated errors_ratio.
    #
    # Group A (network errors present): err(socket) > 3 AND err(recv) > 3
    #   → cascading_timeout / authentication_failure / data_race_condition
    # Group B (no network errors): err(socket) ≈ 1 AND err(recv) ≈ 1
    #   → transaction_deadlock / disk_io_saturation
    #
    # Within Group A:
    #   recv+socket errors > open+close errors → authentication_failure
    #   open+close errors ≥ recv+socket errors → cascading_timeout / data_race
    # Within Group B:
    #   fsync errors > write errors             → disk_io_saturation
    #   write/read errors dominate, fsync low   → transaction_deadlock

    network_error_present = err("socket") > 3.0 and err("recv") > 3.0
    recv_socket_avg = (err("recv") + err("socket")) / 2.0
    open_close_avg  = (err("open") + err("close"))  / 2.0
    recv_lead       = recv_socket_avg - open_close_avg  # +ve → auth, -ve → cascading

    # ── stale_cache: read count elevated, write count low ────────────────────
    scores["stale_cache"] = max(0.0, (cnt("read") - 1.0) * 5 + max(0.0, 1.0 - cnt("write")) * 3)

    # ── memory_leak: mmap count rising, positive memory slope ────────────────
    scores["memory_leak"] = (
        max(0.0, cnt("mmap") - 1.0) * 10 +
        max(0.0, mem_slope * 1000)
    )

    # ── thread_pool_exhaustion: recv count high, send low, socket errors ─────
    scores["thread_pool_exhaustion"] = (
        max(0.0, cnt("recv") - 1.0) * 5 +
        max(0.0, err("socket") - 1.0) * 3 +
        max(0.0, 1.0 - cnt("send")) * 3
    )

    # ── transaction_deadlock ─────────────────────────────────────────────────
    # Network errors absent, R/W errors high, fsync NOT dominant over write
    scores["transaction_deadlock"] = (
        max(0.0, err("write") - 3) * 3 +
        max(0.0, err("read")  - 3) * 2 +
        (10.0 if not network_error_present else -25.0) +
        (-8.0 if err("fsync") > err("write") else 0.0)  # penalise → disk_io instead
    )

    # ── disk_io_saturation ───────────────────────────────────────────────────
    # Network errors absent, fsync errors are the dominant channel
    scores["disk_io_saturation"] = (
        max(0.0, err("fsync") - 3) * 5 +
        max(0.0, z("fsync")   - 3) * 1 +
        (10.0 if not network_error_present else -25.0) +
        (6.0  if err("fsync") >= err("write") else -4.0)  # fsync leads → disk_io
    )

    # ── cascading_timeout ────────────────────────────────────────────────────
    # Network errors present, open+close error ratio >= recv+socket ratio
    scores["cascading_timeout"] = (
        max(0.0, err("socket") - 3) * 3 +
        max(0.0, err("recv")   - 3) * 2 +
        max(0.0, -recv_lead)        * 6 +   # bonus when open_close > recv_socket
        (-15.0 if not network_error_present else 0.0)
    )

    # ── authentication_failure ───────────────────────────────────────────────
    # Network errors present, recv+socket errors clearly HIGHER than open+close
    scores["authentication_failure"] = (
        max(0.0, err("recv")   - 3) * 3 +
        max(0.0, err("socket") - 3) * 3 +
        recv_lead                   * 8 +   # positive when recv>open pushes score up
        (-15.0 if not network_error_present else 0.0)
    )

    # ── data_corruption: write errors and write duration both elevated ────────
    # Penalise heavily if network errors are also present: true data corruption
    # is an isolated write-path fault, not a service-wide network failure.
    scores["data_corruption"] = (
        max(0.0, err("write") - 2) * 4 +
        max(0.0, z("write")   - 3) * 1 -
        (30.0 if network_error_present else 0.0)
    )

    # ── api_version_mismatch: open call count elevated ────────────────────────
    scores["api_version_mismatch"] = max(0.0, cnt("open") - 1.5) * 5

    # ── data_race_condition ───────────────────────────────────────────────────
    # Both read AND write errors elevated simultaneously, network errors present
    min_rw_err = min(err("read"), err("write"))
    scores["data_race_condition"] = (
        max(0.0, min_rw_err - 3) * 3 +
        (5.0 if network_error_present else -5.0) +
        (-5.0 if err("fsync") > 10 else 0.0)   # penalise if fsync dominant → disk
    )

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top_cat, top_score = ranked[0]

    # Confidence: ratio of top score to runner-up
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    conf = (
        "HIGH"   if top_score > 10 and (runner_up == 0 or top_score / runner_up > 2) else
        "MEDIUM" if top_score > 3  else
        "LOW"
    )
    if top_score < 1:
        top_cat  = "unknown"
        conf     = "LOW"

    lines = [
        "classify_fault_pattern result:",
        f"  Top category : {top_cat}  (score={top_score:.2f}, confidence={conf})",
        f"  Runner-up    : {ranked[1][0]}  (score={ranked[1][1]:.2f})" if len(ranked) > 1 else "",
        f"  All scores   : { {k: round(v,2) for k,v in ranked} }",
        f"  ->  Call check_sop('{top_cat}') for the investigation checklist.",
    ]

    return ToolResult(
        summary="\n".join(l for l in lines if l),
        data={
            "top_category":  top_cat,
            "top_score":     round(top_score, 2),
            "confidence":    conf,
            "all_scores":    {k: round(v, 2) for k, v in ranked},
            "recommended_sop": top_cat,
        },
    )


# ── Tool 3: check_sop ────────────────────────────────────────────────────────

def check_sop(fault_category: str) -> ToolResult:
    """
    Return the Standard Operating Procedure (SOP) for a given fault category.

    Phase 6 behaviour:
      1. Looks up the category in SopLibrary (the authoritative Phase 6 source).
      2. If not found in the built-in library, calls AutoSopGenerator to
         synthesise a novel SOP using the LLM, then caches and returns it.
      3. Falls back to the legacy _SOPS dict only if SopLibrary import fails.

    Known categories (built-in):
      stale_cache, memory_leak, thread_pool_exhaustion, transaction_deadlock,
      disk_io_saturation, cascading_timeout, auth_failure, data_corruption,
      api_version_mismatch, data_race_condition, unknown.

    Novel categories trigger auto-generation (LLM call, ~5s).
    """
    # Phase 6 path: delegate to SopLibrary
    try:
        # Normalise aliases: authentication_failure → auth_failure (internal SOP key)
        sop_key = "auth_failure" if fault_category == "authentication_failure" else fault_category
        SopLibrary = _get_sop_library()
        sop = SopLibrary.get(sop_key)

        if sop is not None:
            return ToolResult(
                summary=sop.text,
                data={
                    "category":        sop.fault_category,
                    "category_group":  sop.category_group,
                    "sop":             sop.text,
                    "key_signals":     sop.key_signals,
                    "diagnostic_steps": sop.diagnostic_steps,
                    "recommended_tools": sop.recommended_tools,
                    "is_auto_generated": sop.is_auto_generated,
                },
            )

        # Not in built-in library → auto-generate via LLM
        AutoSopGenerator = _get_auto_sop_generator()
        generated = AutoSopGenerator.get_or_generate(fault_category, signals={})

        return ToolResult(
            summary=generated.text,
            data={
                "category":          generated.fault_category,
                "category_group":    generated.category_group,
                "sop":               generated.text,
                "key_signals":       generated.key_signals,
                "diagnostic_steps":  generated.diagnostic_steps,
                "recommended_tools": generated.recommended_tools,
                "is_auto_generated": True,
                "note":              "auto-generated SOP (novel fault category)",
            },
        )

    except ImportError:
        # ── Legacy fallback if sops package not importable ───────────────────
        key = fault_category.lower().replace("-", "_").replace(" ", "_")
        if key not in _SOPS:
            close = [k for k in _SOPS if k.startswith(key[:4])]
            hint  = f"  Did you mean one of: {close}?" if close else ""
            return ToolResult(
                summary=(
                    f"check_sop: unknown fault category '{fault_category}'.{hint}\n"
                    f"  Valid categories: {', '.join(_SOPS.keys())}"
                ),
                data={"error": "unknown_category", "available": list(_SOPS.keys())},
            )
        sop_text = _SOPS[key]
        return ToolResult(
            summary=sop_text,
            data={"category": key, "sop": sop_text},
        )


# ── Tool 4: explain_evidence (LLM — Phase 4) ─────────────────────────────────

def explain_evidence(
    service:        str,
    fault_category: str,
    evidence_list:  Optional[List[str]] = None,
) -> ToolResult:
    """
    Generate a concise natural-language explanation of why `service` is the
    root cause of `fault_category`, synthesising the provided evidence using
    the LLM.

    Parameters
    ----------
    service         : Root-cause service name.
    fault_category  : Fault category (e.g. "stale_cache", "memory_leak").
    evidence_list   : Ordered list of evidence strings from prior tool calls.
                      If omitted, the LLM works from service + category alone.

    Returns
    -------
    ToolResult with a 2–3 paragraph natural-language explanation.
    """
    ev_lines = "\n".join(f"  • {e}" for e in (evidence_list or []))
    if not ev_lines:
        ev_lines = "  (no specific evidence provided)"

    prompt = (
        f"You are an expert SRE writing up a root cause analysis report.\n\n"
        f"Root-cause service : {service}\n"
        f"Fault category     : {fault_category}\n\n"
        f"Evidence gathered during investigation:\n{ev_lines}\n\n"
        f"Write a concise 2–3 paragraph technical explanation of:\n"
        f"  1. Why {service} is identified as the root cause.\n"
        f"  2. How the fault manifested and propagated.\n"
        f"  3. What engineering action should resolve it.\n\n"
        f"Be technically precise. Cite specific evidence where possible. "
        f"Do not speculate beyond the evidence."
    )

    try:
        simple_chat = _get_llm_simple_chat()
        explanation = simple_chat(
            user_prompt   = prompt,
            system_prompt = "You are an expert SRE producing concise RCA reports.",
            temperature   = 0.2,
        )
        summary = (
            f"explain_evidence({service}, {fault_category}):\n"
            f"────────────────────────────────────────\n"
            f"{explanation}\n"
            f"────────────────────────────────────────"
        )
        return ToolResult(
            summary=summary,
            data={
                "service":        service,
                "fault_category": fault_category,
                "explanation":    explanation,
                "evidence_list":  evidence_list or [],
            },
        )
    except Exception as e:
        return ToolResult(
            summary=(
                f"explain_evidence({service}, {fault_category}): "
                f"LLM call failed — {type(e).__name__}: {e}\n"
                f"  → Proceed with finalize_rca using manual evidence list."
            ),
            data={
                "status":         "llm_error",
                "error":          str(e),
                "service":        service,
                "fault_category": fault_category,
            },
        )


# ── Tool 5: finalize_rca ──────────────────────────────────────────────────────

def finalize_rca(
    root_cause:        str,
    confidence:        str,
    evidence_list:     List[str],
    fault_category:    str = "unknown",
    propagation_path:  Optional[List[str]] = None,
    affected_services: Optional[List[str]] = None,
    composite_score:   Optional[float] = None,
    notes:             str = "",
) -> ToolResult:
    """
    Produce the final Root Cause Analysis verdict.

    Parameters
    ----------
    root_cause : str
        Service name identified as the root cause.
    confidence : str
        "HIGH" | "MEDIUM" | "LOW"
    evidence_list : list[str]
        Ordered list of evidence strings supporting the conclusion.
    fault_category : str
        One of the 10 fault categories (or "unknown").
    propagation_path : list[str], optional
        Services in the propagation chain from root to observed symptom.
    affected_services : list[str], optional
        All services that showed elevated anomaly scores.
    composite_score : float, optional
        Composite anomaly score for the root cause service.
    notes : str
        Free-text notes for the analysis record.
    """
    conf_upper = confidence.upper()
    path_str   = " → ".join(propagation_path) if propagation_path else "N/A"
    aff_str    = ", ".join(affected_services)  if affected_services  else "N/A"
    score_str  = f"{composite_score:.2f}" if composite_score is not None else "N/A"
    ev_str     = "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(evidence_list))

    report = textwrap.dedent(f"""\
        ════════════════════════════════════════════
        ROOT CAUSE ANALYSIS — FINAL VERDICT
        ════════════════════════════════════════════
        Root cause        : {root_cause}
        Fault category    : {fault_category}
        Confidence        : {conf_upper}
        Composite score   : {score_str}
        Propagation path  : {path_str}
        Affected services : {aff_str}
        ────────────────────────────────────────────
        Evidence:
        {ev_str}
        ────────────────────────────────────────────
        Notes: {notes or 'N/A'}
        ════════════════════════════════════════════
    """)

    return ToolResult(
        summary=report,
        data={
            "root_cause":        root_cause,
            "fault_category":    fault_category,
            "confidence":        conf_upper,
            "composite_score":   composite_score,
            "propagation_path":  propagation_path,
            "affected_services": affected_services,
            "evidence_list":     evidence_list,
            "notes":             notes,
        },
    )
