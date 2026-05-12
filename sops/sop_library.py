"""
omnirca/sops/sop_library.py — Structured SOP Library (Phase 6).

Replaces the plain-string _SOPS dict in rca_tools.py with a first-class
typed library.  Each SOP is a dataclass carrying:
  • fault_category   — lookup key (stale_cache, cascading_timeout, …)
  • category_group   — plan-level grouping (see plan.md §6)
  • description      — one-sentence fault summary
  • key_signals      — signal patterns that indicate this fault (no service names)
  • diagnostic_steps — ordered investigation commands / decisions
  • recommended_tools— tool names referenced by the steps
  • text             — full prose SOP (used by check_sop() for backward compat)
  • is_auto_generated— True only for SOPs produced by AutoSopGenerator

Fairness rule (plan.md):
  SOPs MUST be pattern-based only — NO numbered service names (cache_0,
  backend_4…), NO fault IDs.  Role-level names (auth layer, cache tier)
  are acceptable because they describe a topology role, not a dataset instance.

Category groups (four from plan.md §6):
  data_integrity      — data_corruption, stale_cache, api_version_mismatch,
                        data_race_condition
  resource_exhaustion — memory_leak, thread_pool_exhaustion, disk_io_saturation
  cascading_deadlock  — cascading_timeout, transaction_deadlock
  auth                — auth_failure
  unknown             — catch-all when no pattern matches
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# SOP dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SOP:
    """A single Standard Operating Procedure entry."""
    fault_category:    str
    category_group:    str
    description:       str
    key_signals:       List[str]
    diagnostic_steps:  List[str]
    recommended_tools: List[str]
    text:              str
    is_auto_generated: bool = False

    def __repr__(self) -> str:
        return (
            f"<SOP category={self.fault_category!r} "
            f"group={self.category_group!r} "
            f"steps={len(self.diagnostic_steps)} "
            f"auto={self.is_auto_generated}>"
        )

    def summary(self) -> str:
        """Return a concise single-line description for logging."""
        return f"[{self.category_group}] {self.fault_category}: {self.description}"


# ─────────────────────────────────────────────────────────────────────────────
# Canonical SOP definitions
# ─────────────────────────────────────────────────────────────────────────────

def _sop(
    fault_category: str,
    category_group: str,
    description: str,
    key_signals: List[str],
    diagnostic_steps: List[str],
    recommended_tools: List[str],
    text: str,
) -> SOP:
    """Helper to build a SOP with dedented text."""
    return SOP(
        fault_category=fault_category,
        category_group=category_group,
        description=description,
        key_signals=key_signals,
        diagnostic_steps=diagnostic_steps,
        recommended_tools=recommended_tools,
        text=textwrap.dedent(text).strip(),
        is_auto_generated=False,
    )


_LIBRARY_LIST: List[SOP] = [

    # ── GROUP 1: Data Integrity ───────────────────────────────────────────────

    _sop(
        fault_category="stale_cache",
        category_group="data_integrity",
        description="Cache serves stale data — fast responses but data validation failures downstream.",
        key_signals=[
            "read_count_ratio > 2  (cache-miss storm → upstream reads)",
            "write_count_ratio < 0.8  (no invalidation writes)",
            "latency DECREASE is anomalous (stale serves fast)",
            "downstream services show error_rate spike despite low latency",
        ],
        diagnostic_steps=[
            "1. syscall_multi_service_compare(t_start, t_end) — identify highest-z cache-tier service.",
            "2. syscall_sub_channel_analysis(service, t_start, t_end) — confirm read_count_ratio > 2, write_count_ratio < 0.8.",
            "3. compare_trace_latency(service, t_start, t_end) — expect latency DECREASE (stale path is fast).",
            "4. get_propagation_candidates(affected_downstream, t_start, t_end) — verify cache tier is upstream.",
            "5. cross_correlate_services([cache_tier, downstream], 'error_rate', t_start, t_end) — confirm error propagation.",
            "6. check onset order: cache-tier anomaly must precede downstream error spikes.",
        ],
        recommended_tools=[
            "syscall_multi_service_compare",
            "syscall_sub_channel_analysis",
            "compare_trace_latency",
            "get_propagation_candidates",
            "cross_correlate_services",
        ],
        text="""\
            SOP: stale_cache  [data_integrity group]
            ─────────────────────────────────────────
            Symptom: cache serves stale data → fast but invalid responses → downstream
                     data validation failures.  Latency DECREASE is a key counter-intuitive signal.
            Sub-channel signature: read_count_ratio > 2, write_count_ratio < 0.8.

            Diagnostic steps:
              1. Run syscall_multi_service_compare — locate the cache-tier service with the highest
                 syscall z-score.
              2. Run syscall_sub_channel_analysis on that service — confirm:
                 read_count_ratio > 2 (cache-miss storm), write_count_ratio < 0.8 (no invalidation).
              3. Run compare_trace_latency — stale cache shows LATENCY DECREASE; verify this.
              4. Run get_propagation_candidates on a high-error downstream service — confirm the
                 cache-tier service is ranked as a root candidate.
              5. Cross-correlate: cache-tier error_rate and downstream error_rate should be correlated.
              6. Confirm temporal order: cache-tier anomaly onset precedes downstream errors.

            Root cause confirmation: cache-tier service with read-dominated syscalls AND
            downstream error propagation AND earlier onset = stale_cache root cause.
        """,
    ),

    _sop(
        fault_category="data_corruption",
        category_group="data_integrity",
        description="Corrupted writes — write errors escalate, downstream readers receive invalid data.",
        key_signals=[
            "write_errors_ratio > 3  (corruption at write time)",
            "write_avg_duration z-score elevated  (write conflicts/retries)",
            "downstream read_errors rising after write errors appear",
        ],
        diagnostic_steps=[
            "1. syscall_multi_service_compare(t_start, t_end) — identify data-tier service with highest z.",
            "2. syscall_sub_channel_analysis(service, t_start, t_end) — confirm write_errors_ratio > 3 AND z(write) > 3.",
            "3. detect_error_burst(service, t_start, t_end) — verify error_rate spike is sudden (not gradual).",
            "4. query_logs(service, t_start, t_end) — look for checksum/parse/validation failures.",
            "5. get_propagation_candidates(downstream_erroring_service, t_start, t_end) — confirm data-tier is upstream.",
            "6. detect_causal_order([write_error_service, read_error_services]) — write errors must precede read errors.",
        ],
        recommended_tools=[
            "syscall_multi_service_compare",
            "syscall_sub_channel_analysis",
            "detect_error_burst",
            "query_logs",
            "get_propagation_candidates",
            "detect_causal_order",
        ],
        text="""\
            SOP: data_corruption  [data_integrity group]
            ─────────────────────────────────────────────
            Symptom: corrupted writes → write error spike → downstream readers propagate bad data.
            Sub-channel signature: write_errors_ratio > 3, write_avg_duration z > 3.

            Diagnostic steps:
              1. syscall_multi_service_compare — find the data-tier service with highest syscall z.
              2. syscall_sub_channel_analysis — confirm write_errors_ratio > 3 AND z(write) > 3.
              3. detect_error_burst — confirm the spike is sudden (not a slow rise → not memory_leak).
              4. query_logs — look for checksum failure, parse error, schema validation messages.
              5. get_propagation_candidates — verify the corrupted service is upstream of affected readers.
              6. detect_causal_order — write errors must start before downstream read errors.

            Distinguishing from stale_cache: stale_cache shows LOW write + HIGH read;
            data_corruption shows HIGH write_errors specifically.
        """,
    ),

    _sop(
        fault_category="api_version_mismatch",
        category_group="data_integrity",
        description="Incompatible API versions — clients retry connections repeatedly, open_count floods.",
        key_signals=[
            "open_count_ratio > 2  (connection churn from rejected requests)",
            "error_rate elevated but latency moderate  (rejected early, not slow-processed)",
            "clients of the mismatched service show error propagation",
        ],
        diagnostic_steps=[
            "1. syscall_multi_service_compare(t_start, t_end) — identify service with elevated z.",
            "2. syscall_sub_channel_analysis(service, t_start, t_end) — confirm open_count_ratio > 2.",
            "3. query_logs(service, t_start, t_end) — look for version mismatch, 400/406 status codes.",
            "4. get_service_dependencies(service) — find all callers who may be sending wrong version.",
            "5. query_traces(service, t_start, t_end) — check for systematic success=False pattern.",
        ],
        recommended_tools=[
            "syscall_multi_service_compare",
            "syscall_sub_channel_analysis",
            "query_logs",
            "get_service_dependencies",
            "query_traces",
        ],
        text="""\
            SOP: api_version_mismatch  [data_integrity group]
            ────────────────────────────────────────────────────
            Symptom: API callers send incompatible request format → connections rejected →
                     clients retry → open_count floods.
            Sub-channel signature: open_count_ratio > 2, moderate latency, elevated error_rate.

            Diagnostic steps:
              1. syscall_multi_service_compare — identify the service whose connections are being rejected.
              2. syscall_sub_channel_analysis — confirm open_count_ratio > 2 (connect-retry storm).
              3. query_logs — look for version mismatch, unsupported content-type, 400/406 errors.
              4. get_service_dependencies — map which callers are sending the incompatible requests.
              5. query_traces — verify systematic success=False (connection rejection not slow path).

            Distinguishing from thread_pool_exhaustion: thread exhaustion has high recv + socket errors;
            api_version_mismatch has high open_count but normal recv.
        """,
    ),

    _sop(
        fault_category="data_race_condition",
        category_group="data_integrity",
        description="Concurrent read/write contention — sporadic short intense bursts in both channels.",
        key_signals=[
            "z(read) > 3 AND z(write) > 3 simultaneously  (both channels hot)",
            "z(fsync) < 2  (not a disk fault)",
            "burst pattern: spikes are short-lived, not sustained",
        ],
        diagnostic_steps=[
            "1. syscall_multi_service_compare(t_start, t_end) — identify the contended service.",
            "2. syscall_sub_channel_analysis(service, t_start, t_end) — confirm z(read) > 3 AND z(write) > 3, z(fsync) < 2.",
            "3. query_syscalls(service, t_start, t_end) — inspect raw time series for burst vs. sustained pattern.",
            "4. query_logs(service, t_start, t_end) — look for concurrent modification or lock failure messages.",
            "5. cross_correlate_services(caller_list, 'error_rate', t_start, t_end) — find simultaneous callers creating contention.",
        ],
        recommended_tools=[
            "syscall_multi_service_compare",
            "syscall_sub_channel_analysis",
            "query_syscalls",
            "query_logs",
            "cross_correlate_services",
        ],
        text="""\
            SOP: data_race_condition  [data_integrity group]
            ─────────────────────────────────────────────────
            Symptom: concurrent read/write contention → sporadic short spikes in both channels.
            Sub-channel signature: z(read) > 3 AND z(write) > 3, z(fsync) < 2.

            Diagnostic steps:
              1. syscall_multi_service_compare — identify the contended shared resource service.
              2. syscall_sub_channel_analysis — confirm BOTH read AND write elevated, fsync low.
              3. query_syscalls — inspect raw time series: race conditions show burst pattern (minutes),
                 not sustained rise (unlike memory_leak).
              4. query_logs — look for concurrent modification exceptions or optimistic lock failures.
              5. cross_correlate_services — find callers whose peaks overlap in time (joint contention).

            Distinguishing from disk_io_saturation: disk saturation has high fsync; race condition
            has low fsync but both read AND write elevated.
        """,
    ),

    # ── GROUP 2: Resource Exhaustion ─────────────────────────────────────────

    _sop(
        fault_category="memory_leak",
        category_group="resource_exhaustion",
        description="Gradual memory growth — mmap count rising, service eventually collapses under GC pressure.",
        key_signals=[
            "mmap_count_ratio > 1.3  (monotonically increasing allocations)",
            "memory_slope_mb_per_min > 0  (linearly growing, not spiky)",
            "latency degrades gradually, not in sudden spike",
        ],
        diagnostic_steps=[
            "1. syscall_multi_service_compare(t_start, t_end) — identify service with elevated syscall z.",
            "2. syscall_sub_channel_analysis(service, t_start, t_end) — confirm mmap_count_ratio > 1.3.",
            "3. detect_memory_slope(service, t_start, t_end) — confirm positive slope (LEAK CANDIDATE verdict).",
            "4. compare_trace_latency(service, t_start, t_end) — expect gradual p99 increase, not sudden spike.",
            "5. get_propagation_candidates(downstream, t_start, t_end) — verify this service is upstream of degraded callers.",
            "6. detect_causal_order([leaking_service, downstream_services]) — leaking service must have earliest onset.",
        ],
        recommended_tools=[
            "syscall_multi_service_compare",
            "syscall_sub_channel_analysis",
            "detect_memory_slope",
            "compare_trace_latency",
            "get_propagation_candidates",
            "detect_causal_order",
        ],
        text="""\
            SOP: memory_leak  [resource_exhaustion group]
            ───────────────────────────────────────────────
            Symptom: gradual mmap allocation growth → GC pressure → latency degradation.
            Sub-channel signature: mmap_count_ratio > 1.3 (rising), memory_slope > 0.
            NOTE: metric memory_mb slope is near-flat in some datasets — absent slope does NOT rule this out.

            Diagnostic steps:
              1. syscall_multi_service_compare — identify the service with elevated syscall duration z.
              2. syscall_sub_channel_analysis — confirm mmap_count_ratio > 1.3 (the primary discriminator).
              3. detect_memory_slope — confirm positive slope with LEAK CANDIDATE verdict.
              4. compare_trace_latency — expect gradual p99 rise, not sudden spike (confirms slow leak not crash).
              5. get_propagation_candidates — confirm this service is upstream of downstream degradation.
              6. detect_causal_order — leaking service onset must precede downstream degradation.

            Distinguishing from thread_pool_exhaustion: memory_leak has mmap_count rising;
            thread exhaustion has recv spike + socket errors.
        """,
    ),

    _sop(
        fault_category="thread_pool_exhaustion",
        category_group="resource_exhaustion",
        description="All worker threads blocked — incoming requests queue, recv spikes, send drops.",
        key_signals=[
            "recv_count_ratio > 2  (incoming requests queueing up)",
            "socket_errors_ratio > 2  (accept/connect failures due to queue full)",
            "send_count_ratio < 1  (no processed responses going out)",
        ],
        diagnostic_steps=[
            "1. syscall_multi_service_compare(t_start, t_end) — identify service with highest z.",
            "2. syscall_sub_channel_analysis(service, t_start, t_end) — confirm recv_count > 2, socket_errors > 2, send_count < 1.",
            "3. compare_trace_latency(service, t_start, t_end) — expect extreme p99 vs. normal p50 (queue-wait dominates tail).",
            "4. query_logs(service, t_start, t_end) — look for 'thread pool full', 'rejected execution', or queue-depth messages.",
            "5. get_propagation_candidates(downstream_caller, t_start, t_end) — callers see this service as upstream bottleneck.",
        ],
        recommended_tools=[
            "syscall_multi_service_compare",
            "syscall_sub_channel_analysis",
            "compare_trace_latency",
            "query_logs",
            "get_propagation_candidates",
        ],
        text="""\
            SOP: thread_pool_exhaustion  [resource_exhaustion group]
            ─────────────────────────────────────────────────────────
            Symptom: all worker threads blocked → requests queue → recv spikes, send drops.
            Sub-channel signature: recv_count_ratio > 2, socket_errors_ratio > 2, send_count_ratio < 1.

            Diagnostic steps:
              1. syscall_multi_service_compare — identify the service whose thread pool is exhausted.
              2. syscall_sub_channel_analysis — confirm recv HIGH, socket_errors HIGH, send LOW (hallmark pattern).
              3. compare_trace_latency — confirm extreme p99 vs. normal p50: callers wait in queue (p99 extreme)
                 but when scheduled, processing is fast (p50 normal).
              4. query_logs — look for thread pool rejected-execution or queue-length warnings.
              5. get_propagation_candidates — callers of this service will see timeout propagation.

            Distinguishing from cascading_timeout: thread exhaustion is at the processing service;
            cascading timeout starts at a slow upstream and fans out downstream.
        """,
    ),

    _sop(
        fault_category="disk_io_saturation",
        category_group="resource_exhaustion",
        description="Storage I/O bandwidth exhausted — all fsync and write operations slow simultaneously.",
        key_signals=[
            "z(fsync) > 3  (flush operations backed up)",
            "z(write) > 3  (write operations slow)",
            "read_count_ratio > 1.5  (read retries due to slow writes blocking I/O)",
        ],
        diagnostic_steps=[
            "1. syscall_multi_service_compare(t_start, t_end) — identify the storage-tier service.",
            "2. syscall_sub_channel_analysis(service, t_start, t_end) — confirm z(fsync) > 3 AND z(write) > 3 AND read_count > 1.5.",
            "3. compare_trace_latency(service, t_start, t_end) — expect BOTH p50 AND p99 elevated (all I/O is slow).",
            "4. get_propagation_candidates(downstream, t_start, t_end) — downstream services see slow query responses.",
            "5. detect_causal_order([storage_service, downstream]) — storage service onset must be earliest.",
        ],
        recommended_tools=[
            "syscall_multi_service_compare",
            "syscall_sub_channel_analysis",
            "compare_trace_latency",
            "get_propagation_candidates",
            "detect_causal_order",
        ],
        text="""\
            SOP: disk_io_saturation  [resource_exhaustion group]
            ──────────────────────────────────────────────────────
            Symptom: storage I/O bandwidth exhausted → all reads, writes, and fsyncs slow.
            Sub-channel signature: z(fsync) > 3 AND z(write) > 3, read_count_ratio > 1.5.

            Diagnostic steps:
              1. syscall_multi_service_compare — identify the data/storage service with highest z.
              2. syscall_sub_channel_analysis — confirm fsync AND write BOTH elevated (key differentiator).
              3. compare_trace_latency — disk saturation: BOTH p50 AND p99 are elevated (all I/O is slow,
                 unlike thread exhaustion where only p99 is extreme).
              4. get_propagation_candidates — downstream callers experience slow query responses.
              5. detect_causal_order — storage service anomaly must precede downstream degradation.

            Distinguishing from transaction_deadlock: deadlock has extreme write_duration but read is moderate;
            disk saturation has BOTH fsync and write extreme.
        """,
    ),

    # ── GROUP 3: Cascading / Deadlock ────────────────────────────────────────

    _sop(
        fault_category="cascading_timeout",
        category_group="cascading_deadlock",
        description="One slow service propagates timeout fan-out to all dependents downstream.",
        key_signals=[
            "socket_errors_ratio > 3  (timeout-close events at each hop)",
            "z(recv_avg_duration) > 3  (waiting for slow upstream)",
            "multiple services show simultaneous high-z (fan-out effect)",
            "earliest-onset service = root; others = victims",
        ],
        diagnostic_steps=[
            "1. syscall_multi_service_compare(t_start, t_end) — identify ALL services with elevated z.",
            "2. detect_causal_order(all_anomalous_services, t_start, t_end) — find the earliest onset — this is the root.",
            "3. syscall_sub_channel_analysis(earliest_service, t_start, t_end) — confirm socket_errors > 3, z(recv) > 3.",
            "4. build_call_path(earliest_service, each_victim) — confirm propagation paths exist in graph.",
            "5. get_dynamic_graph(t_start, t_end) — verify actual call patterns during the incident match propagation theory.",
            "6. compare_trace_latency(earliest_service, t_start, t_end) — confirm p99 extreme at root service.",
        ],
        recommended_tools=[
            "syscall_multi_service_compare",
            "detect_causal_order",
            "syscall_sub_channel_analysis",
            "build_call_path",
            "get_dynamic_graph",
            "compare_trace_latency",
        ],
        text="""\
            SOP: cascading_timeout  [cascading_deadlock group]
            ─────────────────────────────────────────────────
            Symptom: one slow service propagates timeouts fan-out style to all dependents.
            Sub-channel signature: socket_errors_ratio > 3, z(recv_avg_duration) > 3 at origin.

            Diagnostic steps:
              1. syscall_multi_service_compare — list ALL services with elevated z (expect multiple).
              2. detect_causal_order — find which service's anomaly started FIRST; that service is root.
              3. syscall_sub_channel_analysis on root — confirm socket_errors > 3 and recv duration z > 3.
              4. build_call_path from root to each victim — confirm propagation paths exist in topology.
              5. get_dynamic_graph — verify the actual call graph during incident shows root-to-victim edges.
              6. compare_trace_latency on root — p99 extreme while p50 moderate (waiting pattern).

            Key insight: temporal ordering is THE discriminator for cascading faults.
            Without detect_causal_order, any simultaneous victim looks like a candidate.
        """,
    ),

    _sop(
        fault_category="transaction_deadlock",
        category_group="cascading_deadlock",
        description="Circular lock waits in a data-store — write duration extreme while reads moderate.",
        key_signals=[
            "z(write_avg_duration) >> z(read_avg_duration)  (writes blocked, reads still proceeding)",
            "write_errors escalating  (lock timeout kills)",
            "service role is a data-store  (database or stateful cache tier)",
        ],
        diagnostic_steps=[
            "1. syscall_multi_service_compare(t_start, t_end) — identify the data-store service.",
            "2. syscall_sub_channel_analysis(service, t_start, t_end) — confirm z(write) > 5 AND z(write) >> z(read).",
            "3. query_logs(service, t_start, t_end) — look for lock timeout, deadlock victim, or rollback messages.",
            "4. get_service_dependencies(service) — identify concurrent callers that may hold conflicting locks.",
            "5. cross_correlate_services(concurrent_callers, 'error_rate', t_start, t_end) — confirm joint lock contention timing.",
            "6. compare_trace_latency(service, t_start, t_end) — write latency extreme, read latency moderate.",
        ],
        recommended_tools=[
            "syscall_multi_service_compare",
            "syscall_sub_channel_analysis",
            "query_logs",
            "get_service_dependencies",
            "cross_correlate_services",
            "compare_trace_latency",
        ],
        text="""\
            SOP: transaction_deadlock  [cascading_deadlock group]
            ──────────────────────────────────────────────────────
            Symptom: circular lock waits → write duration extreme → transactions roll back.
            Sub-channel signature: z(write) >> z(read), write_errors escalating.

            Diagnostic steps:
              1. syscall_multi_service_compare — identify the data-store tier service.
              2. syscall_sub_channel_analysis — confirm z(write) > 5 AND z(write) >> z(read).
                 This asymmetry distinguishes deadlock from disk saturation (which has both high).
              3. query_logs — look for "deadlock detected", "lock timeout", or "transaction rolled back".
              4. get_service_dependencies — find all concurrent callers that write to this store.
              5. cross_correlate_services — concurrent writers should show correlated error spikes.
              6. compare_trace_latency — write paths have extreme latency; read paths are moderate.

            Distinguishing from disk_io_saturation: disk saturation = fsync AND write both extreme;
            transaction_deadlock = write extreme but fsync normal.
        """,
    ),

    # ── GROUP 4: Auth / Identity ──────────────────────────────────────────────

    _sop(
        fault_category="auth_failure",
        category_group="auth",
        description="Authentication layer failure — system-wide secondary degradation because all services depend on auth.",
        key_signals=[
            "auth-tier service has highest syscall z  (all dependents are affected)",
            "read_count_ratio > 1.5 at auth tier  (repeated credential lookups)",
            "write/send flat  (requests rejected before processing; no writes needed)",
            "system-wide secondary elevation across many services simultaneously",
        ],
        diagnostic_steps=[
            "1. syscall_multi_service_compare(t_start, t_end) — confirm auth-tier service appears in top 3.",
            "2. syscall_sub_channel_analysis(auth_service, t_start, t_end) — confirm read_count > 1.5, write/send flat.",
            "3. get_service_dependencies(auth_service) — confirm the auth service has a large fan-in (many callers).",
            "4. compare_trace_latency(auth_service, t_start, t_end) — confirm p99 elevated at auth tier.",
            "5. cross_correlate_services(all_dependents, 'error_rate', t_start, t_end) — system-wide simultaneous degradation confirms auth root.",
            "6. detect_causal_order([auth_service, sample_of_dependents]) — auth service anomaly must precede all dependent degradation.",
        ],
        recommended_tools=[
            "syscall_multi_service_compare",
            "syscall_sub_channel_analysis",
            "get_service_dependencies",
            "compare_trace_latency",
            "cross_correlate_services",
            "detect_causal_order",
        ],
        text="""\
            SOP: auth_failure  [auth group]
            ────────────────────────────────
            Symptom: authentication-layer failure → all dependent services cannot authenticate
                     → system-wide secondary degradation.
            Sub-channel signature: read_count_ratio > 1.5 at auth tier, write/send flat.

            Diagnostic steps:
              1. syscall_multi_service_compare — confirm the auth-tier service is in the top 3 anomalous.
              2. syscall_sub_channel_analysis — confirm read_count > 1.5, write/send both flat
                 (auth is read-heavy: it looks up credentials, not write them).
              3. get_service_dependencies — verify the auth service has a large fan-in (called by many).
                 System-wide impact from one service = that service is a universal dependency.
              4. compare_trace_latency — confirm elevated p99 at auth tier.
              5. cross_correlate_services on dependents — if ALL callers degrade simultaneously → auth root.
              6. detect_causal_order — auth-tier anomaly must precede all dependent degradation.

            Key distinguisher: the UNIQUE property of the auth tier is it has the MOST callers.
            No other fault type produces simultaneous system-wide degradation from a single node.

            NOTE: In this dataset — 0 ERROR-level log entries. Log silence does NOT rule out auth failure.
        """,
    ),

    # ── CATCH-ALL ─────────────────────────────────────────────────────────────

    _sop(
        fault_category="unknown",
        category_group="unknown",
        description="No strong sub-channel pattern matched — generic investigation path.",
        key_signals=[
            "weak or ambiguous sub-channel signals",
            "multiple fault categories score similarly in classify_fault_pattern",
        ],
        diagnostic_steps=[
            "1. syscall_multi_service_compare(t_start, t_end) — identify top anomalous services.",
            "2. syscall_sub_channel_analysis on top 2 services — collect sub-channel hints.",
            "3. compute_anomaly_score on top 3 candidates — rank by composite signal.",
            "4. get_service_dependencies(top_service) — map topology context.",
            "5. get_propagation_candidates(top_service, t_start, t_end) — find upstream candidates.",
            "6. detect_causal_order(top_candidates, t_start, t_end) — find earliest onset.",
            "7. search_fault_knowledge(sub_channel_hints) — attempt RAG-based fault classification.",
            "8. finalize_rca with confidence=LOW if no pattern emerges.",
        ],
        recommended_tools=[
            "syscall_multi_service_compare",
            "syscall_sub_channel_analysis",
            "compute_anomaly_score",
            "get_service_dependencies",
            "get_propagation_candidates",
            "detect_causal_order",
            "search_fault_knowledge",
        ],
        text="""\
            SOP: unknown fault  [unknown group]
            ────────────────────────────────────
            No strong sub-channel match. Follow this generic investigation path:
              1. syscall_multi_service_compare — identify top anomalous services.
              2. syscall_sub_channel_analysis on top 2 services — collect sub-channel hints.
              3. compute_anomaly_score on top 3 candidates — rank by composite signal.
              4. get_service_dependencies on top service — map topology context.
              5. get_propagation_candidates — find upstream candidates.
              6. detect_causal_order among top candidates — find earliest onset.
              7. search_fault_knowledge with sub-channel hints — attempt RAG classification.
              8. finalize_rca with confidence=LOW if pattern remains ambiguous.
        """,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Build lookup index
# ─────────────────────────────────────────────────────────────────────────────

_LIBRARY: Dict[str, SOP] = {sop.fault_category: sop for sop in _LIBRARY_LIST}


# ─────────────────────────────────────────────────────────────────────────────
# SopLibrary — public interface
# ─────────────────────────────────────────────────────────────────────────────

class SopLibrary:
    """
    Static facade over the SOP library.

    All methods are class-methods so no instantiation is needed.
    The library is module-level (_LIBRARY dict) to keep state simple.

    External SOPs added via AutoSopGenerator are registered here via
    SopLibrary.register(sop), making them available to all subsequent
    check_sop() calls in the same process.
    """

    # Mutable overlay for auto-generated SOPs (not in _LIBRARY_LIST at import time)
    _extra: Dict[str, SOP] = {}

    @classmethod
    def get(cls, fault_category: str) -> Optional[SOP]:
        """
        Look up a SOP by fault_category key.

        Returns None if the category is neither known nor auto-generated.
        Keys are normalised (lowercase, spaces/hyphens → underscores).
        """
        key = _normalise(fault_category)
        return _LIBRARY.get(key) or cls._extra.get(key)

    @classmethod
    def get_text(cls, fault_category: str) -> Optional[str]:
        """Return the full SOP text string, or None if not found."""
        sop = cls.get(fault_category)
        return sop.text if sop else None

    @classmethod
    def register(cls, sop: SOP) -> None:
        """
        Register an auto-generated SOP into the extra overlay.
        Built-in SOPs cannot be overwritten.
        """
        key = _normalise(sop.fault_category)
        if key in _LIBRARY:
            return   # never overwrite canonical SOPs
        cls._extra[key] = sop

    @classmethod
    def list_categories(cls) -> List[str]:
        """Return all known fault category keys (built-in + registered)."""
        return sorted(list(_LIBRARY.keys()) + list(cls._extra.keys()))

    @classmethod
    def list_groups(cls) -> List[str]:
        """Return the four canonical category group names plus 'unknown'."""
        return sorted({s.category_group for s in _LIBRARY.values()})

    @classmethod
    def by_group(cls, group: str) -> List[SOP]:
        """Return all SOPs belonging to the given category_group."""
        return [s for s in _LIBRARY.values() if s.category_group == group]

    @classmethod
    def all(cls) -> Dict[str, SOP]:
        """Return the full combined library (built-in + auto-generated)."""
        combined = dict(_LIBRARY)
        combined.update(cls._extra)
        return combined

    @classmethod
    def is_known(cls, fault_category: str) -> bool:
        """True if the category exists in the built-in library (not auto-gen)."""
        return _normalise(fault_category) in _LIBRARY

    @classmethod
    def clear_extra(cls) -> None:
        """Remove all auto-generated SOPs (useful for testing)."""
        cls._extra.clear()

    def __repr__(self) -> str:
        return (
            f"<SopLibrary built_in={len(_LIBRARY)} "
            f"auto_generated={len(self._extra)}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(key: str) -> str:
    """Canonical fault-category key: lower, spaces/hyphens → underscore."""
    return key.lower().replace("-", "_").replace(" ", "_")
