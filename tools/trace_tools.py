"""
Trace tools — query traces.csv, build fan-out views, compare latency vs. baseline.

Important notes from prebuild_check2.py:
  - traces.csv uses the column `start_time` (NOT `timestamp`).
  - Only 10 of 20 services appear in traces — gracefully return empty results for others.
  - Only 2 failed spans across all 5,301 rows — success rate near 100%.
  - z-scores will be low (~0.5) but it's worth checking for latency spikes on traced edges.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from ..data_layer.loader import get_loader
from .base import ToolResult, to_ts, get_baseline_window, compute_z_score


# ── Tool 1: query_traces ──────────────────────────────────────────────────────

def query_traces(
    service: str,
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> ToolResult:
    """
    Return raw trace spans for a single service in the given time window.

    Note: the timestamp column in traces.csv is `start_time`, not `timestamp`.
    Only 10 of 20 services appear in the trace data; missing services return
    an informative empty result.
    """
    loader  = get_loader()
    t_start = to_ts(t_start)
    t_end   = to_ts(t_end)
    loader.resolve_service(service)

    tr   = loader.traces
    rows = tr[
        (tr["service_name"] == service) &
        (tr["start_time"]   >= t_start) &
        (tr["start_time"]   <= t_end)
    ].copy()

    if rows.empty:
        return ToolResult(
            summary=f"query_traces({service}): NO spans in window "
                    f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')} "
                    f"(service may not be instrumented for tracing)",
            data=rows,
        )

    success_rate = rows["success"].mean() if "success" in rows.columns else float("nan")
    p50 = rows["duration_ms"].median()
    p99 = rows["duration_ms"].quantile(0.99)
    summary = (
        f"query_traces({service}, "
        f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}) "
        f"— {len(rows)} spans: "
        f"p50={p50:.1f}ms  p99={p99:.1f}ms  "
        f"success_rate={success_rate:.3f}"
    )
    return ToolResult(summary=summary, data=rows)


# ── Tool 2: trace_fan_out ─────────────────────────────────────────────────────

def trace_fan_out(
    service: str,
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> ToolResult:
    """
    Identify every downstream service that `service` directly called in the window,
    using the parent_span_id join in traces.csv.

    Algorithm:
      1. Find all span_ids belonging to `service` in the window.
      2. Find all spans in the FULL trace set whose parent_span_id is in step-1 set.
      3. Group step-2 spans by service_name to get callee services + their latencies.

    Returns a ranked table: callee_service, call_count, avg_duration_ms, p99_duration_ms,
    success_rate.
    """
    loader  = get_loader()
    t_start = to_ts(t_start)
    t_end   = to_ts(t_end)
    loader.resolve_service(service)

    tr = loader.traces

    # Step 1: spans owned by the target service in the window
    svc_spans = tr[
        (tr["service_name"] == service) &
        (tr["start_time"]   >= t_start) &
        (tr["start_time"]   <= t_end)
    ]
    if svc_spans.empty:
        return ToolResult(
            summary=f"trace_fan_out({service}): service has no spans in window "
                    f"— likely not instrumented.",
            data={},
        )

    svc_span_ids = set(svc_spans["span_id"].dropna().astype(str))

    # Step 2: spans whose parent_span_id points to one of service's spans
    # (look across full trace file so we catch callees outside the window boundary)
    callee_spans = tr[
        tr["parent_span_id"].astype(str).isin(svc_span_ids)
    ]
    if callee_spans.empty:
        return ToolResult(
            summary=f"trace_fan_out({service}): no outgoing calls found "
                    f"(leaf service or no callee spans in data).",
            data={},
        )

    # Step 3: aggregate by callee service
    grouped = (
        callee_spans.groupby("service_name")["duration_ms"]
        .agg(
            call_count="count",
            avg_duration_ms="mean",
            p99_duration_ms=lambda x: x.quantile(0.99),
        )
        .reset_index()
        .rename(columns={"service_name": "callee_service"})
        .sort_values("avg_duration_ms", ascending=False)
    )
    grouped["avg_duration_ms"] = grouped["avg_duration_ms"].round(2)
    grouped["p99_duration_ms"] = grouped["p99_duration_ms"].round(2)

    lines = [
        f"trace_fan_out({service}, "
        f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}) "
        f"— {int(svc_spans['span_id'].nunique())} parent spans → "
        f"{len(grouped)} callee service(s):",
        f"  {'Callee':<22} {'Calls':>7} {'Avg ms':>8} {'P99 ms':>8}",
        "  " + "-" * 50,
    ]
    for _, row in grouped.iterrows():
        lines.append(
            f"  {row['callee_service']:<22} "
            f"{int(row['call_count']):>7} "
            f"{row['avg_duration_ms']:>8.2f} "
            f"{row['p99_duration_ms']:>8.2f}"
        )

    return ToolResult(
        summary="\n".join(lines),
        data=grouped,
    )


# ── Tool 3: compare_trace_latency ─────────────────────────────────────────────

def compare_trace_latency(
    service: str,
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> ToolResult:
    """
    Compare trace span latency (p50, p99) in the fault window against the 1-hour baseline.
    Returns z-scores and ratio for quick anomaly triage.

    Note: z-scores here are typically low (≤0.5) because trace coverage is sparse;
    use syscall_multi_service_compare for authoritative anomaly ranking.
    """
    loader  = get_loader()
    t_start = to_ts(t_start)
    t_end   = to_ts(t_end)
    bs, be  = get_baseline_window(t_start)
    loader.resolve_service(service)

    tr     = loader.traces
    svc_tr = tr[tr["service_name"] == service]

    base = svc_tr[(svc_tr["start_time"] >= bs)      & (svc_tr["start_time"] < be)]
    dur  = svc_tr[(svc_tr["start_time"] >= t_start) & (svc_tr["start_time"] <= t_end)]

    if dur.empty:
        return ToolResult(
            summary=f"compare_trace_latency({service}): no spans in window — "
                    f"service not traced or no activity.",
            data={"z_score": 0.0, "p50_during": None, "p99_during": None},
        )

    z_score = compute_z_score(dur["duration_ms"], base["duration_ms"])

    p50_b = base["duration_ms"].median() if not base.empty else float("nan")
    p99_b = base["duration_ms"].quantile(0.99) if not base.empty else float("nan")
    p50_d = dur["duration_ms"].median()
    p99_d = dur["duration_ms"].quantile(0.99)
    ratio = p99_d / p99_b if p99_b and p99_b > 0 else float("nan")

    summary = (
        f"compare_trace_latency({service}, "
        f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}):\n"
        f"  p50:   {p50_b:.1f}ms → {p50_d:.1f}ms\n"
        f"  p99:   {p99_b:.1f}ms → {p99_d:.1f}ms  (ratio={ratio:.2f}×)\n"
        f"  z_score:  {z_score:.3f}  "
        f"[spans: baseline={len(base)}, during={len(dur)}]"
    )
    return ToolResult(
        summary=summary,
        data={
            "service":          service,
            "z_score":          round(z_score, 4),
            "p50_baseline_ms":  round(p50_b, 2) if not isinstance(p50_b, float) or p50_b == p50_b else None,
            "p50_during_ms":    round(p50_d, 2),
            "p99_baseline_ms":  round(p99_b, 2) if not isinstance(p99_b, float) or p99_b == p99_b else None,
            "p99_during_ms":    round(p99_d, 2),
            "latency_ratio":    round(ratio, 3) if not isinstance(ratio, float) or ratio == ratio else None,
            "baseline_spans":   len(base),
            "during_spans":     len(dur),
        },
    )
