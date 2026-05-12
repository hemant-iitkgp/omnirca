"""
Metric tools — query metrics.csv and compute composite anomaly scores.

Important findings from assess.py:
  - response_time_p95 and response_time_p99 are ALL ZEROS — never use these columns.
  - cpu_percent, latency_ms: z-score ≤ 0.07 in fault windows — very weak signal.
  - memory_mb slope detection is weakly useful for leak-type faults.
  - All three tools are included for completeness; downstream (rca_tools) uses weights
    that correctly reflect this hierarchy (syscalls 4×, metrics ≤ 0.5×).
"""
from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np

from ..config import ANOMALY_THRESHOLD, SIGNAL_WEIGHTS, BASELINE_MINUTES
from ..data_layer.loader import get_loader
from .base import (
    ToolResult,
    to_ts,
    get_baseline_window,
    compute_z_score,
    compute_ratio,
    severity_label,
)

# Columns that are ALWAYS zero in this dataset — never include in scoring
_ZERO_COLS = {"response_time_p95", "response_time_p99"}

# Metric columns that carry any usable (even weak) signal
_SCORE_COLS = ["cpu_percent", "latency_ms", "memory_mb", "error_rate", "request_rate"]


# ── Tool 1: query_metrics ──────────────────────────────────────────────────────

def query_metrics(
    service: str,
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> ToolResult:
    """
    Return raw metric rows for a single service in the given time window.
    Excludes permanently-zero columns (response_time_p95/p99).
    """
    loader  = get_loader()
    t_start = to_ts(t_start)
    t_end   = to_ts(t_end)
    loader.resolve_service(service)

    m    = loader.metrics
    rows = m[
        (m["service_name"] == service) &
        (m["timestamp"]    >= t_start) &
        (m["timestamp"]    <= t_end)
    ].drop(columns=[c for c in _ZERO_COLS if c in m.columns]).copy()

    if rows.empty:
        return ToolResult(
            summary=f"query_metrics({service}): NO data in window {t_start}–{t_end}",
            data=rows,
        )

    stats = rows[_SCORE_COLS].describe().loc[["mean", "std"]].to_dict()
    stats_str = "  ".join(
        f"{col}: mean={v['mean']:.3f} std={v['std']:.3f}"
        for col, v in stats.items()
        if col in rows.columns
    )
    summary = (
        f"query_metrics({service}, "
        f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}) "
        f"— {len(rows)} rows:\n  {stats_str}"
    )
    return ToolResult(summary=summary, data=rows)


# ── Tool 2: compute_anomaly_score ─────────────────────────────────────────────

def compute_anomaly_score(
    service: str,
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> ToolResult:
    """
    Compute a composite anomaly score for a single service in the given window,
    combining all available signals using empirically calibrated weights.

    Weight hierarchy (syscalls >> metrics >> traces > logs):
      syscall_avg_duration_us : 4.0 ×
      syscall_p99_duration_us : 4.0 ×
      syscall_error_rate      : 2.0 × (ratio-based)
      memory_slope            : 1.5 ×
      cpu_percent             : 0.5 ×
      latency_ms              : 0.5 ×
      trace_latency           : 0.5 ×
      log_error_density       : 0.2 ×

    Returns: total_score (float), severity label, and per-signal breakdown.
    """
    loader  = get_loader()
    t_start = to_ts(t_start)
    t_end   = to_ts(t_end)
    bs, be  = get_baseline_window(t_start)
    loader.resolve_service(service)

    component_scores: dict[str, float] = {}
    details:          dict[str, object] = {}

    # ── SYSCALL SIGNALS ────────────────────────────────────────────────────────
    sc     = loader.syscalls
    svc_sc = sc[sc["service_name"] == service]
    sc_b   = svc_sc[(svc_sc["timestamp"] >= bs)      & (svc_sc["timestamp"] < be)]
    sc_d   = svc_sc[(svc_sc["timestamp"] >= t_start) & (svc_sc["timestamp"] <= t_end)]

    if len(sc_b) >= 3 and len(sc_d) >= 1:
        z_avg = compute_z_score(sc_d["avg_duration_us"], sc_b["avg_duration_us"])
        component_scores["syscall_avg_duration_us"] = z_avg * SIGNAL_WEIGHTS["syscall_avg_duration_us"]
        details["syscall_avg_duration_z"] = round(z_avg, 2)

        z_p99 = compute_z_score(sc_d["p99_duration_us"], sc_b["p99_duration_us"])
        component_scores["syscall_p99_duration_us"] = z_p99 * SIGNAL_WEIGHTS["syscall_p99_duration_us"]
        details["syscall_p99_duration_z"] = round(z_p99, 2)

        err_ratio = compute_ratio(sc_d["error_rate"], sc_b["error_rate"])
        component_scores["syscall_error_rate"] = (
            abs(err_ratio - 1.0) * SIGNAL_WEIGHTS["syscall_error_rate"]
        )
        details["syscall_error_rate_ratio"] = round(err_ratio, 2)

    # ── METRIC SIGNALS ────────────────────────────────────────────────────────
    m     = loader.metrics
    svc_m = m[m["service_name"] == service]
    m_b   = svc_m[(svc_m["timestamp"] >= bs)      & (svc_m["timestamp"] < be)]
    m_d   = svc_m[(svc_m["timestamp"] >= t_start) & (svc_m["timestamp"] <= t_end)]

    if len(m_b) >= 5 and len(m_d) >= 2:
        z_cpu = compute_z_score(m_d["cpu_percent"], m_b["cpu_percent"])
        component_scores["cpu_percent"] = z_cpu * SIGNAL_WEIGHTS["cpu_percent"]
        details["cpu_percent_z"] = round(z_cpu, 4)

        z_lat = compute_z_score(m_d["latency_ms"], m_b["latency_ms"])
        component_scores["latency_ms"] = z_lat * SIGNAL_WEIGHTS["latency_ms"]
        details["latency_ms_z"] = round(z_lat, 4)

        # memory_slope: linear regression coefficient in MB/s
        times  = (m_d["timestamp"] - m_d["timestamp"].iloc[0]).dt.total_seconds().values
        mem    = m_d["memory_mb"].values
        if len(times) >= 3 and times[-1] > 0:
            slope = float(np.polyfit(times, mem, 1)[0])          # MB/s
            norm_slope = abs(slope) / 0.005                       # normalize: 0.005 MB/s = 1 unit
            component_scores["memory_slope"] = norm_slope * SIGNAL_WEIGHTS["memory_slope"]
            details["memory_slope_mb_per_s"] = round(slope, 6)

    # ── LOG SIGNALS ───────────────────────────────────────────────────────────
    logs     = loader.logs
    svc_logs = logs[logs["service_name"] == service]
    l_b      = svc_logs[(svc_logs["timestamp"] >= bs)      & (svc_logs["timestamp"] < be)]
    l_d      = svc_logs[(svc_logs["timestamp"] >= t_start) & (svc_logs["timestamp"] <= t_end)]

    win_min  = max((t_end   - t_start).total_seconds() / 60.0, 1.0)
    base_min = max((be      - bs      ).total_seconds() / 60.0, 1.0)
    err_d    = int((l_d["level"] == "ERROR").sum())
    err_b    = int((l_b["level"] == "ERROR").sum())
    log_density_delta = max(err_d / win_min - err_b / base_min, 0.0)
    component_scores["log_error_density"] = log_density_delta * SIGNAL_WEIGHTS["log_error_density"]
    details["log_errors_during"]  = err_d
    details["log_errors_baseline"] = err_b

    # ── TRACE SIGNALS ─────────────────────────────────────────────────────────
    tr     = loader.traces
    svc_tr = tr[tr["service_name"] == service]
    t_b    = svc_tr[(svc_tr["start_time"] >= bs)      & (svc_tr["start_time"] < be)]
    t_d    = svc_tr[(svc_tr["start_time"] >= t_start) & (svc_tr["start_time"] <= t_end)]

    if len(t_b) >= 3 and len(t_d) >= 1:
        z_trace = compute_z_score(t_d["duration_ms"], t_b["duration_ms"])
        component_scores["trace_latency"] = z_trace * SIGNAL_WEIGHTS["trace_latency"]
        details["trace_latency_z"] = round(z_trace, 4)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total    = sum(component_scores.values())
    severity = severity_label(total)

    lines = [
        f"compute_anomaly_score({service}, "
        f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}):",
        f"  Composite score: {total:.2f} [{severity}]",
        f"  Signal breakdown (only signals with data):",
    ]
    for sig, raw_score in sorted(component_scores.items(), key=lambda x: -x[1]):
        lines.append(f"    {sig:<35} weighted={raw_score:.3f}")
    lines.append(f"  Detail values: {details}")

    return ToolResult(
        summary="\n".join(lines),
        data={
            "service":           service,
            "t_start":           str(t_start),
            "t_end":             str(t_end),
            "total_score":       round(total, 4),
            "severity":          severity,
            "component_scores":  component_scores,
            "details":           details,
        },
    )


# ── Tool 3: detect_memory_slope ───────────────────────────────────────────────

def detect_memory_slope(
    service: str,
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> ToolResult:
    """
    Fit a linear regression to memory_mb over the fault window.
    A positive slope suggests a memory leak; a slope near zero is normal.

    Returns slope in MB/s and annualised MB/min, compared against the baseline slope.
    """
    loader  = get_loader()
    t_start = to_ts(t_start)
    t_end   = to_ts(t_end)
    bs, be  = get_baseline_window(t_start)
    loader.resolve_service(service)

    m     = loader.metrics
    svc_m = m[m["service_name"] == service]

    def _slope(df: pd.DataFrame) -> Optional[float]:
        if len(df) < 3:
            return None
        t = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().values
        if t[-1] < 1:
            return None
        return float(np.polyfit(t, df["memory_mb"].values, 1)[0])

    base_rows = svc_m[(svc_m["timestamp"] >= bs)      & (svc_m["timestamp"] < be)]
    dur_rows  = svc_m[(svc_m["timestamp"] >= t_start) & (svc_m["timestamp"] <= t_end)]

    s_base = _slope(base_rows)
    s_dur  = _slope(dur_rows)

    if s_dur is None:
        return ToolResult(
            summary=f"detect_memory_slope({service}): "
                    f"Insufficient metric rows in window ({len(dur_rows)} rows).",
            data={"slope_mb_per_s": None},
        )

    base_str = f"{s_base*60:.4f} MB/min" if s_base is not None else "N/A"
    verdict  = (
        "LEAK CANDIDATE" if s_dur > 0.002 else
        "MODEST GROWTH"  if s_dur > 0.0005 else
        "STABLE"
    )
    summary = (
        f"detect_memory_slope({service}, "
        f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}):\n"
        f"  During window : {s_dur*60:.4f} MB/min  [{verdict}]\n"
        f"  Baseline slope: {base_str}"
    )
    return ToolResult(
        summary=summary,
        data={
            "service":              service,
            "slope_mb_per_s":       round(s_dur, 8),
            "slope_mb_per_min":     round(s_dur * 60, 6),
            "baseline_slope_mb_per_s": round(s_base, 8) if s_base else None,
            "verdict":              verdict,
        },
    )
