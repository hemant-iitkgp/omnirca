"""
Log tools — query logs.csv and detect ERROR bursts.

Findings from assess.py:
  - logs.csv has 0 ERROR entries in all 10 fault windows (WARNING-level entries only).
  - These tools should still be called as part of every investigation — a zero-error
    result is itself informative (rules out log-level anomalies).
  - detect_error_burst falls back to WARNING-level density when ERROR count is zero.
"""
from __future__ import annotations

from typing import Optional
import pandas as pd

from ..data_layer.loader import get_loader
from .base import ToolResult, to_ts, get_baseline_window, compute_ratio


# ── Tool 1: query_logs ────────────────────────────────────────────────────────

def query_logs(
    service: str,
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
    level:   Optional[str] = None,
) -> ToolResult:
    """
    Return log rows for a single service in the given time window.

    Parameters
    ----------
    service : str
        Service name or integer id.
    t_start, t_end : str | pd.Timestamp
        Window bounds.
    level : str, optional
        If provided, filter to this log level (e.g. "ERROR", "WARNING", "INFO").
        Case-insensitive.
    """
    loader  = get_loader()
    t_start = to_ts(t_start)
    t_end   = to_ts(t_end)
    loader.resolve_service(service)

    logs = loader.logs
    mask = (
        (logs["service_name"] == service) &
        (logs["timestamp"]    >= t_start) &
        (logs["timestamp"]    <= t_end)
    )
    rows = logs[mask].copy()

    if level is not None:
        rows = rows[rows["level"].str.upper() == level.upper()]

    if rows.empty:
        level_str = f"[level={level}] " if level else ""
        return ToolResult(
            summary=f"query_logs({service}) {level_str}— NO entries in window "
                    f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}",
            data=rows,
        )

    counts = rows["level"].value_counts().to_dict()
    count_str = "  ".join(f"{k}={v}" for k, v in counts.items())
    summary = (
        f"query_logs({service}, "
        f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}) "
        f"— {len(rows)} entries: {count_str}"
    )
    return ToolResult(summary=summary, data=rows)


# ── Tool 2: detect_error_burst ────────────────────────────────────────────────

def detect_error_burst(
    service: str,
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> ToolResult:
    """
    Detects a sudden spike in log ERROR (or WARNING) density in the fault window
    compared to the 1-hour baseline.

    Algorithm:
      1. Count ERROR entries per minute in both windows.
      2. If ERROR count == 0 in both windows, fall back to WARNING entries.
      3. Use ratio (during_rate / baseline_rate) rather than z-score because
         the baseline std is typically near zero for log levels.

    A ratio > 3 is flagged as a burst.
    """
    loader  = get_loader()
    t_start = to_ts(t_start)
    t_end   = to_ts(t_end)
    bs, be  = get_baseline_window(t_start)
    loader.resolve_service(service)

    logs     = loader.logs
    svc_logs = logs[logs["service_name"] == service]

    base = svc_logs[(svc_logs["timestamp"] >= bs)      & (svc_logs["timestamp"] < be)]
    dur  = svc_logs[(svc_logs["timestamp"] >= t_start) & (svc_logs["timestamp"] <= t_end)]

    win_min  = max((t_end - t_start).total_seconds() / 60.0, 1.0)
    base_min = max((be    - bs      ).total_seconds() / 60.0, 1.0)

    def _rate(df: pd.DataFrame, lvl: str) -> float:
        return int((df["level"].str.upper() == lvl).sum()) / win_min if "level" in df.columns else 0.0

    err_d = _rate(dur,  "ERROR")
    err_b = _rate(base, "ERROR")

    # Fall back to WARNING if no ERRORs in either window
    use_level = "ERROR"
    if err_d == 0 and err_b == 0:
        err_d = _rate(dur,  "WARNING")
        err_b = _rate(base, "WARNING")
        use_level = "WARNING (ERROR fallback)"

    ratio   = (err_d / err_b) if err_b > 0.0001 else (float("inf") if err_d > 0 else 1.0)
    burst   = ratio > 3.0
    verdict = "BURST DETECTED" if burst else "NORMAL"

    summary = (
        f"detect_error_burst({service}, "
        f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}) "
        f"[level={use_level}]:\n"
        f"  During rate  : {err_d:.3f} entries/min\n"
        f"  Baseline rate: {err_b:.3f} entries/min\n"
        f"  Ratio        : {ratio:.2f}×  [{verdict}]"
    )
    return ToolResult(
        summary=summary,
        data={
            "service":          service,
            "level_used":       use_level,
            "during_rate_per_min":  round(err_d, 4),
            "baseline_rate_per_min": round(err_b, 4),
            "ratio":            round(ratio, 2) if not isinstance(ratio, float) or ratio != float("inf") else "inf",
            "burst_detected":   burst,
        },
    )
