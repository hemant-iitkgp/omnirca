"""
Shared base types and helper functions for all tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np

from ..config import BASELINE_MINUTES, BASELINE_SKIP_MINUTES


# ── ToolResult ────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """
    The common return type for every tool.

    summary  — short text the LLM gets in its context window.
    data     — full structured result (DataFrame, dict, scalar …).
                Stored in KV store by the ToolExecutor (Phase 4).
    kv_key   — populated by ToolExecutor after storage; empty during Phase 1 testing.
    """
    summary: str
    data: Any
    kv_key: str = field(default="")

    def __str__(self) -> str:
        if self.kv_key:
            return f"{self.summary}\n[Full data: {self.kv_key}]"
        return self.summary


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def to_ts(t: "str | pd.Timestamp") -> pd.Timestamp:
    """Accept a string or Timestamp; always return a Timestamp."""
    return pd.Timestamp(t) if not isinstance(t, pd.Timestamp) else t


def get_baseline_window(t_start: "str | pd.Timestamp") -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Returns (baseline_start, baseline_end) by going BASELINE_MINUTES back from
    t_start, skipping the final BASELINE_SKIP_MINUTES to avoid contamination.

    Example with defaults (BASELINE_MINUTES=60, BASELINE_SKIP_MINUTES=5):
      fault at 03:23 → baseline is 02:18–03:18
    """
    t = to_ts(t_start)
    baseline_end   = t - pd.Timedelta(minutes=BASELINE_SKIP_MINUTES)
    baseline_start = t - pd.Timedelta(minutes=BASELINE_MINUTES)
    return baseline_start, baseline_end


# ── Statistical helpers ────────────────────────────────────────────────────────

def compute_z_score(
    during: pd.Series,
    baseline: pd.Series,
    min_baseline_rows: int = 3,
) -> float:
    """
    |z| = |mean(during) - mean(baseline)| / std(baseline).
    Returns 0.0 when data is insufficient or variance is near zero.
    A large z-score (> ANOMALY_THRESHOLD) signals a detection.
    """
    if len(baseline) < min_baseline_rows or len(during) < 1:
        return 0.0
    b_std = baseline.std()
    if b_std < 0.001:
        return 0.0
    return float(abs(during.mean() - baseline.mean()) / b_std)


def compute_ratio(
    during: pd.Series,
    baseline: pd.Series,
    min_baseline_rows: int = 3,
) -> float:
    """
    during_mean / baseline_mean.
    Use this for columns where std ≈ 0 (e.g. syscall_error_rate baseline is always ~0.01).
    Returns 1.0 (no change) when data is insufficient.
    """
    if len(baseline) < min_baseline_rows or len(during) < 1:
        return 1.0
    b_mean = baseline.mean()
    if abs(b_mean) < 1e-9:
        return 1.0
    return float(during.mean() / b_mean)


def memory_slope_mb_per_s(timestamps: pd.Series, memory_mb: pd.Series) -> float:
    """
    Linear regression slope of memory_mb over time (MB/s).
    Positive = growing (leak candidate).  Returns 0.0 on insufficient data.
    """
    if len(timestamps) < 5:
        return 0.0
    t0 = timestamps.iloc[0]
    times = (timestamps - t0).dt.total_seconds().values
    if times[-1] < 1:
        return 0.0
    return float(np.polyfit(times, memory_mb.values, 1)[0])


def severity_label(score: float) -> str:
    from ..config import SEVERITY_HIGH, SEVERITY_MEDIUM
    if score >= SEVERITY_HIGH:
        return "HIGH"
    if score >= SEVERITY_MEDIUM:
        return "MEDIUM"
    return "LOW"


# ── Tool: read_kv ─────────────────────────────────────────────────────────────

def read_kv(key: str) -> "ToolResult":
    """
    Retrieve full data stored in the KV store by key (kv://... URI or raw key).
    Use this when a prior tool returned a kv_key and you need the full DataFrame or dict.
    """
    from ..data_layer.kv_store import get_store
    clean_key = key.replace("kv://", "")
    value = get_store().get(clean_key)
    if value is None:
        return ToolResult(
            summary=f"read_kv: no data found for key '{key}'",
            data=None,
        )
    type_str = type(value).__name__
    size_str = f"{len(value)} rows" if hasattr(value, "__len__") else ""
    return ToolResult(
        summary=f"read_kv({key}): retrieved {type_str} {size_str}",
        data=value,
    )
