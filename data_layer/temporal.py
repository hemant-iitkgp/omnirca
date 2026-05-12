"""
Temporal anomaly analysis for OmniRCA Phase 2.

Detects WHEN (to the nearest minute) a service's syscall latency first crossed
the anomaly threshold within a fault window.

Key principle: the service that became anomalous EARLIEST is the most likely
root cause.  Services that become anomalous later are likely downstream victims.

Relies exclusively on syscalls.csv (the only high-SNR signal in this dataset).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ..config import ANOMALY_THRESHOLD, BASELINE_MINUTES, BASELINE_SKIP_MINUTES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_ts(t: "str | pd.Timestamp") -> pd.Timestamp:
    return pd.Timestamp(t) if not isinstance(t, pd.Timestamp) else t


def _get_baseline_window(t_start: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    bs = t_start - pd.Timedelta(minutes=BASELINE_MINUTES)
    be = t_start - pd.Timedelta(minutes=BASELINE_SKIP_MINUTES)
    return bs, be


# ── Public API ────────────────────────────────────────────────────────────────

def detect_anomaly_onset(
    service: str,
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> Optional[pd.Timestamp]:
    """
    Scan 1-minute rows in [t_start, t_end] for `service`.
    Returns the timestamp of the FIRST minute where:
        |avg_duration_us - baseline_mean| / baseline_std > ANOMALY_THRESHOLD

    Returns None if the service is never anomalous in the window (or data missing).
    """
    from .loader import get_loader   # lazy import to avoid circular dependency

    t_start = _to_ts(t_start)
    t_end   = _to_ts(t_end)
    bs, be  = _get_baseline_window(t_start)

    loader  = get_loader()
    sc      = loader.syscalls
    svc_sc  = sc[sc["service_name"] == service].sort_values("timestamp")

    base_rows  = svc_sc[(svc_sc["timestamp"] >= bs)      & (svc_sc["timestamp"] < be)]
    fault_rows = svc_sc[(svc_sc["timestamp"] >= t_start) & (svc_sc["timestamp"] <= t_end)]

    if len(base_rows) < 3 or fault_rows.empty:
        return None

    b_mean = base_rows["avg_duration_us"].mean()
    b_std  = base_rows["avg_duration_us"].std()
    if b_std < 0.001:
        return None

    for _, row in fault_rows.iterrows():
        z = abs(row["avg_duration_us"] - b_mean) / b_std
        if z > ANOMALY_THRESHOLD:
            return row["timestamp"]

    return None


def rank_services_by_onset(
    service_list: list[str],
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> list[dict]:
    """
    For each service in service_list compute anomaly onset.

    Returns a list of dicts sorted by onset_ts ascending (earliest first).
    Services with no detected onset are appended at the end.

    Each entry:
        service      — service name
        onset_ts     — pd.Timestamp | None
        z_at_onset   — z-score at onset row (0.0 if not detected)
        lead_minutes — minutes between onset and t_end (how early the anomaly started)
        detected     — bool
    """
    from .loader import get_loader   # lazy import

    t_start = _to_ts(t_start)
    t_end   = _to_ts(t_end)
    bs, be  = _get_baseline_window(t_start)

    loader = get_loader()
    sc     = loader.syscalls

    results: list[dict] = []

    for svc in service_list:
        onset = detect_anomaly_onset(svc, t_start, t_end)

        z_at_onset   = 0.0
        lead_minutes = 0.0

        if onset is not None:
            lead_minutes = (t_end - onset).total_seconds() / 60.0
            # Compute z at onset row
            svc_sc     = sc[sc["service_name"] == svc]
            base_rows  = svc_sc[(svc_sc["timestamp"] >= bs) & (svc_sc["timestamp"] < be)]
            onset_row  = svc_sc[svc_sc["timestamp"] == onset]["avg_duration_us"]
            if len(base_rows) >= 3 and not onset_row.empty:
                b_std = base_rows["avg_duration_us"].std()
                b_mean = base_rows["avg_duration_us"].mean()
                if b_std > 0.001:
                    z_at_onset = abs(onset_row.iloc[0] - b_mean) / b_std

        results.append({
            "service":      svc,
            "onset_ts":     onset,
            "z_at_onset":   round(z_at_onset, 2),
            "lead_minutes": round(lead_minutes, 1),
            "detected":     onset is not None,
        })

    detected     = [r for r in results if r["detected"]]
    not_detected = [r for r in results if not r["detected"]]
    detected.sort(key=lambda r: r["onset_ts"])
    return detected + not_detected
