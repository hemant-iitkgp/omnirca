"""
Syscall tools — the primary detection and localization layer.

All three tools use syscalls.csv which covers ALL 20 services at 1-minute intervals.
Confirmed signal strength (from assess.py):
  avg_duration_us z-score = 30–70 for every one of the 10 faults
  p99_duration_us z-score = 48–70
  error_rate: baseline ~0.010, fault window ~0.150 (15× absolute ratio)
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from ..config import ANOMALY_THRESHOLD, SIGNAL_WEIGHTS, PRIMARY_CHANNELS
from ..data_layer.loader import get_loader
from .base import (
    ToolResult,
    to_ts,
    get_baseline_window,
    compute_z_score,
    compute_ratio,
)

# Sub-channel syscall types (confirmed from prebuild_check2.py)
_SYSCALL_TYPES = [
    "read", "write", "open", "close",
    "socket", "send", "recv",
    "fsync", "mmap",
]


# ── Tool 1: query_syscalls ─────────────────────────────────────────────────────

def query_syscalls(
    service: str,
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> ToolResult:
    """
    Return raw syscall statistics for a single service in the given time window.
    Columns: avg_duration_us, p99_duration_us, error_rate, total_syscalls,
             syscall_{type}_count/errors/avg_duration for 9 types.
    """
    loader  = get_loader()
    t_start = to_ts(t_start)
    t_end   = to_ts(t_end)

    loader.resolve_service(service)   # validate name
    sc = loader.syscalls
    rows = sc[
        (sc["service_name"] == service) &
        (sc["timestamp"] >= t_start) &
        (sc["timestamp"] <= t_end)
    ].copy()

    if rows.empty:
        return ToolResult(
            summary=f"query_syscalls({service}): NO data in window "
                    f"{t_start}–{t_end}",
            data=rows,
        )

    avg_dur = rows["avg_duration_us"].mean()
    p99_dur = rows["p99_duration_us"].mean()
    err_rt  = rows["error_rate"].mean()
    summary = (
        f"query_syscalls({service}, "
        f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}) "
        f"— {len(rows)} rows:\n"
        f"  avg_duration_us={avg_dur:.1f}  p99_duration_us={p99_dur:.1f}  "
        f"error_rate={err_rt:.4f}"
    )
    return ToolResult(summary=summary, data=rows)


# ── Tool 2: syscall_multi_service_compare ─────────────────────────────────────

def syscall_multi_service_compare(
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> ToolResult:
    """
    MANDATORY FIRST TOOL. Ranks all 20 services by syscall avg_duration_us z-score
    in the fault window vs. the 1-hour preceding baseline.

    Returns a DataFrame with columns:
      rank, service_name, z_avg_duration, z_p99_duration,
      error_rate_ratio, baseline_mean_us, during_mean_us,
      during_rows, baseline_rows, anomalous
    """
    loader  = get_loader()
    t_start = to_ts(t_start)
    t_end   = to_ts(t_end)
    bs, be  = get_baseline_window(t_start)
    sc      = loader.syscalls

    results = []
    for svc in loader.service_names:
        svc_sc    = sc[sc["service_name"] == svc]
        baseline  = svc_sc[(svc_sc["timestamp"] >= bs)      & (svc_sc["timestamp"] < be)]
        during    = svc_sc[(svc_sc["timestamp"] >= t_start) & (svc_sc["timestamp"] <= t_end)]

        z_avg = compute_z_score(during["avg_duration_us"], baseline["avg_duration_us"])
        z_p99 = compute_z_score(during["p99_duration_us"], baseline["p99_duration_us"])
        err_ratio = compute_ratio(during["error_rate"], baseline["error_rate"])

        # Strongest anomaly across the whole primary panel.  On datasets_complex
        # the panel is just the two duration columns, so this equals z_avg and
        # ranking is unchanged; on multi-channel datasets it prevents a fault
        # that shows up in (say) CPU from being missed by a latency-only rank.
        z_by_channel: dict[str, float] = {}
        for chan in PRIMARY_CHANNELS:
            if chan in svc_sc.columns:
                z_by_channel[chan] = round(
                    compute_z_score(during[chan], baseline[chan]), 2)
        z_max = max(z_by_channel.values(), default=z_avg)
        top_chan = max(z_by_channel, key=z_by_channel.get) if z_by_channel else "avg_duration_us"

        results.append({
            "service_name":     svc,
            "z_max_channel":    round(z_max, 2),
            "top_channel":      top_chan,
            "z_avg_duration":   round(z_avg, 2),
            "z_p99_duration":   round(z_p99, 2),
            "error_rate_ratio": round(err_ratio, 2),
            "baseline_mean_us": round(baseline["avg_duration_us"].mean(), 2)
                                if len(baseline) > 0 else 0.0,
            "during_mean_us":   round(during["avg_duration_us"].mean(), 2)
                                if len(during) > 0 else 0.0,
            "during_rows":      len(during),
            "baseline_rows":    len(baseline),
            "anomalous":        z_max > ANOMALY_THRESHOLD,
        })

    df = (
        pd.DataFrame(results)
        .sort_values("z_max_channel", ascending=False)
        .reset_index(drop=True)
    )
    df.index = df.index + 1   # 1-based rank
    df.index.name = "rank"

    # ── summary ───
    lines = [
        f"syscall_multi_service_compare "
        f"({t_start.strftime('%Y-%m-%d %H:%M')}–{t_end.strftime('%H:%M')}):",
        f"Baseline: {bs.strftime('%H:%M')}–{be.strftime('%H:%M')}  |  "
        f"Anomaly threshold: z > {ANOMALY_THRESHOLD}",
        "",
        f"{'Rank':<5} {'Service':<22} {'z_max':>7} {'channel':>14} {'z_avg':>7} "
        f"{'z_p99':>7} {'err×':>6} {'anomalous':>10}",
        "-" * 88,
    ]
    for rank, row in df.head(20).iterrows():
        flag = " <<< ANOMALOUS" if row["anomalous"] else ""
        lines.append(
            f"{rank:<5} {row['service_name']:<22} "
            f"{row['z_max_channel']:>7.2f} {row['top_channel']:>14} "
            f"{row['z_avg_duration']:>7.2f} {row['z_p99_duration']:>7.2f} "
            f"{row['error_rate_ratio']:>6.2f}"
            f"{flag}"
        )

    anomalous = df[df["anomalous"]]
    lines.append("")
    lines.append(
        f"Anomalous services ({len(anomalous)}): "
        + ", ".join(anomalous["service_name"].tolist())
    )

    return ToolResult(summary="\n".join(lines), data=df)


# ── Tool 3: syscall_sub_channel_analysis ─────────────────────────────────────

def syscall_sub_channel_analysis(
    service: str,
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> ToolResult:
    """
    Breaks down syscall sub-channels (read/write/fsync/mmap/socket/send/recv/open/close)
    to attempt fault-type discrimination after syscall_multi_service_compare localizes
    the anomalous service.

    For each syscall type returns:
      - count_ratio  (during / baseline counts)
      - errors_ratio (during / baseline error counts)
      - duration_z   (z-score of avg_duration per type)

    Appends a classification hint based on the sub-channel discrimination table
    from plan Section 1.4.
    """
    loader  = get_loader()
    t_start = to_ts(t_start)
    t_end   = to_ts(t_end)
    bs, be  = get_baseline_window(t_start)

    loader.resolve_service(service)
    sc     = loader.syscalls
    svc_sc = sc[sc["service_name"] == service]
    base   = svc_sc[(svc_sc["timestamp"] >= bs)      & (svc_sc["timestamp"] < be)]
    during = svc_sc[(svc_sc["timestamp"] >= t_start) & (svc_sc["timestamp"] <= t_end)]

    if len(base) < 3 or len(during) < 1:
        return ToolResult(
            summary=f"syscall_sub_channel_analysis({service}): "
                    f"Insufficient data — {len(during)} during rows, {len(base)} baseline rows.",
            data={},
        )

    channel_stats = {}
    rows = []
    for sc_type in _SYSCALL_TYPES:
        cnt_col  = f"syscall_{sc_type}_count"
        err_col  = f"syscall_{sc_type}_errors"
        dur_col  = f"syscall_{sc_type}_avg_duration"

        # Some columns may not exist (tolerate gracefully)
        cnt_ratio  = compute_ratio(during[cnt_col],  base[cnt_col])  if cnt_col  in sc.columns else 1.0
        err_ratio  = compute_ratio(during[err_col],  base[err_col])  if err_col  in sc.columns else 1.0
        dur_z      = compute_z_score(during[dur_col], base[dur_col]) if dur_col  in sc.columns else 0.0

        channel_stats[sc_type] = {
            "count_ratio":  round(cnt_ratio, 2),
            "errors_ratio": round(err_ratio, 2),
            "duration_z":   round(dur_z, 2),
        }
        rows.append({
            "syscall_type": sc_type,
            "count_ratio":  round(cnt_ratio, 2),
            "errors_ratio": round(err_ratio, 2),
            "duration_z":   round(dur_z, 2),
        })

    df = pd.DataFrame(rows).sort_values("duration_z", ascending=False).reset_index(drop=True)

    # ── Fault-type hint based on sub-channel discrimination table (plan §1.4) ─
    hint = _classify_sub_channels(channel_stats)

    # ── summary ───
    lines = [
        f"syscall_sub_channel_analysis({service}, "
        f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}):",
        f"  {'Type':<8} {'cnt×':>6} {'err×':>6} {'dur_z':>7}",
        "  " + "-" * 35,
    ]
    for _, row in df.iterrows():
        prominent = "  <<<" if row["duration_z"] > ANOMALY_THRESHOLD or row["errors_ratio"] > 3 else ""
        lines.append(
            f"  {row['syscall_type']:<8} "
            f"{row['count_ratio']:>6.2f} "
            f"{row['errors_ratio']:>6.2f} "
            f"{row['duration_z']:>7.2f}"
            f"{prominent}"
        )
    lines.append("")
    lines.append(f"  Fault-type hint: {hint}")

    return ToolResult(
        summary="\n".join(lines),
        data={"channel_stats": channel_stats, "ranked_df": df, "fault_hint": hint},
    )


# ── Internal: sub-channel pattern → fault type hint ───────────────────────────

def _classify_sub_channels(stats: dict) -> str:
    """
    Rule-based heuristic that maps sub-channel patterns to candidate fault types.
    Based on the discrimination table in plan §1.4.
    Returns a human-readable string with ranked candidates.
    """
    s = stats   # alias

    def z(t):  return s.get(t, {}).get("duration_z", 0)
    def cnt(t): return s.get(t, {}).get("count_ratio", 1)
    def err(t): return s.get(t, {}).get("errors_ratio", 1)

    candidates = {}

    # disk_io_saturation: fsync + write both extreme AND read elevated
    if z("fsync") > 3 and z("write") > 3 and cnt("read") > 1.5:
        candidates["disk_io_saturation"] = z("fsync") + z("write") + cnt("read")

    # memory_leak: mmap_count increasing (ratio > 1.3)
    if cnt("mmap") > 1.3:
        candidates["memory_leak"] = cnt("mmap") * 10

    # thread_pool_exhaustion: recv_count spikes, send drops, socket errors
    if cnt("recv") > 2 and err("socket") > 2 and cnt("send") < 1.0:
        candidates["thread_pool_exhaustion"] = cnt("recv") + err("socket")

    # transaction_deadlock: write_duration extreme, read moderate
    if z("write") > 5 and z("read") < z("write") * 0.7:
        candidates["transaction_deadlock"] = z("write")

    # stale_cache: read_count HIGH, write_count LOW
    if cnt("read") > 2 and cnt("write") < 0.8:
        candidates["stale_cache"] = cnt("read")

    # cascading_timeout: socket_errors high, recv_duration spikes
    if err("socket") > 3 and z("recv") > 3:
        candidates["cascading_timeout"] = err("socket") + z("recv")

    # auth_failure: read_count spike, write/send flat
    if cnt("read") > 1.5 and cnt("write") < 1.1 and cnt("send") < 1.1:
        candidates["auth_failure"] = cnt("read")

    # data_corruption: write_errors elevated alongside duration spike
    if err("write") > 3 and z("write") > 3:
        candidates["data_corruption"] = err("write") + z("write")

    # api_version_mismatch: open_count elevated
    if cnt("open") > 2:
        candidates["api_version_mismatch"] = cnt("open")

    # data_race_condition: intense read+write contention (both elevated)
    if z("read") > 3 and z("write") > 3 and z("fsync") < 2:
        candidates["data_race_condition"] = z("read") + z("write")

    if not candidates:
        return ("No strong sub-channel match. Generic I/O spike detected. "
                "Use search_fault_knowledge with observed symptoms for classification.")

    ranked = sorted(candidates.items(), key=lambda x: -x[1])
    top = ranked[0][0]
    others = ", ".join(f[0] for f in ranked[1:3]) if len(ranked) > 1 else "—"
    return (f"Primary candidate: {top} (score={ranked[0][1]:.2f}). "
            f"Other possibilities: {others}. "
            f"Confirm with search_fault_knowledge and follow SOP.")
