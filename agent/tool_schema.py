"""
omnirca/agent/tool_schema.py — OpenAI function-calling schemas for all 24 tools
plus a dispatcher that maps tool names to callables and invokes them.
"""
from __future__ import annotations

import json
from typing import Any

from omnirca.tools.syscall_tools import (
    query_syscalls,
    syscall_multi_service_compare,
    syscall_sub_channel_analysis,
)
from omnirca.tools.metric_tools import (
    query_metrics,
    compute_anomaly_score,
    detect_memory_slope,
)
from omnirca.tools.log_tools import query_logs, detect_error_burst
from omnirca.tools.trace_tools import query_traces, trace_fan_out, compare_trace_latency
from omnirca.tools.graph_tools import (
    get_service_dependencies,
    build_call_path,
    get_dynamic_graph,
    get_propagation_candidates,
    get_fused_graph_summary,
    detect_causal_order,
    cross_correlate_services,
)
from omnirca.tools.rca_tools import (
    search_fault_knowledge,
    classify_fault_pattern,
    check_sop,
    explain_evidence,
    finalize_rca,
)
from omnirca.tools.base import ToolResult

# Import read_kv from kv_store directly (it is re-exported as a tool)
from omnirca.data_layer.kv_store import get_store as _get_store

def _read_kv(key: str) -> ToolResult:
    val = _get_store().get(key)
    if val is None:
        return ToolResult(summary=f"read_kv('{key}'): key not found.", data={"value": None})
    return ToolResult(summary=f"read_kv('{key}'): {str(val)[:500]}", data={"value": val})


# ── Tool → callable dispatch map ─────────────────────────────────────────────

_TOOL_MAP: dict[str, Any] = {
    "query_syscalls":               query_syscalls,
    "syscall_multi_service_compare": syscall_multi_service_compare,
    "syscall_sub_channel_analysis": syscall_sub_channel_analysis,
    "query_metrics":                query_metrics,
    "compute_anomaly_score":        compute_anomaly_score,
    "detect_memory_slope":          detect_memory_slope,
    "query_logs":                   query_logs,
    "detect_error_burst":           detect_error_burst,
    "query_traces":                 query_traces,
    "trace_fan_out":                trace_fan_out,
    "compare_trace_latency":        compare_trace_latency,
    "get_service_dependencies":     get_service_dependencies,
    "build_call_path":              build_call_path,
    "get_dynamic_graph":            get_dynamic_graph,
    "get_propagation_candidates":   get_propagation_candidates,
    "get_fused_graph_summary":      get_fused_graph_summary,
    "detect_causal_order":          detect_causal_order,
    "cross_correlate_services":     cross_correlate_services,
    "search_fault_knowledge":       search_fault_knowledge,
    "classify_fault_pattern":       classify_fault_pattern,
    "check_sop":                    check_sop,
    "explain_evidence":             explain_evidence,
    "finalize_rca":                 finalize_rca,
    "read_kv":                      _read_kv,
}


def dispatch_tool(name: str, args: dict) -> ToolResult:
    """
    Dispatch a tool call by name with the given arguments dict.
    Returns a ToolResult; never raises — errors are wrapped in a ToolResult.
    """
    fn = _TOOL_MAP.get(name)
    if fn is None:
        return ToolResult(
            summary=f"ERROR: unknown tool '{name}'. "
                    f"Available: {', '.join(_TOOL_MAP.keys())}",
            data={"error": "unknown_tool", "requested": name},
        )
    try:
        return fn(**args)
    except TypeError as e:
        return ToolResult(
            summary=f"ERROR calling {name}(**{args}): {e}",
            data={"error": "bad_arguments", "tool": name, "detail": str(e)},
        )
    except Exception as e:
        return ToolResult(
            summary=f"ERROR in {name}: {type(e).__name__}: {e}",
            data={"error": str(type(e).__name__), "tool": name, "detail": str(e)},
        )


# ── Timestamp description reused across tool schemas ─────────────────────────
_TS = {"type": "string", "description": "Timestamp in format '2024-01-01 HH:MM:SS'"}
_SVC = {"type": "string", "description": "Service name (e.g. 'cache_0', 'backend_3') or integer id 0-19"}


# ── 24 OpenAI function-calling schemas ───────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [

    # ── Syscall tools ────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "query_syscalls",
            "description": (
                "Return raw syscall statistics for a single service in the time window. "
                "Columns include avg_duration_us, p99_duration_us, error_rate, "
                "total_syscalls, and per-type (read/write/socket/etc.) count/errors. "
                "Use for detailed per-service inspection after triage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": _SVC,
                    "t_start": _TS,
                    "t_end":   _TS,
                },
                "required": ["service", "t_start", "t_end"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "syscall_multi_service_compare",
            "description": (
                "MANDATORY FIRST TOOL. Ranks ALL 20 services by syscall z-score "
                "in the fault window vs. the 1-hour baseline. Returns a table with "
                "rank, service_name, z_avg_duration, z_p99_duration, error_rate_ratio, "
                "anomalous flag. Call this FIRST before any other tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "t_start": _TS,
                    "t_end":   _TS,
                },
                "required": ["t_start", "t_end"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "syscall_sub_channel_analysis",
            "description": (
                "Break down syscall anomaly by sub-channel type (read, write, open, close, "
                "socket, send, recv, fsync, mmap) for a single service. "
                "Returns count_ratio and errors_ratio per type — use these to identify "
                "the fault pattern (e.g. high read_count → stale_cache; high recv_count "
                "+ socket errors → thread_pool_exhaustion; high fsync → disk_io_saturation). "
                "NOTE: duration sub-channels all equal global duration — ignore duration per type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": _SVC,
                    "t_start": _TS,
                    "t_end":   _TS,
                },
                "required": ["service", "t_start", "t_end"],
            },
        },
    },

    # ── Metric tools ─────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "query_metrics",
            "description": (
                "Return raw metric rows for a service (cpu_percent, latency_ms, "
                "memory_mb, error_rate). WARNING: metric z-scores are near zero (0.00-0.07) "
                "for most faults in this dataset. Use primarily for memory_mb trend "
                "(memory leak detection). Not useful for fault detection alone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": _SVC,
                    "t_start": _TS,
                    "t_end":   _TS,
                },
                "required": ["service", "t_start", "t_end"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "compute_anomaly_score",
            "description": (
                "Compute a weighted composite anomaly score for a service combining "
                "syscall (dominant weight 4×), metric (0.5×), log (0.2×), and trace (0.5×) signals. "
                "Returns total_score, severity label (CRITICAL/HIGH/MEDIUM/LOW/NORMAL), "
                "and per-signal breakdown. Use after triage to quantify the top candidates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": _SVC,
                    "t_start": _TS,
                    "t_end":   _TS,
                },
                "required": ["service", "t_start", "t_end"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "detect_memory_slope",
            "description": (
                "Fit a linear regression to memory_mb to detect a memory leak. "
                "Returns slope in MB/s compared to the baseline. "
                "A positive slope (e.g. +0.005 MB/s = +5 MB/min) indicates memory leak. "
                "Use when classify_fault_pattern or search_fault_knowledge suggests memory_leak."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": _SVC,
                    "t_start": _TS,
                    "t_end":   _TS,
                },
                "required": ["service", "t_start", "t_end"],
            },
        },
    },

    # ── Log tools ────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "query_logs",
            "description": (
                "Return log entries for a service in the window. "
                "WARNING: this dataset has ZERO ERROR-level log entries. "
                "Only WARNING entries exist. Call only if you need WARNING message content "
                "for corroboration. Do not call expecting to find ERROR logs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": _SVC,
                    "t_start": _TS,
                    "t_end":   _TS,
                    "level": {
                        "type": "string",
                        "description": "Optional filter: 'ERROR', 'WARNING', 'INFO'. Default: all levels.",
                    },
                },
                "required": ["service", "t_start", "t_end"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "detect_error_burst",
            "description": (
                "Detect bursts of WARNING/ERROR log entries within the window compared "
                "to baseline frequency. Returns burst_detected flag, peak_rate, "
                "and relevant log messages. Falls back to WARNING entries in this dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": _SVC,
                    "t_start": _TS,
                    "t_end":   _TS,
                },
                "required": ["service", "t_start", "t_end"],
            },
        },
    },

    # ── Trace tools ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "query_traces",
            "description": (
                "Return raw trace spans for a service in the window. "
                "Only 10 of 20 services appear in trace data; missing ones return empty. "
                "Includes duration_ms, success flag, span_id, parent_span_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": _SVC,
                    "t_start": _TS,
                    "t_end":   _TS,
                },
                "required": ["service", "t_start", "t_end"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "trace_fan_out",
            "description": (
                "Show which downstream services a given service called in the window, "
                "with call counts and latencies. Useful for understanding propagation "
                "direction and identifying which callee caused latency."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": _SVC,
                    "t_start": _TS,
                    "t_end":   _TS,
                },
                "required": ["service", "t_start", "t_end"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "compare_trace_latency",
            "description": (
                "Compare p50 and p99 trace latency in the fault window "
                "vs. the 1-hour baseline for a single service. "
                "Returns z-score and ratio. Note: z-scores are typically LOW (≤0.5) "
                "here due to sparse trace coverage — use syscalls for primary ranking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": _SVC,
                    "t_start": _TS,
                    "t_end":   _TS,
                },
                "required": ["service", "t_start", "t_end"],
            },
        },
    },

    # ── Graph tools ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_service_dependencies",
            "description": (
                "Return the direct callers (upstream) and callees (downstream) of a "
                "service in the static architecture graph. "
                "Useful for understanding fault propagation direction."
            ),
            "parameters": {
                "type": "object",
                "properties": {"service": _SVC},
                "required": ["service"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "build_call_path",
            "description": (
                "Find the shortest call path between two services in the architecture. "
                "Optionally provide t_start/t_end to use the phase-2 fused graph "
                "(static + dynamic trace evidence combined)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "src":     _SVC,
                    "dst":     _SVC,
                    "t_start": {**_TS, "description": "Optional: use fused graph if provided."},
                    "t_end":   {**_TS, "description": "Optional: use fused graph if provided."},
                },
                "required": ["src", "dst"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_dynamic_graph",
            "description": (
                "Build a dynamic call graph from actual trace spans in the window. "
                "Shows which services called which, with call counts. "
                "Complements the static architecture graph with observed behaviour."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "t_start": _TS,
                    "t_end":   _TS,
                },
                "required": ["t_start", "t_end"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_propagation_candidates",
            "description": (
                "Phase-2 fused-graph analysis: for a given affected service, rank "
                "all UPSTREAM services as potential root causes. "
                "Scoring = syscall_z × path_probability × temporal_bonus. "
                "Excludes NEVER_ROOT_CAUSE (logging). Use after triage to narrow candidates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "affected_service": _SVC,
                    "t_start":          _TS,
                    "t_end":            _TS,
                },
                "required": ["affected_service", "t_start", "t_end"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_fused_graph_summary",
            "description": (
                "Return a summary of the fused (static + dynamic) graph built for "
                "the given time window: node count, edge count, top edges by weight. "
                "Useful for understanding which call paths were most active."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "t_start": _TS,
                    "t_end":   _TS,
                },
                "required": ["t_start", "t_end"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "detect_causal_order",
            "description": (
                "Rank a list of anomalous services by the minute at which their "
                "anomaly FIRST appeared (onset detection via syscall z-score). "
                "The service with the EARLIEST onset is most likely the root cause. "
                "Pass the top 3-5 anomalous services from your triage step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of service names to rank by first anomaly onset.",
                    },
                    "t_start": _TS,
                    "t_end":   _TS,
                },
                "required": ["service_list", "t_start", "t_end"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "cross_correlate_services",
            "description": (
                "Compute the Pearson correlation matrix of a metric across multiple "
                "services in the window. High correlation between services suggests "
                "simultaneous fault propagation. "
                "metric: any column like 'avg_duration_us', 'cpu_percent', 'memory_mb'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of service names to correlate.",
                    },
                    "metric": {
                        "type": "string",
                        "description": "Metric column name, e.g. 'avg_duration_us' or 'memory_mb'.",
                    },
                    "t_start": _TS,
                    "t_end":   _TS,
                },
                "required": ["service_list", "metric", "t_start", "t_end"],
            },
        },
    },

    # ── RCA tools ────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_fault_knowledge",
            "description": (
                "Search the RAG-indexed FAULT_ENCYCLOPEDIA for fault archetypes matching "
                "the described symptoms. Returns the top-2 most similar fault types with "
                "their category, counter_intuitive flag, and relevant text excerpt. "
                "Pass a natural-language description of what you observe."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symptom_text": {
                        "type": "string",
                        "description": (
                            "Natural-language description of observed symptoms, e.g. "
                            "'high read syscall count, latency decreased, error rate up 25%'"
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return. Default 2.",
                        "default": 2,
                    },
                },
                "required": ["symptom_text"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "classify_fault_pattern",
            "description": (
                "Classify the fault category from a signals dict using heuristic rules. "
                "Build this dict from results of syscall_sub_channel_analysis and "
                "compute_anomaly_score. "
                "Returns top_category, confidence (HIGH/MEDIUM/LOW), ranked scores, "
                "and recommended SOP key."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signals": {
                        "type": "object",
                        "description": (
                            "Optional dict with keys: 'syscall_avg_duration_z' (float), "
                            "'syscall_error_rate_ratio' (float), "
                            "'memory_slope_mb_per_s' (float, default 0.0), "
                            "'channel_stats' (object with keys 'read','write','open',"
                            "'socket','send','recv','fsync','mmap', each having "
                            "'count_ratio', 'errors_ratio', 'duration_z' floats). "
                            "If not available, call with no arguments to get a "
                            "text-based classification fallback."
                        ),
                    },
                },
                "required": [],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "check_sop",
            "description": (
                "Return the Standard Operating Procedure (SOP) for a fault category. "
                "Valid categories: stale_cache, memory_leak, thread_pool_exhaustion, "
                "transaction_deadlock, disk_io_saturation, cascading_timeout, "
                "auth_failure, data_corruption, api_version_mismatch, "
                "data_race_condition, unknown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fault_category": {
                        "type": "string",
                        "description": "Fault category key (snake_case).",
                    },
                },
                "required": ["fault_category"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "explain_evidence",
            "description": (
                "Generate a natural-language explanation of why the identified service "
                "is the root cause, synthesising evidence using the LLM. "
                "Call this BEFORE finalize_rca to produce the narrative."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "The identified root-cause service name.",
                    },
                    "fault_category": {
                        "type": "string",
                        "description": "The identified fault category (snake_case).",
                    },
                    "evidence_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of evidence strings gathered during investigation "
                            "(tool results, z-scores, sub-channel findings, etc.)"
                        ),
                    },
                },
                "required": ["service", "fault_category"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "finalize_rca",
            "description": (
                "MANDATORY LAST TOOL. Produce the final Root Cause Analysis verdict. "
                "Call this as your LAST action after gathering all evidence. "
                "Returns a formatted RCA report with root cause, confidence, evidence, "
                "propagation path, and notes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "root_cause": {
                        "type": "string",
                        "description": "Service name identified as the root cause.",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["HIGH", "MEDIUM", "LOW"],
                        "description": "Confidence level: HIGH, MEDIUM, or LOW.",
                    },
                    "evidence_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered list of evidence strings.",
                    },
                    "fault_category": {
                        "type": "string",
                        "description": "Fault category key (snake_case). Default: 'unknown'.",
                    },
                    "propagation_path": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: propagation chain from root to symptom.",
                    },
                    "affected_services": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: all anomalous services.",
                    },
                    "composite_score": {
                        "type": "number",
                        "description": "Optional: composite anomaly score of root cause service.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional: free-text notes.",
                    },
                },
                "required": ["root_cause", "confidence", "evidence_list"],
            },
        },
    },

    # ── KV store ─────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "read_kv",
            "description": (
                "Read a value from the KV store by key. "
                "The KV store caches intermediate computation results. "
                "Useful if a prior tool call saved a result you want to retrieve."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "KV store key to retrieve.",
                    },
                },
                "required": ["key"],
            },
        },
    },
]
