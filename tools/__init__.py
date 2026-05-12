"""
omnirca.tools — all 24 tools available for import.

Usage:
    from omnirca.tools import syscall_multi_service_compare, compute_anomaly_score
    # or import the whole namespace:
    from omnirca import tools
    tools.syscall_multi_service_compare(...)
"""

# Syscall tools (primary signal — weight 4×)
from .syscall_tools import (
    query_syscalls,
    syscall_multi_service_compare,
    syscall_sub_channel_analysis,
)

# Metric tools
from .metric_tools import (
    query_metrics,
    compute_anomaly_score,
    detect_memory_slope,
)

# Log tools
from .log_tools import (
    query_logs,
    detect_error_burst,
)

# Trace tools
from .trace_tools import (
    query_traces,
    trace_fan_out,
    compare_trace_latency,
)

# Graph tools
from .graph_tools import (
    get_service_dependencies,
    build_call_path,
    get_dynamic_graph,
    get_propagation_candidates,
    get_fused_graph_summary,
    detect_causal_order,
    cross_correlate_services,
)

# RCA tools
from .rca_tools import (
    search_fault_knowledge,
    classify_fault_pattern,
    check_sop,
    explain_evidence,
    finalize_rca,
)

# Utility
from .base import read_kv

__all__ = [
    # syscall (3)
    "query_syscalls",
    "syscall_multi_service_compare",
    "syscall_sub_channel_analysis",
    # metric (3)
    "query_metrics",
    "compute_anomaly_score",
    "detect_memory_slope",
    # log (2)
    "query_logs",
    "detect_error_burst",
    # trace (3)
    "query_traces",
    "trace_fan_out",
    "compare_trace_latency",
    # graph (7)
    "get_service_dependencies",
    "build_call_path",
    "get_dynamic_graph",
    "get_propagation_candidates",
    "get_fused_graph_summary",
    "detect_causal_order",
    "cross_correlate_services",
    # rca (5)
    "search_fault_knowledge",
    "classify_fault_pattern",
    "check_sop",
    "explain_evidence",
    "finalize_rca",
    # utility (1)
    "read_kv",
]
