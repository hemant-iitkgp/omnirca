"""
Graph tools — static (architecture.pkl), dynamic (traces), and fused graph operations.

Phase 1: static graph only for propagation analysis.
Phase 2: fused graph (static + dynamic evidence) with temporal-onset scoring.

Architecture facts (inspect_arch.py + topology analysis):
  - 20 integer nodes (0-19), 68 directed edges in static graph
  - Edge direction: A → B means "A calls B" (A depends on B)
  - auth_service (id=18): called by ALL 18 non-logging services — system-wide fan-in
  - logging (id=19): called by everyone — NEVER a root cause (in NEVER_ROOT_CAUSE)
  - cache_1 (id=15): ZERO incoming edges — isolated, never appears in propagation paths

Phase 2 additions:
  - get_propagation_candidates: upgraded to fused graph + temporal_bonus
  - build_call_path: optional fused-graph pathfinding via t_start/t_end args
  - get_fused_graph_summary: agent-readable fused graph statistics
  - detect_causal_order: rank services by anomaly onset timestamp
"""
from __future__ import annotations

from typing import List, Optional
import pandas as pd
import networkx as nx

from ..config import NEVER_ROOT_CAUSE, ANOMALY_THRESHOLD, TEMPORAL_BONUS
from ..data_layer.loader import get_loader
from ..data_layer.graph_loader import get_arch_graph, get_fused_graph
from .base import ToolResult, to_ts, get_baseline_window, compute_z_score


# ── Tool 1: get_service_dependencies ─────────────────────────────────────────

def get_service_dependencies(service: str) -> ToolResult:
    """
    Return the direct upstream (callers) and downstream (callees) of a service
    in the static architecture graph.

    Uses name ↔ integer-id translation so callers can use service names.
    """
    loader = get_loader()
    G      = get_arch_graph()
    svc_id, svc_name = loader.resolve_service(service)

    if svc_id not in G:
        return ToolResult(
            summary=f"get_service_dependencies({svc_name}): "
                    f"node {svc_id} not in architecture graph.",
            data={},
        )

    callers  = [loader.id_to_name.get(n, str(n)) for n in G.predecessors(svc_id)]
    callees  = [loader.id_to_name.get(n, str(n)) for n in G.successors(svc_id)]

    summary = (
        f"get_service_dependencies({svc_name}):\n"
        f"  Callers  ({len(callers)}): {', '.join(sorted(callers)) or 'none'}\n"
        f"  Callees  ({len(callees)}): {', '.join(sorted(callees)) or 'none'}"
    )
    return ToolResult(
        summary=summary,
        data={
            "service":  svc_name,
            "callers":  callers,
            "callees":  callees,
            "in_degree":  G.in_degree(svc_id),
            "out_degree": G.out_degree(svc_id),
        },
    )


# ── Tool 2: build_call_path ───────────────────────────────────────────────────

def build_call_path(
    src: str,
    dst: str,
    t_start: "Optional[str | pd.Timestamp]" = None,
    t_end:   "Optional[str | pd.Timestamp]" = None,
) -> ToolResult:
    """
    Find all simple call paths from src to dst.

    If t_start and t_end are provided, uses the FUSED graph (static + dynamic
    trace evidence); otherwise falls back to the static architecture graph.
    Returns at most 5 paths, sorted shortest first.
    """
    loader      = get_loader()
    _, src_name = loader.resolve_service(src)
    _, dst_name = loader.resolve_service(dst)

    use_fused = t_start is not None and t_end is not None

    if use_fused:
        t_start = to_ts(t_start)
        t_end   = to_ts(t_end)
        G_f     = get_fused_graph(t_start, t_end)   # service-name nodes
        try:
            paths_raw = list(nx.all_simple_paths(G_f, src_name, dst_name, cutoff=6))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            paths_raw = []
        graph_label = "fused"
        paths_named = [list(p) for p in paths_raw[:5]]
    else:
        G      = get_arch_graph()   # int-node static graph
        src_id = loader.name_to_id.get(src_name)
        dst_id = loader.name_to_id.get(dst_name)
        if src_id is None or dst_id is None:
            return ToolResult(
                summary=f"build_call_path: unknown service '{src}' or '{dst}'.",
                data={"paths": []},
            )
        try:
            paths_raw = list(nx.all_simple_paths(G, src_id, dst_id, cutoff=6))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            paths_raw = []
        paths_raw.sort(key=len)
        paths_named = [
            [loader.id_to_name.get(n, str(n)) for n in p]
            for p in paths_raw[:5]
        ]
        graph_label = "static"

    if not paths_named:
        return ToolResult(
            summary=(f"build_call_path({src_name} → {dst_name}) [{graph_label}]: "
                     f"NO path found."),
            data={"paths": []},
        )

    lines = [
        f"build_call_path({src_name} → {dst_name}) [{graph_label}] "
        f"— {len(paths_named)} path(s):"
    ]
    for i, p in enumerate(paths_named, 1):
        lines.append(f"  {i}. {' → '.join(p)}")

    return ToolResult(
        summary="\n".join(lines),
        data={"src": src_name, "dst": dst_name, "graph": graph_label,
              "paths": paths_named},
    )


# ── Tool 3: get_dynamic_graph ─────────────────────────────────────────────────

def get_dynamic_graph(
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> ToolResult:
    """
    Build a dynamic call graph from actual trace spans in the given window.

    Algorithm: join each span to its parent span → parent_service → child_service edge.
    Edge weight = call count in the window.

    NOTE (Phase 2): This graph will be fused with the static architecture graph.
    For now (Phase 1), it is returned standalone for inspection and comparison.
    """
    loader  = get_loader()
    t_start = to_ts(t_start)
    t_end   = to_ts(t_end)

    tr = loader.traces
    window_spans = tr[
        (tr["start_time"] >= t_start) &
        (tr["start_time"] <= t_end)
    ]

    if window_spans.empty:
        return ToolResult(
            summary=f"get_dynamic_graph({t_start.strftime('%H:%M')}–"
                    f"{t_end.strftime('%H:%M')}): no trace spans in window.",
            data={"edges": [], "graph": nx.DiGraph()},
        )

    # Build span_id → service_name map (from all traces, not just window,
    # to handle parent spans that started slightly before the window)
    span_to_svc = tr.set_index("span_id")["service_name"].to_dict()

    G = nx.DiGraph()
    edge_counts: dict[tuple, int] = {}

    for _, row in window_spans.iterrows():
        child_svc = row["service_name"]
        parent_id = row.get("parent_span_id")
        if pd.isna(parent_id):
            continue
        parent_svc = span_to_svc.get(str(parent_id))
        if parent_svc is None or parent_svc == child_svc:
            continue
        key = (parent_svc, child_svc)
        edge_counts[key] = edge_counts.get(key, 0) + 1

    for (u, v), w in edge_counts.items():
        G.add_edge(u, v, weight=w)

    lines = [
        f"get_dynamic_graph({t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}):",
        f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}",
        f"  (static arch has 20 nodes, 68 edges for reference)",
        f"  Top edges by call count:",
    ]
    top_edges = sorted(edge_counts.items(), key=lambda x: -x[1])[:8]
    for (u, v), w in top_edges:
        lines.append(f"    {u} → {v}  ({w} calls)")

    return ToolResult(
        summary="\n".join(lines),
        data={"edges": list(edge_counts.items()), "graph": G},
    )


# ── Tool 4: get_propagation_candidates ───────────────────────────────────────

def get_propagation_candidates(
    affected_service: str,
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> ToolResult:
    """
    Phase 2 implementation — fused graph + temporal-onset scoring.

    For each service A (not the affected service, not NEVER_ROOT_CAUSE):
      1. A must be anomalous (syscall z-score > ANOMALY_THRESHOLD).
      2. A must have a path TO affected_service in the fused graph (A calls … calls affected).
      3. path_prob  = product of fused_weight values along the shortest high-weight path.
      4. temporal_bonus = TEMPORAL_BONUS(+20%) if A's first anomalous minute precedes
                          affected_service's first anomalous minute by ≥ 1 min.
      5. final_score = z_score(A) × path_prob × (1 + temporal_bonus).

    Edge semantics: A → B means "A calls B".  Predecessors of B = callers of B.
    Temporal bonus: A became anomalous first → A may have overloaded / triggered B.
    """
    from ..data_layer.temporal import detect_anomaly_onset as _onset

    loader   = get_loader()
    t_start  = to_ts(t_start)
    t_end    = to_ts(t_end)
    _, svc_name = loader.resolve_service(affected_service)
    bs, be   = get_baseline_window(t_start)
    sc       = loader.syscalls

    G        = get_fused_graph(t_start, t_end)  # service-name nodes
    aff_onset = _onset(svc_name, t_start, t_end)

    candidates: dict[str, dict] = {}

    for node in G.nodes():
        if node == svc_name or node in NEVER_ROOT_CAUSE:
            continue
        if not nx.has_path(G, node, svc_name):
            continue

        # Compute syscall z-score
        svc_sc    = sc[sc["service_name"] == node]
        base_rows = svc_sc[(svc_sc["timestamp"] >= bs)      & (svc_sc["timestamp"] < be)]
        dur_rows  = svc_sc[(svc_sc["timestamp"] >= t_start) & (svc_sc["timestamp"] <= t_end)]
        z_score   = compute_z_score(dur_rows["avg_duration_us"], base_rows["avg_duration_us"])

        if z_score < ANOMALY_THRESHOLD:
            continue   # skip services that are not themselves anomalous

        # Shortest path weighted by inverse fused_weight (higher weight = stronger link)
        try:
            path = nx.shortest_path(
                G, node, svc_name,
                weight=lambda u, v, d: 1.0 / max(d.get("fused_weight", 0.5), 1e-6),
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        # path_prob = product of fused_weights along path
        path_prob = 1.0
        for i in range(len(path) - 1):
            path_prob *= G[path[i]][path[i + 1]].get("fused_weight", 0.5)
        hop_count = len(path) - 1

        # Temporal bonus: did A become anomalous ≥1 min before affected_service?
        onset_a = _onset(node, t_start, t_end)
        t_bonus = 0.0
        if onset_a is not None and aff_onset is not None:
            if onset_a + pd.Timedelta(minutes=1) <= aff_onset:
                t_bonus = TEMPORAL_BONUS
        elif onset_a is not None and aff_onset is None:
            t_bonus = TEMPORAL_BONUS   # A anomalous but affected service is not → A is suspect

        final_score = z_score * path_prob * (1.0 + t_bonus)

        if node not in candidates or candidates[node]["final_score"] < final_score:
            candidates[node] = {
                "service":        node,
                "hops":           hop_count,
                "z_score":        round(z_score, 2),
                "path_prob":      round(path_prob, 4),
                "temporal_bonus": t_bonus,
                "onset_ts":       str(onset_a) if onset_a else None,
                "final_score":    round(final_score, 3),
                "path":           list(path),
            }

    if not candidates:
        return ToolResult(
            summary=(f"get_propagation_candidates({svc_name}, "
                     f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}): "
                     f"no anomalous upstream candidates found "
                     f"(service may be the root cause)."),
            data=[],
        )

    ranked = sorted(candidates.values(), key=lambda r: -r["final_score"])
    lines  = [
        f"get_propagation_candidates({svc_name}, fused+temporal, "
        f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}):",
        f"  {'Service':<22} {'Hops':>5} {'z':>7} {'path_p':>8} {'bonus':>6} {'score':>8}",
        "  " + "-" * 62,
    ]
    for r in ranked[:8]:
        bonus_str = f"+{r['temporal_bonus']:.0%}" if r["temporal_bonus"] > 0 else ""
        lines.append(
            f"  {r['service']:<22} "
            f"{r['hops']:>5} "
            f"{r['z_score']:>7.2f} "
            f"{r['path_prob']:>8.4f} "
            f"{bonus_str:>6} "
            f"{r['final_score']:>8.3f}"
        )

    return ToolResult(summary="\n".join(lines), data=ranked)


# ── Tool 6: get_fused_graph_summary ──────────────────────────────────────────

def get_fused_graph_summary(
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> ToolResult:
    """
    Build (or retrieve from cache) the fused graph for the given window and
    return a human-readable structural summary.

    Reports:
      - Total node / edge counts
      - How many edges are static-only, dynamic-only, or both
      - Strongest fused edges (highest fused_weight)
    """
    t_start = to_ts(t_start)
    t_end   = to_ts(t_end)
    G       = get_fused_graph(t_start, t_end)

    static_only  = sum(1 for _, _, d in G.edges(data=True)
                       if d.get("static_weight", 0) > 0 and d.get("dynamic_weight", 0) == 0)
    dynamic_only = sum(1 for _, _, d in G.edges(data=True)
                       if d.get("static_weight", 0) == 0 and d.get("dynamic_weight", 0) > 0)
    both         = G.number_of_edges() - static_only - dynamic_only

    top_edges = sorted(G.edges(data=True),
                       key=lambda e: -e[2].get("fused_weight", 0))[:8]

    lines = [
        f"get_fused_graph_summary("
        f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}):",
        f"  Nodes: {G.number_of_nodes()}   Edges: {G.number_of_edges()}",
        f"  Static-only: {static_only}   Dynamic-only: {dynamic_only}   Both: {both}",
        f"  Strongest fused edges:",
    ]
    for u, v, d in top_edges:
        fw = d.get("fused_weight", 0)
        sw = d.get("static_weight", 0)
        dw = d.get("dynamic_weight", 0)
        lines.append(f"    {u:<22} → {v:<22}  fused={fw:.4f}  "
                     f"(static={sw:.3f}, dyn={dw:.3f})")

    return ToolResult(
        summary="\n".join(lines),
        data={
            "nodes":        list(G.nodes()),
            "edge_count":   G.number_of_edges(),
            "static_only":  static_only,
            "dynamic_only": dynamic_only,
            "both":         both,
        },
    )


# ── Tool 7: detect_causal_order ───────────────────────────────────────────────

def detect_causal_order(
    service_list: List[str],
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> ToolResult:
    """
    Rank services by the timestamp at which they first became anomalous.

    Earlier onset → more likely to be the root cause of downstream effects.
    NEVER_ROOT_CAUSE services are silently excluded.

    Returns a ToolResult whose data is the list from rank_services_by_onset,
    sorted earliest-onset first.
    """
    from ..data_layer.temporal import rank_services_by_onset

    t_start  = to_ts(t_start)
    t_end    = to_ts(t_end)
    eligible = [s for s in service_list if s not in NEVER_ROOT_CAUSE]
    ranked   = rank_services_by_onset(eligible, t_start, t_end)

    lines = [
        f"detect_causal_order("
        f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}):",
        f"  {'Service':<22} {'onset_ts':>20} {'z@onset':>8} {'lead_min':>9}",
        "  " + "-" * 65,
    ]
    for r in ranked[:10]:
        ts_str = r["onset_ts"].strftime("%H:%M:%S") if r["onset_ts"] else "not detected"
        lines.append(
            f"  {r['service']:<22} {ts_str:>20} "
            f"{r['z_at_onset']:>8.2f} {r['lead_minutes']:>9.1f}"
        )

    return ToolResult(summary="\n".join(lines), data=ranked)


# ── Tool 5 (renumbered): cross_correlate_services ────────────────────────────

def cross_correlate_services(
    service_list: List[str],
    metric: str,
    t_start: "str | pd.Timestamp",
    t_end:   "str | pd.Timestamp",
) -> ToolResult:
    """
    Compute the Pearson correlation matrix of a given metric's time series
    across a set of services in the fault window.

    `metric` may be any numeric column in metrics.csv (e.g. "cpu_percent",
    "latency_ms", "memory_mb") or in syscalls.csv (e.g. "avg_duration_us").

    High correlation between services suggests fault propagation along that axis.
    """
    loader  = get_loader()
    t_start = to_ts(t_start)
    t_end   = to_ts(t_end)

    # Resolve all service names (validates them)
    resolved = [loader.resolve_service(s)[1] for s in service_list]

    # Decide which DataFrame to use
    m  = loader.metrics
    sc = loader.syscalls

    if metric in m.columns:
        source = m
        time_col = "timestamp"
    elif metric in sc.columns:
        source = sc
        time_col = "timestamp"
    else:
        return ToolResult(
            summary=f"cross_correlate_services: metric '{metric}' not found in "
                    f"metrics.csv or syscalls.csv.",
            data={},
        )

    window = source[
        (source[time_col] >= t_start) &
        (source[time_col] <= t_end)
    ]

    series_dict: dict[str, pd.Series] = {}
    for svc in resolved:
        svc_rows = window[window["service_name"] == svc].set_index(time_col)[metric]
        if not svc_rows.empty:
            series_dict[svc] = svc_rows

    if len(series_dict) < 2:
        return ToolResult(
            summary=f"cross_correlate_services: fewer than 2 services had data for "
                    f"'{metric}' in window — cannot compute correlation.",
            data={},
        )

    df_wide = pd.DataFrame(series_dict).sort_index()
    # Forward-fill minor gaps then drop any remaining NaN rows
    df_wide = df_wide.ffill().dropna()

    if len(df_wide) < 3:
        return ToolResult(
            summary=f"cross_correlate_services: not enough aligned rows after "
                    f"fill/drop — cannot compute correlation.",
            data={},
        )

    corr = df_wide.corr().round(3)

    # Find highest off-diagonal correlations
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            svc_a = corr.columns[i]
            svc_b = corr.columns[j]
            r     = corr.iloc[i, j]
            pairs.append((svc_a, svc_b, r))
    pairs.sort(key=lambda x: -abs(x[2]))

    lines = [
        f"cross_correlate_services({metric}, "
        f"{t_start.strftime('%H:%M')}–{t_end.strftime('%H:%M')}) "
        f"— {len(df_wide)} aligned rows:",
        f"  Top correlated pairs:",
    ]
    for a, b, r in pairs[:6]:
        label = "STRONG" if abs(r) > 0.8 else "MOD" if abs(r) > 0.5 else "weak"
        lines.append(f"    {a} ↔ {b}: r={r:+.3f}  [{label}]")

    return ToolResult(
        summary="\n".join(lines),
        data={"correlation_matrix": corr, "top_pairs": pairs[:10]},
    )
