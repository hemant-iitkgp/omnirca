"""
omnirca/evaluation/stress_analysis.py — Static + live stress testing framework.

No LLM involved in ANY of the static analyses.  Live tests are isolated
in StressBattery and can be run separately.

Static analyses:
  SignalAnalyzer        — syscall z-score ranking per fault, signal quality
  GraphHopAnalyzer      — hop distance from reported service → root cause
  WindowSensitivity     — how z-score degrades with window shifts / truncation
  NormalPeriodBaseline  — z-scores during normal (non-fault) windows
  CategorySignalMap     — which fault types have distinguishable sub-channel signatures

Tiered live test definitions (no execution here — see StressBattery):
  Tier 0: Control  — exact same queries as original eval (reproducibility)
  Tier 1: Easy     — single-service faults, root = reported
  Tier 2: Fuzzy    — shifted/truncated windows (±5 min, half window)
  Tier 3: Harder   — report root directly for victim-reported faults
  Tier 4: Adversarial — report service during no-fault window / unrelated service
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO = str(Path(__file__).parent.parent.parent)

# ─── shared constants ────────────────────────────────────────────────────────
ANOMALY_THRESHOLD = 2.5   # same as config.py
BASELINE_MINUTES  = 60
BASELINE_SKIP     = 5

_ID_TO_NAME = {
    0:"frontend_0", 1:"frontend_1", 2:"api_gateway",
    3:"backend_0",  4:"backend_1",  5:"backend_2",
    6:"backend_3",  7:"backend_4",  8:"backend_5",
    9:"backend_6",  10:"backend_7", 11:"database_0",
    12:"database_1",13:"database_2",14:"cache_0",
    15:"cache_1",   16:"message_queue_0",17:"message_queue_1",
    18:"auth_service",19:"logging",
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _z(during: pd.Series, baseline: pd.Series) -> float:
    if len(baseline) < 3 or len(during) < 1:
        return 0.0
    std = baseline.std()
    if std < 0.001:
        return 0.0
    return float(abs(during.mean() - baseline.mean()) / std)


def _load_data():
    """Load syscalls.csv and faults.csv once."""
    data_dir = Path(_REPO) / "datasets_complex"
    sc = pd.read_csv(data_dir / "syscalls.csv", parse_dates=["timestamp"])
    faults = pd.read_csv(data_dir / "faults.csv", parse_dates=["start_time","end_time"])
    return sc, faults


def _baseline_window(t_start: pd.Timestamp):
    end = t_start - pd.Timedelta(minutes=BASELINE_SKIP)
    start = t_start - pd.Timedelta(minutes=BASELINE_MINUTES)
    return start, end


def _rank_all_services(sc: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp) -> pd.DataFrame:
    """Return DataFrame ranked by z_avg for all services in the given window."""
    bs, be = _baseline_window(t_start)
    rows = []
    for sid, sname in _ID_TO_NAME.items():
        svc_sc   = sc[sc["service_name"] == sname]
        baseline = svc_sc[(svc_sc["timestamp"] >= bs) & (svc_sc["timestamp"] < be)]
        during   = svc_sc[(svc_sc["timestamp"] >= t_start) & (svc_sc["timestamp"] <= t_end)]
        z = _z(during["avg_duration_us"], baseline["avg_duration_us"]) if (
            len(during) > 0 and len(baseline) > 0
        ) else 0.0
        rows.append({"service": sname, "z_avg": round(z, 2), "rows_during": len(during)})

    df = pd.DataFrame(rows).sort_values("z_avg", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 1. SignalAnalyzer
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FaultSignal:
    fault_id:       int
    fault_type:     str
    root_service:   str
    reported_service: str
    root_rank:      int           # rank of root in z-score table (1 = strongest)
    root_z:         float         # z-score of root cause service
    top1_service:   str           # who was actually ranked #1
    top1_z:         float
    num_anomalous:  int           # count of services with z > threshold
    signal_ratio:   float         # root_z / top1_z  (1.0 = root is #1)
    reported_rank:  int           # rank of REPORTED service (may differ from root)
    reported_z:     float
    window_minutes: float
    notes: str = ""


class SignalAnalyzer:
    """Compute syscall signal quality for every fault without any LLM."""

    def __init__(self):
        self._sc, self._faults = _load_data()
        # Reported service IDs per fault (from query_builder.py)
        self._reported_ids = {
            0: 18,  # auth_service (victim)
            1: 4,   # backend_1 (victim)
            2: 12,  # database_1 (victim)
            3: 18,  # auth_service (root)
            4: 10,  # backend_7 (root)
            5: 18,  # auth_service (victim)
            6: 7,   # backend_4 (root)
            7: 3,   # backend_0 (root)
            8: 5,   # backend_2 (root)
            9: 9,   # backend_6 (root)
        }

    def analyze_fault(self, fault_id: int, t_start_override=None, t_end_override=None) -> FaultSignal:
        row = self._faults[self._faults["fault_id"] == fault_id].iloc[0]
        t_start = pd.Timestamp(t_start_override) if t_start_override else row["start_time"]
        t_end   = pd.Timestamp(t_end_override)   if t_end_override   else row["end_time"]

        root_id   = int(row["root_cause_service"])
        root_name = _ID_TO_NAME[root_id]
        rep_name  = _ID_TO_NAME[self._reported_ids[fault_id]]
        window_mins = (t_end - t_start).total_seconds() / 60.0

        ranked = _rank_all_services(self._sc, t_start, t_end)

        root_row = ranked[ranked["service"] == root_name]
        rep_row  = ranked[ranked["service"] == rep_name]

        root_rank = int(root_row["rank"].iloc[0]) if len(root_row) > 0 else 999
        root_z    = float(root_row["z_avg"].iloc[0]) if len(root_row) > 0 else 0.0
        rep_rank  = int(rep_row["rank"].iloc[0]) if len(rep_row) > 0 else 999
        rep_z     = float(rep_row["z_avg"].iloc[0]) if len(rep_row) > 0 else 0.0

        top1      = ranked.iloc[0]
        top1_name = str(top1["service"])
        top1_z    = float(top1["z_avg"])

        num_anom  = int((ranked["z_avg"] > ANOMALY_THRESHOLD).sum())
        sig_ratio = root_z / top1_z if top1_z > 0 else 0.0

        # Auto-flag issues
        notes = []
        if root_rank > 1:
            notes.append(f"root not #1 (rank={root_rank}, top1={top1_name} z={top1_z:.1f})")
        if num_anom > 3:
            notes.append(f"HIGH CONTAMINATION: {num_anom} anomalous services")
        if rep_rank > root_rank:
            notes.append(f"reported service ranked LOWER than root ({rep_rank} vs {root_rank})")
        if root_z < ANOMALY_THRESHOLD:
            notes.append(f"ROOT BELOW THRESHOLD (z={root_z:.2f} < {ANOMALY_THRESHOLD})")

        return FaultSignal(
            fault_id=fault_id,
            fault_type=str(row["type"]),
            root_service=root_name,
            reported_service=rep_name,
            root_rank=root_rank,
            root_z=root_z,
            top1_service=top1_name,
            top1_z=top1_z,
            num_anomalous=num_anom,
            signal_ratio=round(sig_ratio, 3),
            reported_rank=rep_rank,
            reported_z=rep_z,
            window_minutes=round(window_mins, 1),
            notes="; ".join(notes) if notes else "OK",
        )

    def analyze_all(self) -> list[FaultSignal]:
        return [self.analyze_fault(i) for i in range(10)]

    def print_report(self, signals: list[FaultSignal]) -> None:
        w = 110
        print(f"\n{'═'*w}")
        print(f"  SIGNAL QUALITY ANALYSIS — all 10 faults (static, no LLM)")
        print(f"{'═'*w}")
        hdr = (f"  {'ID':>2}  {'Type':<26} {'Root':<14} {'RootRk':>6} {'RootZ':>7} "
               f"{'Top1':>14} {'Top1Z':>7} {'#Anom':>6} {'SigRatio':>9}  Notes")
        print(hdr)
        print(f"  {'─'*2}  {'─'*26} {'─'*14} {'─'*6} {'─'*7} {'─'*14} {'─'*7} {'─'*6} {'─'*9}  {'─'*30}")
        for s in signals:
            ok = "✓" if s.root_rank == 1 and s.notes == "OK" else "✗"
            print(
                f"  {ok}{s.fault_id:>2}  {s.fault_type:<26} {s.root_service:<14} "
                f"{s.root_rank:>6} {s.root_z:>7.2f} {s.top1_service:>14} "
                f"{s.top1_z:>7.2f} {s.num_anomalous:>6} {s.signal_ratio:>9.3f}  {s.notes}"
            )
        print(f"{'═'*w}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 2. GraphHopAnalyzer
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HopResult:
    fault_id:       int
    reported:       str
    root:           str
    hop_distance:   int    # shortest path length reported → root
    path:           list[str]
    reachable:      bool   # can we reach root from reported via graph?


class GraphHopAnalyzer:
    """Measure how many graph hops separate the reported service from the root cause."""

    def __init__(self):
        self._faults = pd.read_csv(
            Path(_REPO) / "datasets_complex" / "faults.csv"
        )
        self._reported_ids = {
            0: 18, 1: 4, 2: 12, 3: 18,
            4: 10, 5: 18, 6: 7, 7: 3, 8: 5, 9: 9,
        }

    def _get_graph(self):
        from omnirca.data_layer.graph_loader import get_arch_graph
        return get_arch_graph()

    def _id_to_name_map(self):
        from omnirca.data_layer.loader import get_loader
        ld = get_loader()
        return ld.id_to_name, ld.name_to_id

    def analyze_fault(self, fault_id: int) -> HopResult:
        import networkx as nx
        row       = self._faults[self._faults["fault_id"] == fault_id].iloc[0]
        root_id   = int(row["root_cause_service"])
        rep_id    = self._reported_ids[fault_id]
        root_name = _ID_TO_NAME[root_id]
        rep_name  = _ID_TO_NAME[rep_id]

        if rep_id == root_id:
            return HopResult(fault_id, rep_name, root_name, 0, [rep_name], True)

        G = self._get_graph()

        # Try both directions (call graph is directed A→B = A calls B)
        # From reporter going upstream (following reverse edges = who does reporter call)
        # and downstream (who calls the reporter)
        for source, target in [(rep_id, root_id), (root_id, rep_id)]:
            try:
                path_ids = nx.shortest_path(G, source=source, target=target)
                path_names = [_ID_TO_NAME.get(n, str(n)) for n in path_ids]
                return HopResult(fault_id, rep_name, root_name,
                                 len(path_ids) - 1, path_names, True)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        # Try undirected
        try:
            path_ids = nx.shortest_path(G.to_undirected(), source=rep_id, target=root_id)
            path_names = [_ID_TO_NAME.get(n, str(n)) for n in path_ids]
            return HopResult(fault_id, rep_name, root_name,
                             len(path_ids) - 1, path_names, True)
        except Exception:
            pass

        return HopResult(fault_id, rep_name, root_name, 999, [], False)

    def analyze_all(self) -> list[HopResult]:
        return [self.analyze_fault(i) for i in range(10)]

    def print_report(self, results: list[HopResult]) -> None:
        print(f"\n{'═'*80}")
        print(f"  GRAPH HOP ANALYSIS — reported service → root cause")
        print(f"{'═'*80}")
        print(f"  {'ID':>2}  {'Reported':<16} {'Root':<14} {'Hops':>5}  {'Path'}")
        print(f"  {'─'*2}  {'─'*16} {'─'*14} {'─'*5}  {'─'*40}")
        for r in results:
            reach = "✓" if r.reachable else "✗UNREACHABLE"
            path_str = " → ".join(r.path) if r.path else "(none)"
            print(f"  {r.fault_id:>2}  {r.reported:<16} {r.root:<14} {r.hop_distance:>5}  {path_str}")
        print(f"{'═'*80}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 3. WindowSensitivity
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WindowTest:
    scenario:       str     # label e.g. "exact", "+5min_late", "half_window"
    fault_shift_s:  int     # seconds to shift t_start (+ = late start)
    end_shift_s:    int     # seconds to shift t_end
    description:    str


WINDOW_SCENARIOS: list[WindowTest] = [
    WindowTest("exact",           0,    0,    "Exact fault window (baseline)"),
    WindowTest("+5min_late",      300,  0,    "Start 5 min late (miss early signal)"),
    WindowTest("+10min_late",     600,  0,    "Start 10 min late"),
    WindowTest("-5min_early",    -300,  0,    "Start 5 min early (5 min of noise before fault)"),
    WindowTest("half_window",     0,   -1,    "Cut window in half (end at midpoint)"),
    WindowTest("2x_window",       0,    1,    "Double window (extra 50% padding at end)"),
    WindowTest("pre_fault",      -1,   -1,    "Window BEFORE fault (no signal — null test)"),
    WindowTest("post_fault",      1,    1,    "Window AFTER fault ends (signal gone)"),
]


class WindowSensitivity:
    """Test how z-score signal degrades as the time window deviates from ground truth."""

    def __init__(self):
        self._sc, self._faults = _load_data()

    def test_fault(self, fault_id: int) -> list[dict]:
        row = self._faults[self._faults["fault_id"] == fault_id].iloc[0]
        t_start = row["start_time"]
        t_end   = row["end_time"]
        dur     = (t_end - t_start).total_seconds()
        root_id = int(row["root_cause_service"])
        root_name = _ID_TO_NAME[root_id]

        results = []
        for sc in WINDOW_SCENARIOS:
            # compute shifted/modified window
            if sc.fault_shift_s == -1:  # "pre_fault"
                ws = t_start - pd.Timedelta(seconds=int(dur))
                we = t_start - pd.Timedelta(seconds=60)
            elif sc.fault_shift_s == 1:  # "post_fault"
                ws = t_end + pd.Timedelta(seconds=60)
                we = t_end + pd.Timedelta(seconds=int(dur))
            else:
                ws = t_start + pd.Timedelta(seconds=sc.fault_shift_s)
                if sc.end_shift_s == -1:   # half window
                    we = ws + pd.Timedelta(seconds=dur / 2)
                elif sc.end_shift_s == 1:  # double window
                    we = t_end + pd.Timedelta(seconds=dur / 2)
                else:
                    we = t_end + pd.Timedelta(seconds=sc.end_shift_s)

            ranked = _rank_all_services(self._sc, ws, we)
            root_row  = ranked[ranked["service"] == root_name]
            root_rank = int(root_row["rank"].iloc[0]) if len(root_row) > 0 else 999
            root_z    = float(root_row["z_avg"].iloc[0]) if len(root_row) > 0 else 0.0
            top1_z    = float(ranked.iloc[0]["z_avg"])
            results.append({
                "fault_id":  fault_id,
                "scenario":  sc.scenario,
                "root":      root_name,
                "root_rank": root_rank,
                "root_z":    round(root_z, 2),
                "top1_z":    round(top1_z, 2),
                "detectable": root_rank == 1 and root_z > ANOMALY_THRESHOLD,
            })
        return results

    def test_all(self, fault_ids: Optional[list[int]] = None) -> pd.DataFrame:
        ids = fault_ids or list(range(10))
        all_rows = []
        for fid in ids:
            all_rows.extend(self.test_fault(fid))
        return pd.DataFrame(all_rows)

    def print_window_report(self, df: pd.DataFrame) -> None:
        print(f"\n{'═'*90}")
        print(f"  WINDOW SENSITIVITY ANALYSIS — how z-score changes with window shifts")
        print(f"{'═'*90}")
        for fid in df["fault_id"].unique():
            sub = df[df["fault_id"] == fid]
            row0 = sub[sub["scenario"] == "exact"].iloc[0]
            print(f"\n  Fault {fid} ({row0['root']:>12}) — "
                  f"exact z={row0['root_z']:.2f} rank={row0['root_rank']}")
            print(f"  {'Scenario':<18} {'RootRank':>9} {'RootZ':>7} {'Top1Z':>7}  {'Detectable':>10}")
            print(f"  {'─'*18} {'─'*9} {'─'*7} {'─'*7}  {'─'*10}")
            for _, r in sub.iterrows():
                flag = "✓" if r["detectable"] else "✗"
                print(f"  {r['scenario']:<18} {r['root_rank']:>9} {r['root_z']:>7.2f} "
                      f"{r['top1_z']:>7.2f}  {flag}")
        print(f"\n{'═'*90}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Normal-Period Baseline (adversarial / false-positive rate)
# ═══════════════════════════════════════════════════════════════════════════

class NormalPeriodChecker:
    """
    For each fault test a query with a window that falls ENTIRELY in a normal
    (non-fault) period.  The system should ideally return low z-scores everywhere
    and find no anomaly.  Measures false-positive rate at the data layer.
    """

    def __init__(self):
        self._sc, self._faults = _load_data()

    def _normal_windows(self) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
        """Return a few hand-picked quiet windows (no overlapping fault)."""
        return [
            (pd.Timestamp("2024-01-01 01:00:00"), pd.Timestamp("2024-01-01 01:15:00"), "Jan1 01:00"),
            (pd.Timestamp("2024-01-01 08:00:00"), pd.Timestamp("2024-01-01 08:15:00"), "Jan1 08:00"),
            (pd.Timestamp("2024-01-02 06:00:00"), pd.Timestamp("2024-01-02 06:15:00"), "Jan2 06:00"),
            (pd.Timestamp("2024-01-02 20:00:00"), pd.Timestamp("2024-01-02 20:15:00"), "Jan2 20:00"),
        ]

    def check(self) -> pd.DataFrame:
        rows = []
        for ws, we, label in self._normal_windows():
            ranked = _rank_all_services(self._sc, ws, we)
            num_anom = int((ranked["z_avg"] > ANOMALY_THRESHOLD).sum())
            top_z    = float(ranked.iloc[0]["z_avg"])
            top_svc  = str(ranked.iloc[0]["service"])
            rows.append({
                "window":    label,
                "top_svc":   top_svc,
                "top_z":     round(top_z, 2),
                "num_anom":  num_anom,
                "clean":     num_anom == 0,
            })
        return pd.DataFrame(rows)

    def print_report(self, df: pd.DataFrame) -> None:
        print(f"\n{'═'*70}")
        print(f"  NORMAL PERIOD FALSE-POSITIVE CHECK (no fault in window)")
        print(f"{'═'*70}")
        print(f"  {'Window':<18} {'TopService':<16} {'TopZ':>7} {'#Anom':>6}  {'Clean':>6}")
        print(f"  {'─'*18} {'─'*16} {'─'*7} {'─'*6}  {'─'*6}")
        for _, r in df.iterrows():
            flag = "✓" if r["clean"] else "✗ FALSE POSITIVE"
            print(f"  {r['window']:<18} {r['top_svc']:<16} {r['top_z']:>7.2f} "
                  f"{r['num_anom']:>6}  {flag}")
        print(f"{'═'*70}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 5. StressQuery — tiered live test definitions
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StressQuery:
    test_id:          str
    tier:             int
    tier_label:       str
    fault_id:         Optional[int]
    t_start:          str
    t_end:            str
    reported_service: str
    symptom:          str
    true_root:        Optional[str]      # None for adversarial (no expected root)
    true_category:    Optional[str]
    description:      str


def build_stress_queries() -> list[StressQuery]:
    """
    Returns the complete tiered stress test battery.

    Tier 0 — Control:         3 exact reproductions (should match original results)
    Tier 1 — Easy reprobe:    single-service faults where root == reported
    Tier 2 — Fuzzy windows:   same queries with shifted/truncated time windows
    Tier 3 — Swap service:    report the root directly for victim-reported faults
    Tier 4 — Adversarial:     wrong service + wrong time → system should say "no clear anomaly"
    """
    sc, faults = _load_data()

    def fault_row(fid):
        return faults[faults["fault_id"] == fid].iloc[0]

    def mid(r):
        dur = (r["end_time"] - r["start_time"]).total_seconds()
        return (r["start_time"] + pd.Timedelta(seconds=dur/2)).strftime("%Y-%m-%d %H:%M:%S")

    def shifted(ts, seconds):
        return (pd.Timestamp(ts) + pd.Timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")

    qs: list[StressQuery] = []

    # ────────────────────────────────────────────────────────────────────────
    # TIER 0 — Control: 3 reprobe runs to check reproducibility
    # ────────────────────────────────────────────────────────────────────────
    for fid, root, reported, symptom, cat in [
        (6, "backend_4", "backend_4",
         "backend_4 memory usage has been climbing steadily for 15 minutes with no sign of levelling off. "
         "Response latency is gradually increasing in proportion to the memory growth.",
         "memory_leak"),
        (8, "backend_2", "backend_2",
         "backend_2 is refusing new incoming connections. The service appears to have exhausted its "
         "worker thread capacity — requests are queuing with no response.",
         "thread_pool_exhaustion"),
        (9, "backend_6", "backend_6",
         "backend_6 response times have increased approximately 10x from baseline. All dependent services "
         "are cascading timeouts. The root slowdown appears to originate at this single service.",
         "cascading_timeout"),
    ]:
        r = fault_row(fid)
        qs.append(StressQuery(
            test_id=f"T0_F{fid}_control",
            tier=0, tier_label="Control",
            fault_id=fid,
            t_start=str(r["start_time"]), t_end=str(r["end_time"]),
            reported_service=reported, symptom=symptom,
            true_root=root, true_category=cat,
            description=f"Fault {fid} exact reproduction — consistency check",
        ))

    # ────────────────────────────────────────────────────────────────────────
    # TIER 1 — Easy: single-service faults, only 1 service affected
    # Same as original but emphasize they are definitionally the easiest case
    # We already have 4,6,7,8,9 in original eval; pick 2 for re-examination
    # ────────────────────────────────────────────────────────────────────────
    for fid, reported, symptom, root, cat in [
        (4, "backend_7",
         "backend_7 is producing invalid outputs. Write operations are failing with data integrity "
         "errors. Outputs fail validation downstream.",
         "backend_7", "data_corruption"),
        (7, "backend_0",
         "backend_0 is producing inconsistent results under concurrent load. Simultaneous requests "
         "interfere, causing sporadic non-deterministic failures.",
         "backend_0", "data_race_condition"),
    ]:
        r = fault_row(fid)
        qs.append(StressQuery(
            test_id=f"T1_F{fid}_easy",
            tier=1, tier_label="Easy",
            fault_id=fid,
            t_start=str(r["start_time"]), t_end=str(r["end_time"]),
            reported_service=reported, symptom=symptom,
            true_root=root, true_category=cat,
            description=f"Fault {fid}: single-service, exact window — should be maximum-confidence correct",
        ))

    # ────────────────────────────────────────────────────────────────────────
    # TIER 2 — Fuzzy windows
    # ────────────────────────────────────────────────────────────────────────

    # 2a. Window starts 5 min late (miss first ~5 min of signal)
    for fid, rep, symptom, root, cat in [
        (6, "backend_4",
         "backend_4 memory usage growing steadily. Service performance degrading over time.",
         "backend_4", "memory_leak"),
        (0, "auth_service",
         "auth_service is returning stale or incorrect data. Responses are completing unusually quickly "
         "with outdated values failing downstream validation.",
         "cache_0", "stale_cache"),
    ]:
        r = fault_row(fid)
        qs.append(StressQuery(
            test_id=f"T2a_F{fid}_5min_late",
            tier=2, tier_label="Fuzzy:+5min late",
            fault_id=fid,
            t_start=shifted(r["start_time"], 300),  # 5 min late
            t_end=str(r["end_time"]),
            reported_service=rep, symptom=symptom,
            true_root=root, true_category=cat,
            description=f"Fault {fid}: window starts 5 min late — miss early signal",
        ))

    # 2b. Window cut to first half only
    for fid, rep, symptom, root, cat in [
        (9, "backend_6",
         "backend_6 is significantly slower than baseline. Dependent services are timing out.",
         "backend_6", "cascading_timeout"),
        (5, "auth_service",
         "auth_service latency has increased dramatically. I/O-bound operations system-wide are slower.",
         "database_2", "disk_io_saturation"),
    ]:
        r = fault_row(fid)
        dur = (r["end_time"] - r["start_time"]).total_seconds()
        qs.append(StressQuery(
            test_id=f"T2b_F{fid}_half_window",
            tier=2, tier_label="Fuzzy:half window",
            fault_id=fid,
            t_start=str(r["start_time"]),
            t_end=shifted(r["start_time"], int(dur/2)),
            reported_service=rep, symptom=symptom,
            true_root=root, true_category=cat,
            description=f"Fault {fid}: only first 50% of fault window used",
        ))

    # 2c. Window with 10 min pre-fault noise (starts 10 min before fault)
    r5 = fault_row(5)
    qs.append(StressQuery(
        test_id="T2c_F5_pre_noise",
        tier=2, tier_label="Fuzzy:-10min start",
        fault_id=5,
        t_start=shifted(r5["start_time"], -600),  # 10 min early
        t_end=str(r5["end_time"]),
        reported_service="auth_service",
        symptom="auth_service latency has increased. I/O-bound operations are degrading across the system.",
        true_root="database_2", true_category="disk_io_saturation",
        description="Fault 5: window starts 10 min before fault with normal-period noise prepended",
    ))

    # ────────────────────────────────────────────────────────────────────────
    # TIER 3 — Report root cause directly for victim-reported faults
    # Tests whether agent does BETTER when given the root vs victim
    # ────────────────────────────────────────────────────────────────────────

    # Fault 1: deadlock — original reported backend_1 (victim), now report database_2 (root)
    r1 = fault_row(1)
    qs.append(StressQuery(
        test_id="T3_F1_report_root",
        tier=3, tier_label="Harder:reportRoot",
        fault_id=1,
        t_start=str(r1["start_time"]), t_end=str(r1["end_time"]),
        reported_service="database_2",
        symptom="database_2 is experiencing severe transaction failures. Write operations hang "
                "indefinitely. Multiple backend services depending on database_2 are now blocked.",
        true_root="database_2", true_category="transaction_deadlock",
        description="Fault 1: report the root cause service directly — should be easier than original",
    ))

    # Fault 2: api_version_mismatch — original reported database_1 (victim), now report backend_6 (root)
    r2 = fault_row(2)
    qs.append(StressQuery(
        test_id="T3_F2_report_root",
        tier=3, tier_label="Harder:reportRoot",
        fault_id=2,
        t_start=str(r2["start_time"]), t_end=str(r2["end_time"]),
        reported_service="backend_6",
        symptom="backend_6 is causing connection rejections across downstream data stores. "
                "Services receiving data from backend_6 are failing with protocol-level errors.",
        true_root="backend_6", true_category="api_version_mismatch",
        description="Fault 2: report the root cause service directly — should be easier than original",
    ))

    # ────────────────────────────────────────────────────────────────────────
    # TIER 4 — Adversarial
    # ────────────────────────────────────────────────────────────────────────

    # 4a. Correct service, normal (non-fault) time window — should find no anomaly
    qs.append(StressQuery(
        test_id="T4a_normal_window",
        tier=4, tier_label="Adversarial:normal",
        fault_id=None,
        t_start="2024-01-01 08:00:00", t_end="2024-01-01 08:15:00",
        reported_service="backend_4",
        symptom="backend_4 seems a bit slow today. Some users report intermittent delays. "
                "No clear pattern observed.",
        true_root=None, true_category=None,
        description="Normal period: no fault present — system should report low confidence or no anomaly",
    ))

    # 4b. Completely unrelated service reported during a real fault window
    r6 = fault_row(6)   # fault 6: backend_4 memory leak, window 01:26-01:41
    qs.append(StressQuery(
        test_id="T4b_wrong_service_real_window",
        tier=4, tier_label="Adversarial:wrongSvc",
        fault_id=6,
        t_start=str(r6["start_time"]), t_end=str(r6["end_time"]),
        reported_service="frontend_0",   # unrelated to fault 6
        symptom="frontend_0 users are reporting slow page loads. The frontend appears sluggish. "
                "No obvious backend alerts have been raised.",
        true_root="backend_4",   # actual root during this window
        true_category="memory_leak",
        description="Fault 6 window but report frontend_0 (not in affected list): "
                    "agent should still find backend_4 via data if it searches broadly",
    ))

    # 4c. Multiple unrelated services active at same time — window spans NO fault
    qs.append(StressQuery(
        test_id="T4c_noisy_normal",
        tier=4, tier_label="Adversarial:noise",
        fault_id=None,
        t_start="2024-01-02 06:00:00", t_end="2024-01-02 06:15:00",
        reported_service="api_gateway",
        symptom="api_gateway error rate slightly elevated. Could be normal traffic variation. "
                "No downstream service is clearly failing.",
        true_root=None, true_category=None,
        description="Normal period: gentle noise, should not hallucinate a root cause",
    ))

    return qs
