"""Run all static stress analyses — no LLM required."""
import sys
from pathlib import Path

from omnirca.evaluation.stress_analysis import (
    SignalAnalyzer, GraphHopAnalyzer, WindowSensitivity,
    NormalPeriodChecker, build_stress_queries,
)
from collections import Counter

print("==== 1. SIGNAL QUALITY (syscall z-score rank of root cause) ====")
sa = SignalAnalyzer()
signals = sa.analyze_all()
sa.print_report(signals)

print("==== 2. GRAPH HOP DISTANCE ====")
ga = GraphHopAnalyzer()
hops = ga.analyze_all()
ga.print_report(hops)

print("==== 3. WINDOW SENSITIVITY (faults 0,1,5,6,9) ====")
ws = WindowSensitivity()
df = ws.test_all(fault_ids=[0, 1, 5, 6, 9])
ws.print_window_report(df)

print("==== 4. NORMAL PERIOD FALSE-POSITIVE CHECK ====")
npc = NormalPeriodChecker()
np_df = npc.check()
npc.print_report(np_df)

print("==== 5. STRESS BATTERY SUMMARY ====")
queries = build_stress_queries()
tier_counts = Counter(q.tier for q in queries)
print(f"  Total stress queries: {len(queries)}")
for t in sorted(tier_counts):
    label = next(q.tier_label for q in queries if q.tier == t)
    key = label.split(":")[0]
    print(f"  Tier {t}  [{key:<20}]: {tier_counts[t]} queries")
print()
for q in queries:
    print(f"  {q.test_id:<35}  {q.tier_label:<24}  reported={q.reported_service}")
