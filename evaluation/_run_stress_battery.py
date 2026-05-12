"""
Run the 15 stress-test queries through the full Phase 5 agent system.
Outputs per-query results then a tier summary.
"""
import sys, time, traceback
from pathlib import Path

from omnirca.evaluation.stress_analysis import build_stress_queries
from omnirca.evaluation.metrics import _norm_cat

def run_stress_battery():
    queries = build_stress_queries()
    print(f"\n{'█'*72}")
    print(f"  OmniRCA STRESS BATTERY — {len(queries)} queries across 5 tiers")
    print(f"{'█'*72}\n")

    from omnirca.agents.main_agent import MainAgent
    agent = MainAgent(verbose=False)

    results = []

    for q in queries:
        query_text = (
            f"Incident window: {q.t_start} → {q.t_end}\n"
            f"Reported affected service: {q.reported_service}\n\n"
            f"Symptom description:\n{q.symptom}\n\n"
            f"Investigate and identify the root cause service and fault type."
        )

        print(f"\n{'═'*70}")
        print(f"  [{q.tier}] {q.test_id}")
        print(f"  {q.description}")
        print(f"  Window: {q.t_start} → {q.t_end}")
        print(f"  Reported: {q.reported_service}")
        if q.true_root:
            print(f"  Expected root: {q.true_root}  |  category: {q.true_category}")
        else:
            print(f"  Expected: NO anomaly / uncertain")
        print(f"{'─'*70}")

        t0 = time.monotonic()
        err = None
        try:
            result = agent.run(query_text, q.t_start, q.t_end)
            dur = time.monotonic() - t0
            pred_root = result.root_cause or "unknown"
            pred_cat  = result.fault_category or "unknown"
            pred_conf = result.confidence or "LOW"
        except Exception as exc:
            dur = time.monotonic() - t0
            pred_root = "ERROR"
            pred_cat  = "unknown"
            pred_conf = "LOW"
            err = str(exc)

        # Evaluate
        if q.true_root:
            top1_ok = pred_root.lower().strip() == q.true_root.lower().strip()
            cat_ok  = _norm_cat(pred_cat) == _norm_cat(q.true_category or "")
            mark_t1 = "✓ CORRECT" if top1_ok else "✗ WRONG"
            mark_cat = "✓" if cat_ok else "✗"
        else:
            # Adversarial: no expected root — flag if agent gives HIGH confidence
            top1_ok = False
            cat_ok  = False
            mark_t1 = "?  (adversarial — no expected root)"
            mark_cat = "?"

        print(f"  Predicted: {pred_root:<16}  [{mark_t1}]")
        print(f"  Category:  {pred_cat:<22} [{mark_cat if q.true_root else '? adversarial'}]")
        print(f"  Confidence: {pred_conf}  |  Duration: {dur:.1f}s")
        if err:
            print(f"  ERROR: {err[:150]}")

        results.append({
            "test_id":   q.test_id,
            "tier":      q.tier,
            "tier_label": q.tier_label,
            "true_root": q.true_root,
            "pred_root": pred_root,
            "true_cat":  q.true_category,
            "pred_cat":  pred_cat,
            "confidence": pred_conf,
            "top1_ok":   top1_ok,
            "cat_ok":    cat_ok,
            "dur":       round(dur, 1),
            "error":     err,
            "adversarial": q.true_root is None,
        })

    # ── Summary by tier ──────────────────────────────────────────────────────
    print(f"\n\n{'█'*72}")
    print(f"  STRESS BATTERY RESULTS SUMMARY")
    print(f"{'█'*72}")

    from collections import defaultdict
    tier_data = defaultdict(list)
    for r in results:
        tier_data[r["tier"]].append(r)

    for tier in sorted(tier_data.keys()):
        rows = tier_data[tier]
        label = rows[0]["tier_label"]
        adv_rows = [r for r in rows if r["adversarial"]]
        eval_rows = [r for r in rows if not r["adversarial"]]
        n_eval = len(eval_rows)
        n_top1 = sum(r["top1_ok"] for r in eval_rows)
        n_cat  = sum(r["cat_ok"]  for r in eval_rows)

        print(f"\n  Tier {tier}: {label}")
        for r in rows:
            if r["adversarial"]:
                flag = f"[conf={r['confidence']}]"
                print(f"    {r['test_id']:<35} adversarial  reported={r['pred_root']:<14} {flag}")
            else:
                t1 = "✓" if r["top1_ok"] else "✗"
                ct = "✓" if r["cat_ok"]  else "✗"
                print(f"    {r['test_id']:<35} T1={t1} Cat={ct}  pred={r['pred_root']:<14}  true={r['true_root']}")
        if n_eval > 0:
            print(f"    ── Top-1: {n_top1}/{n_eval} ({n_top1/n_eval:.0%})  "
                  f"Cat: {n_cat}/{n_eval} ({n_cat/n_eval:.0%})")

    # Overall (excluding adversarial)
    all_eval = [r for r in results if not r["adversarial"]]
    n_all = len(all_eval)
    n_t1 = sum(r["top1_ok"] for r in all_eval)
    n_ct = sum(r["cat_ok"]  for r in all_eval)
    adv = [r for r in results if r["adversarial"]]

    print(f"\n{'═'*72}")
    print(f"  OVERALL STRESS RESULTS (excluding {len(adv)} adversarial queries)")
    print(f"  Service (Top-1): {n_t1}/{n_all} = {n_t1/n_all:.0%}")
    print(f"  Category:        {n_ct}/{n_all} = {n_ct/n_all:.0%}")
    print(f"  Adversarial: {len(adv)} queries — confidence levels:")
    for r in adv:
        print(f"    {r['test_id']:<40} conf={r['confidence']}  pred={r['pred_root']}")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    run_stress_battery()
