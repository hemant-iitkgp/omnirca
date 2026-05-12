"""
run.py — Run a single OmniRCA investigation.

Usage:
    python run.py

Edit QUERY, T_START, and T_END below to match your incident before running.
Datasets must be placed at ../datasets_complex/ relative to this repo root.
"""
from __future__ import annotations

import json
from omnirca.agents.main_agent import MainAgent

# ── Configure your incident ───────────────────────────────────────────────────
QUERY = (
    "Multiple services are timing out and requests are failing. "
    "This started around 1pm on Jan 1 and lasted about an hour."
)
T_START = "2024-01-01 13:00:00"
T_END   = "2024-01-01 14:00:00"
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    agent = MainAgent(verbose=True)
    result = agent.run(QUERY, T_START, T_END)

    print("\n" + "═" * 70)
    print("  OmniRCA RESULT")
    print("═" * 70)
    print(json.dumps({
        "root_cause_service": result.root_cause,
        "fault_category":     result.fault_category,
        "confidence":         result.confidence,
        "evidence":           result.evidence_list,
        "propagation_path":   result.propagation_path,
    }, indent=2))


if __name__ == "__main__":
    main()
