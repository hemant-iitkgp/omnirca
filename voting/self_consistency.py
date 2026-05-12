"""
omnirca/voting/self_consistency.py — Self-consistency voting over N agent runs.

Inspired by: RCAgent (p2.txt) — "run N=3 independent trajectories, majority vote"

Why voting matters
──────────────────
A single LLM trajectory can be confidently wrong. Hallucination is stochastic:
the same model at temperature=0.0 might get cache_0 right on runs 1 and 3 but
incorrectly output "auth_service" on run 2. Majority voting across N=3 trajectories
with DIFFERENT temperatures forces the model to arrive at the same answer through
independent reasoning paths. If it can't, confidence is appropriately reduced.

Voting rules (from plan.md §5.3):
  3/3 agree  → final_confidence = mean(scores)          trust = HIGH
  2/3 agree  → final_confidence = mean(scores) × 0.85   trust = MODERATE
  0/3 agree  → final_confidence = max(scores) × 0.60    trust = LOW

Temperatures used: [0.1, 0.4, 0.7]
  Slightly lower than the original [0.3, 0.5, 0.7] plan because Azure GPT-4o
  with temperature=0.0 already barely varies — this spread encourages diversity.

Implementation detail: We run MainAgent 3× with different LLM temperatures.
Since the AzureOpenAI client's temperature is passed per-call, we monkey-patch
the `chat()` function's default temperature rather than threading it down through
all the sub-agents, which would require changing every sub-agent signature.
"""
from __future__ import annotations

import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from omnirca.agents.main_agent import MainAgent, MultiAgentResult
import omnirca.llm_client as _llm_module


# Temperatures for the three independent runs
_VOTE_TEMPERATURES = [0.1, 0.4, 0.7]


@dataclass
class VotingResult:
    """
    Output of run_with_voting — wraps three MultiAgentResult instances and
    adds the vote tally and final aggregated verdict.
    """
    root_cause:       str
    fault_category:   str
    confidence:       str           # HIGH | MEDIUM | LOW (post-vote)
    confidence_score: float         # 0.0–1.0 numeric score
    evidence_list:    list[str]
    propagation_path: list[str]

    # Vote diagnostics
    vote_counts:      Counter               = field(default_factory=Counter)
    vote_agreement:   str = "UNKNOWN"       # "3/3", "2/3", "1/3"
    individual_runs:  list[MultiAgentResult] = field(default_factory=list)
    total_duration_s: float = 0.0

    def summary(self) -> str:
        return (
            f"Root cause: {self.root_cause}  |  "
            f"Category: {self.fault_category}  |  "
            f"Confidence: {self.confidence} ({self.confidence_score:.2f})  |  "
            f"Vote: {self.vote_agreement}  |  "
            f"Duration: {self.total_duration_s:.1f}s"
        )

    def __repr__(self) -> str:
        return (
            f"<VotingResult root_cause={self.root_cause!r} "
            f"agreement={self.vote_agreement!r} "
            f"confidence={self.confidence!r}>"
        )


def run_with_voting(
    incident_query: str,
    t_start:        str,
    t_end:          str,
    n_votes:        int = 3,
    temperatures:   list[float] | None = None,
    verbose:        bool = False,
) -> VotingResult:
    """
    Run MainAgent n_votes times with different LLM temperatures and aggregate.

    Parameters
    ----------
    incident_query : str
        Free-text symptom description.
    t_start, t_end : str
        Fault window boundaries.
    n_votes : int
        Number of independent runs (default 3).
    temperatures : list[float] | None
        Override temperatures list. If None, uses [0.1, 0.4, 0.7].
    verbose : bool
        Print per-run progress.

    Returns
    -------
    VotingResult with aggregated verdict.
    """
    if temperatures is None:
        temperatures = _VOTE_TEMPERATURES[:n_votes]

    # Pad or trim to n_votes
    while len(temperatures) < n_votes:
        temperatures.append(temperatures[-1] + 0.1)
    temperatures = temperatures[:n_votes]

    run_start    = time.monotonic()
    all_results: list[MultiAgentResult] = []

    for i, temp in enumerate(temperatures):
        if verbose:
            print(f"\n{'═' * 70}")
            print(f"  VOTE {i+1}/{n_votes}  |  temperature={temp}")
            print(f"{'═' * 70}")

        # Set temperature for this run by patching the module-level default
        original_chat = _llm_module.chat

        def _patched_chat(messages, tools=None, tool_choice="auto",
                          temperature=temp, max_tokens=2048):
            return original_chat(
                messages, tools=tools, tool_choice=tool_choice,
                temperature=temp, max_tokens=max_tokens,
            )

        _llm_module.chat = _patched_chat  # type: ignore[assignment]
        try:
            result = MainAgent(verbose=verbose).run(incident_query, t_start, t_end)
        finally:
            _llm_module.chat = original_chat  # always restore

        all_results.append(result)

        if verbose:
            print(f"\n  → Vote {i+1}: root_cause={result.root_cause!r}  "
                  f"category={result.fault_category!r}  "
                  f"confidence={result.confidence!r}")

    total_s = time.monotonic() - run_start

    return _aggregate(all_results, total_s)


# ── Aggregation ───────────────────────────────────────────────────────────────

def _aggregate(runs: list[MultiAgentResult], total_s: float) -> VotingResult:
    """Apply majority-vote logic and build a VotingResult."""
    # Count votes on root_cause
    vote_counts: Counter = Counter(r.root_cause for r in runs)
    top_service, top_count = vote_counts.most_common(1)[0]

    n = len(runs)
    # Determine agreement label
    if top_count == n:
        agreement = f"{n}/{n}"
    elif top_count > 1:
        agreement = f"{top_count}/{n}"
    else:
        agreement = f"1/{n}"

    # Individual confidence scores (HIGH=0.9, MEDIUM=0.7, LOW=0.4)
    _CONF_NUM = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.4}
    numeric_scores = [_CONF_NUM.get(r.confidence, 0.5) for r in runs]

    # Weighted confidence based on agreement
    mean_score = sum(numeric_scores) / len(numeric_scores)
    if top_count == n:
        final_score = mean_score
    elif top_count > 1:
        final_score = mean_score * 0.85
    else:
        final_score = max(numeric_scores) * 0.60

    # Map numeric to category
    if final_score >= 0.80:
        final_conf = "HIGH"
    elif final_score >= 0.60:
        final_conf = "MEDIUM"
    else:
        final_conf = "LOW"

    # Pick best run = highest individual confidence for the top_service
    winning_runs = [r for r in runs if r.root_cause == top_service]
    best_run = max(
        winning_runs,
        key=lambda r: _CONF_NUM.get(r.confidence, 0.5),
    )

    # Vote on fault_category among winning runs
    cat_counts = Counter(r.fault_category for r in winning_runs)
    best_category = cat_counts.most_common(1)[0][0]

    return VotingResult(
        root_cause       = top_service,
        fault_category   = best_category,
        confidence       = final_conf,
        confidence_score = round(final_score, 3),
        evidence_list    = best_run.evidence_list,
        propagation_path = best_run.propagation_path,
        vote_counts      = vote_counts,
        vote_agreement   = agreement,
        individual_runs  = runs,
        total_duration_s = round(total_s, 2),
    )
