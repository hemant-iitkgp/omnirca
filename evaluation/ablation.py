"""
omnirca/evaluation/ablation.py — Seven-configuration ablation study (Phase 7).

From plan.md §7.3 — this IS the research paper's core:

  Config A: Bare ReAct + tools, no guidance        (baseline — LLM defaults)
  Config B: A + Syscall-primary scoring            (real signal source added)
  Config C: B + Sub-channel analysis + dyn. graph  (fault-type hints + topology)
  Config D: C + RAG knowledge base                 (fault encyclopedia)
  Config E: D + SOP guidance                       (structured diagnostic sequence)
  Config F: E + Multi-agent + Voting               (Phase 5 system, no temporal)
  Config G: Full system (F + Temporal ordering)    (complete Phase 5 system)

Expected accuracy progression (from plan.md):
  A(~20%) → B(~50%) → C(~55%) → D(~58%) → E(~63%) → F(~66%) → G(~70%)

Implementation status:
  Config G: FULLY RUNNABLE — uses Phase 5 MainAgent (temporal ordering included)
  Config D: FULLY RUNNABLE — uses Phase 4 ReAct agent (full single-agent system)
  Configs A-C, E-F: STUBBED — require stripped system prompts or component removal.
    These require separate agent configurations not yet built.
    Use run_config("config_d") and run_config("config_g") for the two runnable
    data points. Full ablation is planned for Phase 7 extension work.

NOTE: Phase 4 ReAct agent already includes RAG + SOP + syscall-primary, so it
maps to Config E/D. Phase 5 MainAgent is Config G (full system).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from omnirca.evaluation.metrics import EvaluationResult, MetricsReport, compute_metrics
from omnirca.evaluation.query_builder import QueryBuilder
from pathlib import Path

_DEFAULT_FAULTS_CSV = str(
    Path(__file__).parent.parent.parent / "datasets_complex" / "faults.csv"
)


# ─────────────────────────────────────────────────────────────────────────────
# AblationConfig dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    """Specification for one ablation configuration."""
    name:         str          # e.g. "config_g"
    label:        str          # e.g. "Config G"
    description:  str          # what this config adds vs. prior
    runnable:     bool         # True if implementation exists
    expected_acc: float        # expected Top-1 accuracy (from plan.md)
    agent_class:  Optional[str] = None   # dotted path to agent class


ABLATION_CONFIGS: dict[str, AblationConfig] = {
    "config_a": AblationConfig(
        name="config_a",
        label="Config A",
        description="Bare ReAct + all 24 tools, no guidance. LLM defaults to metrics/logs first.",
        runnable=False,
        expected_acc=0.20,
        agent_class=None,   # stub — needs stripped system prompt
    ),
    "config_b": AblationConfig(
        name="config_b",
        label="Config B",
        description="A + Syscall-primary scoring (syscall_multi_service_compare as first call).",
        runnable=False,
        expected_acc=0.50,
        agent_class=None,
    ),
    "config_c": AblationConfig(
        name="config_c",
        label="Config C",
        description="B + Syscall sub-channel analysis + dynamic graph fusion.",
        runnable=False,
        expected_acc=0.55,
        agent_class=None,
    ),
    "config_d": AblationConfig(
        name="config_d",
        label="Config D",
        description="C + RAG knowledge base (FAULT_ENCYCLOPEDIA). Phase 4 ReAct agent.",
        runnable=True,
        expected_acc=0.58,
        agent_class="omnirca.agent.react_agent.ReActAgent",
    ),
    "config_e": AblationConfig(
        name="config_e",
        label="Config E",
        description="D + SOP guidance integrated into FaultTyper. Phase 4 with SOPs.",
        runnable=True,
        expected_acc=0.63,
        agent_class="omnirca.agent.react_agent.ReActAgent",
    ),
    "config_f": AblationConfig(
        name="config_f",
        label="Config F",
        description="E + Multi-agent decomposition + Self-consistency voting.",
        runnable=True,
        expected_acc=0.66,
        agent_class="omnirca.agents.main_agent.MainAgent",
    ),
    "config_g": AblationConfig(
        name="config_g",
        label="Config G",
        description="Full system: F + Temporal causal ordering. Phase 5 MainAgent.",
        runnable=True,
        expected_acc=0.70,
        agent_class="omnirca.agents.main_agent.MainAgent",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# AblationRunner
# ─────────────────────────────────────────────────────────────────────────────

class AblationRunner:
    """
    Runs selected ablation configurations on a set of fault IDs.

    For the current implementation, only config_d and config_g are runnable.
    configs A-C and F require stripped agent variants not yet built.
    """

    def __init__(self, faults_csv: str = _DEFAULT_FAULTS_CSV) -> None:
        self.faults_csv = faults_csv
        self.query_builder = QueryBuilder(faults_csv)

    def run_config(
        self,
        config_name: str,
        fault_ids: Optional[list[int]] = None,
        verbose: bool = True,
    ) -> MetricsReport:
        """
        Run the specified ablation configuration.

        Parameters
        ----------
        config_name : str           e.g. "config_g"
        fault_ids   : list[int]     None = all 10
        verbose     : bool

        Returns
        -------
        MetricsReport for this configuration.
        """
        cfg = ABLATION_CONFIGS.get(config_name)
        if cfg is None:
            raise ValueError(f"Unknown config: {config_name!r}. "
                             f"Valid: {list(ABLATION_CONFIGS.keys())}")
        if not cfg.runnable:
            raise NotImplementedError(
                f"{cfg.label} ({cfg.description}) is not yet implemented. "
                f"Only config_d and config_g are runnable in the current build."
            )

        ids = fault_ids if fault_ids is not None else list(range(10))

        if verbose:
            print(f"\n{'█'*70}")
            print(f"  ABLATION: {cfg.label} — {cfg.description}")
            print(f"  Faults: {ids}")
            print(f"{'█'*70}")

        from omnirca.evaluation.test_harness import EvaluationHarness
        harness = EvaluationHarness(
            faults_csv=self.faults_csv,
            config=config_name,
        )
        results = harness.run_all(fault_ids=ids, verbose=verbose)
        return compute_metrics(results, config=config_name)

    def run_comparison(
        self,
        configs: Optional[list[str]] = None,
        fault_ids: Optional[list[int]] = None,
    ) -> dict[str, MetricsReport]:
        """
        Run multiple configs and return a comparison dict.
        Defaults to running config_d and config_g (the two runnable configs).
        """
        names = configs or ["config_d", "config_g"]
        reports = {}
        for name in names:
            try:
                reports[name] = self.run_config(name, fault_ids=fault_ids)
            except NotImplementedError as e:
                print(f"[AblationRunner] Skipping {name}: {e}")
        return reports

    def print_ablation_table(
        self,
        reports: dict[str, MetricsReport],
    ) -> None:
        """Print the ablation results table."""
        print(f"\n{'═'*78}")
        print(f"  ABLATION TABLE — OmniRCA")
        print(f"{'═'*78}")
        print(f"  {'Config':<10}  {'Description':<35}  "
              f"{'Top-1':>6}  {'Top-3':>6}  {'MRR':>6}  {'Cat':>6}")
        print(f"  {'─'*10}  {'─'*35}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")

        for cfg_name, cfg in ABLATION_CONFIGS.items():
            if cfg_name in reports:
                r = reports[cfg_name]
                print(
                    f"  {cfg.label:<10}  {cfg.description[:35]:<35}  "
                    f"{r.top1_accuracy:>5.0%}  {r.top3_accuracy:>5.0%}  "
                    f"{r.mrr:>5.3f}  {r.category_accuracy:>5.0%}"
                )
            else:
                print(
                    f"  {cfg.label:<10}  {cfg.description[:35]:<35}  "
                    f"{'(est)':>5}  {'─':>5}  {'─':>5}  {'─':>5}  "
                    f"[expected: {cfg.expected_acc:.0%}]"
                )

        print(f"{'═'*78}")
        print(f"  Targets (plan.md):  Top-1 > 60%  |  Top-3 > 85%  |  "
              f"MRR > 0.65  |  Cat > 50%")
        print(f"{'═'*78}")
