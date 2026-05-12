"""
omnirca/agents — Phase 5 multi-agent system.

Exports the MainAgent orchestrator and its result type.
The self-consistency voting wrapper lives in omnirca.voting.
"""
from omnirca.agents.main_agent import MainAgent, MultiAgentResult

__all__ = ["MainAgent", "MultiAgentResult"]
