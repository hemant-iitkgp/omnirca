"""
omnirca/sops — SOP Library and Auto-SOP Generator package.

Phase 6 exports:
  SopLibrary        — structured library of all known SOPs
  SOP               — dataclass for a single SOP entry
  AutoSopGenerator  — LLM-backed generator for novel fault categories
"""
from omnirca.sops.sop_library import SopLibrary, SOP
from omnirca.sops.sop_generator import AutoSopGenerator

__all__ = ["SopLibrary", "SOP", "AutoSopGenerator"]
