"""
omnirca/sops/sop_generator.py — Auto-SOP Generator (Phase 6).

When an incident's fault_category does not match any of the 9 known built-in
SOPs, AutoSopGenerator uses the LLM to synthesise a new investigation procedure
from the observed signal patterns.

Design principles (from plan.md §6):
  • Generated SOPs list specific tool calls as steps — no vague "investigate" steps.
  • Zero service names or fault IDs — pure pattern-based diagnostic logic.
  • Results are cached in-memory and registered with SopLibrary so that subsequent
    calls for the same novel category are instant (no repeated LLM calls).
  • If the LLM call fails, the generator falls back to the built-in "unknown" SOP.

Usage:
    from omnirca.sops.sop_generator import AutoSopGenerator

    sop = AutoSopGenerator.get_or_generate(
        fault_category = "network_partition",
        signals        = {"syscall_avg_duration_z": 12.3, ...},
    )
    print(sop.text)
"""
from __future__ import annotations

import re
import time
import textwrap
from typing import Dict, List, Optional, Any

from omnirca.sops.sop_library import SOP, SopLibrary, _normalise


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompt template
# ─────────────────────────────────────────────────────────────────────────────

_GENERATE_PROMPT = textwrap.dedent("""\
You are an expert Site Reliability Engineer writing a Standard Operating Procedure (SOP)
for a novel microservice fault category that your diagnostic system has not seen before.

Fault category: {fault_category}

Observed signal patterns:
{signal_lines}

Write a structured SOP for investigating this fault category.

STRICT RULES:
1. Each diagnostic step MUST reference a specific tool call (e.g., syscall_sub_channel_analysis,
   get_propagation_candidates, detect_causal_order, compare_trace_latency, query_logs,
   cross_correlate_services, detect_memory_slope, detect_error_burst, build_call_path).
2. NO service names with numbers (no "cache_0", "backend_4", "database_1").
3. NO fault IDs or dataset-specific references.
4. Pattern-based only: describe WHAT SIGNALS to look for, not which service will show them.
5. Include 5–8 diagnostic steps. Number them 1. 2. 3. etc.

Output format — MUST include these exact section headers:
DESCRIPTION: <one sentence summary of the fault>
KEY_SIGNALS:
- <signal pattern 1>
- <signal pattern 2>
- <signal pattern 3>
DIAGNOSTIC_STEPS:
1. <specific tool call and what to check>
2. <specific tool call and what to check>
3. ...
RECOMMENDED_TOOLS:
- <tool_name_1>
- <tool_name_2>
- ...
""")


# ─────────────────────────────────────────────────────────────────────────────
# AutoSopGenerator
# ─────────────────────────────────────────────────────────────────────────────

class AutoSopGenerator:
    """
    Generates SOPs for novel fault categories using the LLM.

    Class-level cache ensures each novel category is generated only once per
    process lifetime.  Generated SOPs are also registered with SopLibrary so
    that check_sop() can serve them on subsequent calls.
    """

    _cache: Dict[str, SOP] = {}  # fault_category_key → SOP

    @classmethod
    def get_or_generate(
        cls,
        fault_category: str,
        signals: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> SOP:
        """
        Return cached SOP if available; otherwise generate, cache, and return.

        Parameters
        ----------
        fault_category : str
            The novel fault category name (e.g., "network_partition").
        signals        : dict | None
            Signal patterns observed during the incident.  Used to make the
            generated SOP more specific (omit for a more generic procedure).
        verbose        : bool
            If True, print a message when the LLM is called.

        Returns
        -------
        SOP — always returns a valid SOP; on LLM failure returns the "unknown" fallback.
        """
        # Already in built-in library?
        built_in = SopLibrary.get(fault_category)
        if built_in is not None:
            return built_in

        key = _normalise(fault_category)

        # In local cache?
        if key in cls._cache:
            return cls._cache[key]

        if verbose:
            print(f"[AutoSopGenerator] Generating SOP for novel category: {fault_category!r}")

        sop = cls._generate(fault_category, signals or {})

        cls._cache[key] = sop
        SopLibrary.register(sop)

        return sop

    @classmethod
    def get_cached(cls, fault_category: str) -> Optional[SOP]:
        """Return a previously generated SOP from cache, or None."""
        return cls._cache.get(_normalise(fault_category))

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear the auto-generation cache AND remove auto-generated SOPs from
        SopLibrary.  Useful for test isolation.
        """
        cls._cache.clear()
        SopLibrary.clear_extra()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def _generate(
        cls,
        fault_category: str,
        signals: Dict[str, Any],
    ) -> SOP:
        """Call the LLM to generate a new SOP.  Returns fallback on failure."""
        signal_lines = _format_signals(signals)
        prompt = _GENERATE_PROMPT.format(
            fault_category=fault_category,
            signal_lines=signal_lines,
        )

        try:
            from omnirca.llm_client import simple_chat
            raw = simple_chat(
                user_prompt=prompt,
                system_prompt=(
                    "You are an expert SRE writing pattern-based diagnostic procedures. "
                    "Follow the exact output format requested. "
                    "Never include specific service names like 'cache_0' or 'backend_4'."
                ),
                temperature=0.3,
            )
            return _parse_generated_sop(fault_category, raw)
        except Exception as exc:
            # LLM failure → return built-in "unknown" SOP with a note
            fallback = SopLibrary.get("unknown")
            return SOP(
                fault_category=fault_category,
                category_group="unknown",
                description=f"Auto-generation failed ({type(exc).__name__}) — using generic procedure.",
                key_signals=["(generation failed — signals unknown)"],
                diagnostic_steps=fallback.diagnostic_steps if fallback else [],
                recommended_tools=fallback.recommended_tools if fallback else [],
                text=(
                    f"SOP: {fault_category}  [auto-generated — FAILED]\n"
                    f"Generation error: {exc}\n\n"
                    f"{fallback.text if fallback else 'No fallback available.'}"
                ),
                is_auto_generated=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_signals(signals: Dict[str, Any]) -> str:
    """Format the signals dict for insertion into the LLM prompt."""
    if not signals:
        return "  (no specific signal data provided — generate a generic procedure)"
    lines = []
    for k, v in signals.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _parse_generated_sop(fault_category: str, raw_text: str) -> SOP:
    """
    Parse the LLM's structured output into a SOP dataclass.
    Falls back gracefully if sections are missing.
    """
    description   = _extract_section(raw_text, "DESCRIPTION")
    key_signals   = _extract_bullet_list(raw_text, "KEY_SIGNALS")
    steps_raw     = _extract_numbered_list(raw_text, "DIAGNOSTIC_STEPS")
    tools_raw     = _extract_bullet_list(raw_text, "RECOMMENDED_TOOLS")

    # Final cleanup
    if not description:
        description = f"Auto-generated SOP for {fault_category}."
    if not key_signals:
        key_signals = ["(see diagnostic steps for signal patterns)"]
    if not steps_raw:
        steps_raw = ["1. syscall_multi_service_compare — identify anomalous service.",
                     "2. syscall_sub_channel_analysis — investigate sub-channel patterns.",
                     "3. finalize_rca with available evidence."]
    if not tools_raw:
        # Extract tool names mentioned in steps
        tools_raw = _extract_tools_from_steps(steps_raw)

    # Build full text
    text = (
        f"SOP: {fault_category}  [auto-generated]\n"
        f"{'─' * 50}\n"
        f"Description: {description}\n\n"
        f"Key signals:\n" + "\n".join(f"  • {s}" for s in key_signals) + "\n\n"
        f"Diagnostic steps:\n" + "\n".join(f"  {s}" for s in steps_raw) + "\n\n"
        f"Recommended tools: {', '.join(tools_raw)}\n"
    )

    return SOP(
        fault_category=fault_category,
        category_group="auto_generated",
        description=description,
        key_signals=key_signals,
        diagnostic_steps=steps_raw,
        recommended_tools=tools_raw,
        text=text,
        is_auto_generated=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Text parsing utilities
# ─────────────────────────────────────────────────────────────────────────────

def _extract_section(text: str, header: str) -> str:
    """Extract single-line value after a 'HEADER:' label."""
    m = re.search(rf"^{header}:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_bullet_list(text: str, header: str) -> List[str]:
    """Extract '- item' lines that follow 'HEADER:' up to the next header."""
    m = re.search(
        rf"^{header}:\s*\n((?:[ \t]*[-•]\s*.+\n?)*)",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    if not m:
        return []
    block = m.group(1)
    items = re.findall(r"[-•]\s*(.+)", block)
    return [i.strip() for i in items if i.strip()]


def _extract_numbered_list(text: str, header: str) -> List[str]:
    """Extract '1. item' lines that follow 'HEADER:' up to the next blank header."""
    m = re.search(
        rf"^{header}:\s*\n((?:[ \t]*\d+\..+\n?)*)",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    if not m:
        # fallback: find numbered lines anywhere after the header
        start = text.upper().find(header.upper())
        if start == -1:
            return []
        chunk = text[start:]
        items = re.findall(r"\d+\.\s*(.+)", chunk)
        return [i.strip() for i in items[:10] if i.strip()]
    block = m.group(1)
    items = re.findall(r"\d+\.\s*(.+)", block)
    return [i.strip() for i in items if i.strip()]


# Known tool names for extraction from generated text
_KNOWN_TOOLS = {
    "syscall_multi_service_compare",
    "syscall_sub_channel_analysis",
    "query_syscalls",
    "query_metrics",
    "query_logs",
    "query_traces",
    "compute_anomaly_score",
    "detect_memory_slope",
    "detect_error_burst",
    "compare_trace_latency",
    "get_service_dependencies",
    "build_call_path",
    "get_dynamic_graph",
    "get_propagation_candidates",
    "get_fused_graph_summary",
    "detect_causal_order",
    "cross_correlate_services",
    "search_fault_knowledge",
    "classify_fault_pattern",
    "check_sop",
    "explain_evidence",
    "finalize_rca",
}


def _extract_tools_from_steps(steps: List[str]) -> List[str]:
    """Scan step text for known tool names and return unique list."""
    found: List[str] = []
    seen: set = set()
    combined = " ".join(steps)
    for tool in _KNOWN_TOOLS:
        if tool in combined and tool not in seen:
            found.append(tool)
            seen.add(tool)
    return found or ["syscall_multi_service_compare"]
