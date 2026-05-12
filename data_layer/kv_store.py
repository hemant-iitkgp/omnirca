"""
KV store for tool observations.

Implements the RCAgent pattern: tools store their full result here and return
only a short summary to the LLM context.  The agent can call read_kv(key)
to retrieve full data for specific services when needed.
"""
from __future__ import annotations
from typing import Any


class KVStore:
    """In-memory key-value store for tool observations."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def save(self, key: str, value: Any) -> str:
        """Store value under key. Returns the kv:// URI."""
        self._data[key] = value
        return f"kv://{key}"

    def get(self, key: str) -> Any:
        """Retrieve value by key. Raises KeyError if not found."""
        clean = key.replace("kv://", "")
        if clean not in self._data:
            raise KeyError(f"KV key not found: {key!r}")
        return self._data[clean]

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def clear(self) -> None:
        """Clear all stored observations (call between agent runs)."""
        self._data.clear()

    @staticmethod
    def make_key(tool: str, **kwargs) -> str:
        """Build a deterministic KV key from tool name + arguments."""
        parts = [tool]
        for k, v in sorted(kwargs.items()):
            # Sanitize timestamps: keep only digits and colons → underscores
            safe = str(v).replace(" ", "_").replace(":", "").replace("-", "")
            parts.append(f"{k}{safe}")
        return "_".join(parts)


# ── Module-level singleton ────────────────────────────────────────────────────

_store = KVStore()


def get_store() -> KVStore:
    """Return the shared KV store instance."""
    return _store
