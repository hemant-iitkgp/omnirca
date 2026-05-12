"""
omnirca/rag/retriever.py — Run-time semantic search over FAULT_ENCYCLOPEDIA.md.

Public API
----------
retrieve(query: str, top_k: int = 2) -> list[dict]

Each returned dict:
    {
        "fault_type":        str,   # snake_case key, e.g. "stale_cache"
        "fault_name":        str,   # display name, e.g. "Stale Cache"
        "category":          str,   # "functional" | "non-functional"
        "counter_intuitive": bool,  # True if this fault tricks naive analysis
        "similarity":        float, # cosine similarity [0, 1]
        "text":              str,   # full chunk text
        "excerpt":           str,   # first 300 chars
    }
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
_INDEX_DIR   = Path(__file__).resolve().parent / "index"
_INDEX_FILE  = _INDEX_DIR / "faiss.index"
_CHUNKS_FILE = _INDEX_DIR / "chunks.json"

# ── Fairness guard (same patterns as indexer) ─────────────────────────────────
FORBIDDEN_PATTERNS: list[str] = [
    r"\bcache_0\b",
    r"\bauth_service\b",
    r"\bbackend_\d\b",
    r"\bfault_id\b",
    r"\b2024-01-01\b",
    r"\bOmniRCA\b",
    r"\bfaults\.csv\b",
    r"\bservice_\d+\b",
]
_FORBIDDEN_RE = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS]


def _query_is_safe(query: str) -> bool:
    """Return False if the query itself tries to inject dataset-specific knowledge."""
    return not any(pat.search(query) for pat in _FORBIDDEN_RE)


# ── Singleton loader ──────────────────────────────────────────────────────────

class _RAGIndex:
    """Thread-safe lazy singleton that loads the FAISS index + chunk metadata."""

    _instance: Optional["_RAGIndex"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer

        if not _INDEX_FILE.exists() or not _CHUNKS_FILE.exists():
            raise FileNotFoundError(
                f"RAG index not found at {_INDEX_DIR}. "
                f"Run `python -m omnirca.rag.indexer` to build it."
            )

        self._index  = faiss.read_index(str(_INDEX_FILE))
        with open(_CHUNKS_FILE, encoding="utf-8") as f:
            self._chunks: list[dict] = json.load(f)

        self._model  = SentenceTransformer("all-MiniLM-L6-v2")
        self._np     = np

    @classmethod
    def get(cls) -> "_RAGIndex":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def query(self, text: str, top_k: int) -> list[dict]:
        np = self._np
        vec = self._model.encode([text], normalize_embeddings=True)
        vec = np.array(vec, dtype="float32")

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS sentinel
                continue
            chunk = self._chunks[idx].copy()
            chunk["similarity"] = float(score)
            results.append(chunk)

        return results


def retrieve(query: str, top_k: int = 2) -> list[dict]:
    """
    Semantic search over the FAULT_ENCYCLOPEDIA index.

    Parameters
    ----------
    query   : Natural-language description of observed symptoms.
    top_k   : Number of top-matching fault archetypes to return (default 2).

    Returns
    -------
    List of fault-archetype dicts, ranked by cosine similarity (highest first).
    """
    if not query or not query.strip():
        return []

    if not _query_is_safe(query):
        # Strip any injected dataset-specific terms rather than raising, so the
        # agent can still function (just with a sanitized query).
        for pat in _FORBIDDEN_RE:
            query = pat.sub("[REDACTED]", query)

    rag = _RAGIndex.get()
    return rag.query(query.strip(), top_k=top_k)


def is_index_available() -> bool:
    """Return True if the FAISS index has been built and is readable."""
    return _INDEX_FILE.exists() and _CHUNKS_FILE.exists()
