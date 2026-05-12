"""
omnirca/rag/indexer.py — Build-time script.

Reads FAULT_ENCYCLOPEDIA.md, splits into 10 fault-type chunks, embeds with
all-MiniLM-L6-v2, and saves a FAISS FlatL2 index + chunk metadata JSON.

Usage:
    python -m omnirca.rag.indexer        # from project root
    python omnirca/rag/indexer.py        # direct

Outputs (relative to project root):
    omnirca/rag/index/faiss.index
    omnirca/rag/index/chunks.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ── Sentinel: dataset-specific terms must NEVER appear in the encyclopedia ───
# These are names/IDs that belong to OmniRCA's synthetic dataset and would
# give the RAG layer unfair knowledge of the evaluation data.
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


def _check_forbidden(text: str, chunk_label: str) -> None:
    """Raise ValueError if any forbidden pattern appears in `text`."""
    for pat in _FORBIDDEN_RE:
        m = pat.search(text)
        if m:
            raise ValueError(
                f"FAIRNESS VIOLATION in chunk '{chunk_label}': "
                f"pattern '{pat.pattern}' matched '{m.group()}' — "
                f"dataset-specific term detected in knowledge base."
            )


def _detect_counter_intuitive(text: str) -> bool:
    """Return True if this fault chunk contains a known counter-intuitive signal."""
    markers = [
        r"latency.*decreas",
        r"decreas.*latency",
        r"faster\b",
        r"0\.5[×x]",
        r"cpu.*drop",
        r"drop.*cpu",
        r"looks.*healthy",
        r"healthy.*but",
        r"counter.?intuitive",
        r"mislead",
        r"paradox",
    ]
    low = text.lower()
    return any(re.search(m, low) for m in markers)


def _extract_category(text: str) -> str:
    """Classify chunk as functional vs non-functional from its content."""
    functional_hints = [
        "authentication", "api version", "data race", "data corruption",
        "stale cache", "deadlock", "transaction",
    ]
    low = text.lower()
    return "functional" if any(h in low for h in functional_hints) else "non-functional"


def build_index(encyclopedia_path: Path, output_dir: Path) -> None:
    """
    Main build function:
    1. Read and split FAULT_ENCYCLOPEDIA.md into 10 chunks.
    2. Fairness-check each chunk.
    3. Embed with sentence-transformers.
    4. Build FAISS FlatL2 index.
    5. Save index + metadata.
    """
    try:
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"ERROR: missing dependency — {e}")
        print("Install with: pip install faiss-cpu sentence-transformers")
        sys.exit(1)

    # ── 1. Read source ──────────────────────────────────────────────────────
    if not encyclopedia_path.exists():
        print(f"ERROR: encyclopaedia not found at {encyclopedia_path}")
        sys.exit(1)

    raw = encyclopedia_path.read_text(encoding="utf-8")

    # ── 2. Split on section headers (## 1. … through ## 10. …) ─────────────
    # This regex matches any line starting with "## " followed by a digit.
    split_re = re.compile(r"(?m)^(?=## \d+\.)")
    raw_chunks = [c.strip() for c in split_re.split(raw) if c.strip()]

    # Filter out preamble sections (no ## N. header)
    fault_chunks = [c for c in raw_chunks if re.match(r"^## \d+\.", c)]

    if len(fault_chunks) != 10:
        print(
            f"WARNING: expected 10 fault chunks, got {len(fault_chunks)}. "
            f"Proceeding with what was found."
        )

    # ── 3. Parse metadata + fairness check ─────────────────────────────────
    records: list[dict] = []
    for chunk in fault_chunks:
        # Extract fault name from header: "## N. Fault Name [emoji]"
        header_m = re.match(r"^## \d+\.\s+(.+)", chunk.splitlines()[0])
        raw_name = header_m.group(1).strip() if header_m else "unknown"
        # Strip emoji / non-ASCII characters from the display name
        fault_name = re.sub(r"[^\x00-\x7F]+", "", raw_name).strip()
        # Normalise to snake_case key: lowercase, spaces→_, strip special chars
        key = re.sub(r"[^a-z0-9]+", "_", fault_name.lower()).strip("_")

        _check_forbidden(chunk, fault_name)

        records.append(
            {
                "fault_type": key,
                "fault_name": fault_name,
                "category": _extract_category(chunk),
                "counter_intuitive": _detect_counter_intuitive(chunk),
                "text": chunk,
            }
        )

    print(f"Parsed {len(records)} fault chunks.")
    for r in records:
        ci = " [COUNTER-INTUITIVE]" if r["counter_intuitive"] else ""
        print(f"  {r['fault_name']} ({r['category']}){ci}")

    # ── 4. Embed ────────────────────────────────────────────────────────────
    print("\nLoading sentence-transformer (all-MiniLM-L6-v2)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [r["text"] for r in records]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    print(f"Embedded {embeddings.shape[0]} chunks, dim={dim}.")

    # ── 5. Build FAISS index ────────────────────────────────────────────────
    # Use IndexFlatIP (inner-product) because we normalized → cosine similarity.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors.")

    # ── 6. Save ─────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "faiss.index"
    chunks_path = output_dir / "chunks.json"

    faiss.write_index(index, str(index_path))

    # Don't store the raw text in chunks.json for brevity;
    # We keep a short excerpt + full text for retriever use.
    for r in records:
        r["excerpt"] = r["text"][:300].replace("\n", " ")

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nSaved:\n  {index_path}\n  {chunks_path}")
    print("Index build complete. ✓")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent.parent          # project root
    _enc  = _root / "datasets_complex" / "FAULT_ENCYCLOPEDIA.md"
    _out  = Path(__file__).resolve().parent / "index"
    build_index(_enc, _out)
