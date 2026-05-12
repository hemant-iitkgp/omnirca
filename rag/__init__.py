# omnirca/rag — RAG Knowledge Base (Phase 3)
# Provides: indexer (build-time), retriever (run-time)
from .retriever import retrieve

__all__ = ["retrieve"]
