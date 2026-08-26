from __future__ import annotations

from typing import Any

from ..indexer.embedder import embed_texts
from ..indexer.store import search_chunks


def search_papers(query: str, top_k: int = 5, where: str | None = None) -> list[dict[str, Any]]:
    """Semantic search across chunk-level index, with optional SQL filter."""
    embeddings = embed_texts([query], is_query=True)
    results = search_chunks(embeddings[0], top_k=top_k, where=where)
    for r in results:
        r.pop("vector", None)
    return results
