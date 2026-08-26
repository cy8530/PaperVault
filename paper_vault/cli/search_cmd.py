from __future__ import annotations

from ..retriever import search_papers
from ..indexer.store import paper_count, build_where_clause


def search(query: str, top_k: int = 5, year_from: int | None = None,
           year_to: int | None = None, author: str | None = None) -> None:
    """Semantic search across indexed papers, with optional filters."""
    total = paper_count()
    if total == 0:
        print("No papers indexed. Run 'import' first.")
        return

    where = build_where_clause(year_from=year_from, year_to=year_to, author=author)

    # Display
    filter_desc = []
    if year_from or year_to:
        yf = year_from or "*"
        yt = year_to or "*"
        filter_desc.append(f"year {yf}-{yt}")
    if author:
        filter_desc.append(f"author: {author}")
    filter_str = f" [{', '.join(filter_desc)}]" if filter_desc else ""

    print(f"Searching {total} paper(s) for: \"{query}\"{filter_str}\n")
    results = search_papers(query, top_k=top_k, where=where)

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        paper = r.get("paper_id", "unknown")
        title = r.get("title", "")
        year = r.get("year", "")
        text = r.get("text", "")
        preview = text[:250].replace("\n", " ") + ("..." if len(text) > 250 else "")
        meta_info = f"({year}) " if year and year != 0 else ""
        print(f"{i}. {meta_info}[{paper}]")
        if title:
            print(f"   {title[:120]}")
        print(f"   {preview}")
        print()
