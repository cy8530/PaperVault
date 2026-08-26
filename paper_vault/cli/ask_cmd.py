from __future__ import annotations

from ..rag import ask
from ..indexer.store import paper_count, build_where_clause
from ..usage import tracker


def ask_question(question: str, n_papers: int = 5, chunks_per_paper: int | None = None,
                 detail: str | None = None, max_tokens: int | None = None,
                 year_from: int | None = None, year_to: int | None = None, author: str | None = None,
                 session_id: str | None = None, divide_conquer: str | bool = "auto") -> None:
    """RAG-based Q&A across indexed papers. Supports multi-turn via session_id."""
    total = paper_count()
    if total == 0:
        print("No papers indexed. Run 'import' first.")
        return

    where = build_where_clause(year_from=year_from, year_to=year_to, author=author)
    detail_label = {"1": "notes only", "2": "moderate", "3": "extensive", "all": "full text"}.get(detail, "auto")
    if divide_conquer == "1" or divide_conquer is True:
        mode = " [divide & conquer]"
    elif divide_conquer == "auto":
        mode = " [D&C: auto]"
    else:
        mode = ""
    print(f"Searching {total} paper(s) [{detail_label}]{mode}...\n")
    answer = ask(question, n_papers=n_papers, chunks_per_paper=chunks_per_paper,
                 where=where, detail=detail, max_tokens=max_tokens,
                 session_id=session_id, divide_conquer=divide_conquer)
    print(answer)
    print(f"\n{tracker.summary()}")
