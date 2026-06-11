from ..rag import ask
from ..indexer.store import paper_count, build_where_clause
from ..usage import tracker


def ask_question(question: str, n_papers: int = 5, chunks_per_paper: int = None,
                 detail: str = None, max_tokens: int = None,
                 year_from: int = None, year_to: int = None, author: str = None):
    """RAG-based Q&A across indexed papers."""
    total = paper_count()
    if total == 0:
        print("No papers indexed. Run 'import' first.")
        return

    where = build_where_clause(year_from=year_from, year_to=year_to, author=author)
    detail_label = {"1": "notes only", "2": "moderate", "3": "extensive", "all": "full text"}.get(detail, "auto")
    print(f"Searching {total} paper(s) [{detail_label}]...\n")
    answer = ask(question, n_papers=n_papers, chunks_per_paper=chunks_per_paper,
                 where=where, detail=detail, max_tokens=max_tokens)
    print(answer)
    print(f"\n{tracker.summary()}")
