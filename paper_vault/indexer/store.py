import lancedb
import pyarrow as pa
from ..config import config
from .embedder import embedding_dim

_CHUNKS_TABLE = "chunks"
_NOTES_TABLE = "notes_index"


def _escape_sql(s: str) -> str:
    """Escape single quotes for safe SQL string interpolation."""
    return s.replace("'", "''")


def _quote_id(paper_id: str) -> str:
    """Return a SQL-safe quoted paper_id literal."""
    return f"'{_escape_sql(paper_id)}'"


def build_where_clause(year_from: int = None, year_to: int = None,
                       author: str = None) -> str | None:
    """Build a SQL WHERE clause from filter parameters. Returns None if no filters."""
    conditions = []
    if year_from is not None:
        conditions.append(f"year >= {year_from}")
    if year_to is not None:
        conditions.append(f"year <= {year_to}")
    if author:
        conditions.append(f"authors LIKE '%{_escape_sql(author)}%'")
    return " AND ".join(conditions) if conditions else None


def _get_db() -> lancedb.DBConnection:
    return lancedb.connect(str(config.VECTORS_DIR))


def _chunks_schema():
    dim = embedding_dim()
    return pa.schema([
        pa.field("paper_id", pa.string()),
        pa.field("chunk_idx", pa.int32()),
        pa.field("text", pa.string()),
        pa.field("section", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
        pa.field("title", pa.string()),
        pa.field("authors", pa.string()),
        pa.field("year", pa.int32()),
        pa.field("keywords", pa.string()),
    ])


def _notes_schema():
    dim = embedding_dim()
    return pa.schema([
        pa.field("paper_id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("authors", pa.string()),
        pa.field("year", pa.int32()),
        pa.field("keywords", pa.string()),
        pa.field("note", pa.string()),
        pa.field("sections", pa.string()),
        pa.field("chunk_count", pa.int32()),
        pa.field("note_file", pa.string()),
        pa.field("content_hash", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
    ])


# ── Chunks index ──────────────────────────────────────

def _ensure_table(db, name: str, schema):
    """Get or create a table. Never overwrites existing data."""
    try:
        return db.open_table(name)
    except Exception:
        return db.create_table(name, schema=schema, mode="create")


def rebuild_chunks_index(paper_id: str, chunks: list[dict], embeddings, meta: dict = None):
    if meta is None:
        meta = {}

    title = meta.get("title", "")
    authors = ", ".join(meta.get("authors", []))
    year = meta.get("year") or 0
    keywords = ", ".join(meta.get("keywords", []))

    db = _get_db()
    rows = [
        {
            "paper_id": paper_id,
            "chunk_idx": i,
            "text": chunk["text"],
            "section": chunk.get("section", ""),
            "vector": embeddings[i].tolist(),
            "title": title,
            "authors": authors,
            "year": year,
            "keywords": keywords,
        }
        for i, chunk in enumerate(chunks)
    ]

    table = _ensure_table(db, _CHUNKS_TABLE, _chunks_schema())
    # Delete existing chunks for this paper (if any), then add new ones
    table.delete(f"paper_id = {_quote_id(paper_id)}")
    table.add(rows)


def search_chunks(query_vector, top_k: int = 5, where: str = None) -> list[dict]:
    db = _get_db()
    try:
        table = db.open_table(_CHUNKS_TABLE)
        q = table.search(query_vector.tolist()).limit(top_k)
        if where:
            q = q.where(where)
        return q.to_list()
    except Exception:
        return []


def search_chunks_for_papers(query_vector, paper_ids: list[str],
                              per_paper: int = 3,
                              sections: list[str] = None,
                              where: str = None) -> list[dict]:
    """Search chunks limited to specific paper_ids. Optionally filter by sections and/or where clause."""
    if not paper_ids:
        return []
    db = _get_db()
    try:
        table = db.open_table(_CHUNKS_TABLE)
        quoted = ", ".join(_quote_id(pid) for pid in paper_ids)
        conditions = [f"paper_id IN ({quoted})"]
        if sections:
            quoted_secs = ", ".join(f"'{_escape_sql(s)}'" for s in sections)
            conditions.append(f"section IN ({quoted_secs})")
        if where:
            conditions.append(where)
        where_clause = " AND ".join(conditions)
        q = table.search(query_vector.tolist()).where(where_clause).limit(len(paper_ids) * per_paper)
        return q.to_list()
    except Exception:
        return []


# ── Notes index (paper-level) ─────────────────────────

def rebuild_notes_index(paper_id: str, note: str, note_embedding, meta: dict,
                        chunk_count: int = 0, sections: str = "",
                        note_file: str = "", content_hash: str = ""):
    db = _get_db()
    row = {
        "paper_id": paper_id,
        "title": meta.get("title", ""),
        "authors": ", ".join(meta.get("authors", [])),
        "year": meta.get("year") or 0,
        "keywords": ", ".join(meta.get("keywords", [])),
        "note": note,
        "sections": sections,
        "chunk_count": chunk_count,
        "note_file": note_file,
        "content_hash": content_hash,
        "vector": note_embedding.tolist(),
    }

    table = _ensure_table(db, _NOTES_TABLE, _notes_schema())
    table.delete(f"paper_id = {_quote_id(paper_id)}")
    table.add([row])


def search_notes(query_vector, top_k: int = 3, where: str = None) -> list[dict]:
    db = _get_db()
    try:
        table = db.open_table(_NOTES_TABLE)
        q = table.search(query_vector.tolist()).limit(top_k)
        if where:
            q = q.where(where)
        return q.to_list()
    except Exception:
        return []


def paper_count() -> int:
    db = _get_db()
    try:
        table = db.open_table(_NOTES_TABLE)
        return table.count_rows()
    except Exception:
        return 0


def get_indexed_paper_ids() -> set[str]:
    """Return the set of paper_ids already in the notes index."""
    db = _get_db()
    try:
        table = db.open_table(_NOTES_TABLE)
        return {row["paper_id"] for row in table.to_arrow().to_pylist()}
    except Exception:
        return set()


def get_indexed_hashes() -> set[str]:
    """Return the set of content_hashes already in the notes index (non-empty only)."""
    db = _get_db()
    try:
        table = db.open_table(_NOTES_TABLE)
        return {row["content_hash"] for row in table.to_arrow().to_pylist()
                if row.get("content_hash")}
    except Exception:
        return set()


def get_paper_meta(paper_id: str) -> dict | None:
    """Return the full row for a paper from notes_index, or None."""
    db = _get_db()
    try:
        table = db.open_table(_NOTES_TABLE)
        rows = table.to_arrow().to_pylist()
        for r in rows:
            if r["paper_id"] == paper_id:
                return r
        return None
    except Exception:
        return None


def update_paper_metadata(paper_id: str, meta: dict, note_file: str = ""):
    """Update metadata (title, authors, year, keywords, note_file) in both tables."""
    title = meta.get("title", "")
    authors = ", ".join(meta.get("authors", []))
    year = meta.get("year") or 0
    keywords = ", ".join(meta.get("keywords", []))
    values = {"title": title, "authors": authors, "year": year, "keywords": keywords}
    if note_file:
        values["note_file"] = note_file
    where = f"paper_id = {_quote_id(paper_id)}"

    db = _get_db()
    for tbl_name in (_CHUNKS_TABLE, _NOTES_TABLE):
        try:
            table = db.open_table(tbl_name)
            table.update(where=where, values=values)
        except Exception:
            pass


def remove_paper(paper_id: str):
    """Remove a paper from both chunks and notes indexes."""
    db = _get_db()
    for tbl_name in (_CHUNKS_TABLE, _NOTES_TABLE):
        try:
            table = db.open_table(tbl_name)
            table.delete(f"paper_id = {_quote_id(paper_id)}")
        except Exception:
            pass


def get_duplicate_paper_ids() -> dict[str, list[int]]:
    """Find paper_ids that appear multiple times in notes_index.

    Returns {paper_id: [row_indices]} for duplicates only.
    """
    db = _get_db()
    try:
        table = db.open_table(_NOTES_TABLE)
        rows = table.to_arrow().to_pylist()
    except Exception:
        return {}

    seen = {}
    for i, r in enumerate(rows):
        pid = r["paper_id"]
        seen.setdefault(pid, []).append(i)
    return {pid: idxs for pid, idxs in seen.items() if len(idxs) > 1}


def remove_duplicate_rows(keep_count: int = 1) -> int:
    """Keep only the first *keep_count* rows per paper_id, drop the rest.

    LanceDB does not support row-level delete by index, so we rebuild the
    table without duplicates.
    """
    db = _get_db()
    try:
        table = db.open_table(_NOTES_TABLE)
        rows = table.to_arrow().to_pylist()
    except Exception:
        return

    # Collect rows to keep
    kept = []
    seen = {}
    for r in rows:
        pid = r["paper_id"]
        count = seen.get(pid, 0)
        if count < keep_count:
            kept.append(r)
        seen[pid] = count + 1

    removed = len(rows) - len(kept)
    if removed > 0:
        db.drop_table(_NOTES_TABLE)
        if kept:
            db.create_table(_NOTES_TABLE, kept, schema=_notes_schema(), mode="create")

    # Also clean chunks table
    try:
        ctable = db.open_table(_CHUNKS_TABLE)
        crows = ctable.to_arrow().to_pylist()
        ckept = []
        cseen = {}
        for r in crows:
            pid = r["paper_id"]
            count = cseen.get(pid, 0)
            if count < keep_count:
                ckept.append(r)
            cseen[pid] = count + 1
        cremoved = len(crows) - len(ckept)
        if cremoved > 0:
            db.drop_table(_CHUNKS_TABLE)
            if ckept:
                db.create_table(_CHUNKS_TABLE, ckept, schema=_chunks_schema(), mode="create")
    except Exception:
        pass

    return removed
