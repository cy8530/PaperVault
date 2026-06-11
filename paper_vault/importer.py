"""Core PDF import logic — shared by CLI and Web layers."""

import hashlib
import json
import time
from pathlib import Path

from .config import config
from .parser import extract_text
from .notes.generator import generate_note, extract_paper_metadata
from .indexer import chunk_text, embed_texts, rebuild_chunks_index, rebuild_notes_index
from .indexer.store import get_indexed_paper_ids, get_indexed_hashes
from .usage import tracker
from .utils import normalize_id, sanitize_part


def _is_plausible_venue(s: str) -> bool:
    s = s.strip()
    return 3 <= len(s) <= 60 and sum(1 for c in s if c.isalpha()) >= 2


def _make_note_filename(meta: dict, fallback: str) -> str:
    title = (meta.get("title") or "").strip()

    if not title:
        return f"{sanitize_part(fallback)}.md"

    parts = [sanitize_part(title)]
    venue = (meta.get("venue") or "").strip()
    if _is_plausible_venue(venue):
        parts.append(sanitize_part(venue))
    year = meta.get("year") or ""
    if year:
        parts.append(str(year))

    safe = "_".join(parts)
    if len(safe) > config.NOTE_FILENAME_MAX_LEN:
        safe = safe[:config.NOTE_FILENAME_MAX_LEN].rstrip('_')
    return f"{safe}.md"


def _build_section_map(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    sections = []
    current_section = None
    start_idx = None

    for i, chunk in enumerate(chunks):
        sec = chunk.get("section", "") or ""
        if sec != current_section:
            if current_section is not None:
                sections.append({
                    "heading": current_section,
                    "chunk_start": start_idx,
                    "chunk_end": i - 1,
                })
            current_section = sec
            start_idx = i

    if current_section is not None:
        sections.append({
            "heading": current_section,
            "chunk_start": start_idx,
            "chunk_end": len(chunks) - 1,
        })

    return json.dumps(sections, ensure_ascii=False)


def import_one(pdf_path: Path, no_llm: bool = False, no_index: bool = False,
               progress_callback=None, display_name: str = "",
               known_hashes: set = None):
    """Import a single PDF into the vault.

    If *progress_callback* is provided, it will be called at each major step:
        progress_callback(step_name, message_string)
    where step_name is one of: extracting, metadata, generating_note,
    building_index, done.
    """
    def _progress(step, msg):
        print(msg)
        if progress_callback:
            progress_callback(step, msg)

    file_size = pdf_path.stat().st_size / 1024

    max_mb = config.MAX_PDF_SIZE_MB
    if max_mb and file_size > max_mb * 1024:
        raise ValueError(f"PDF too large: {file_size / 1024:.1f} MB (limit: {max_mb} MB)")

    _progress("extracting", f"Extracting text from {pdf_path.name} ({file_size:.0f} KB)...")

    t1 = time.perf_counter()
    text = extract_text(pdf_path, config.EXTRACTED_DIR)
    t2 = time.perf_counter()
    _progress("extracting", f"Text extracted: {len(text):,} chars in {t2 - t1:.2f}s")

    # Content-based dedup: skip if same text already imported under a different filename
    content_hash = hashlib.md5(text[:config.DEDUP_HASH_CHARS].encode()).hexdigest()
    if known_hashes and content_hash in known_hashes:
        _progress("done", f"SKIP — content already indexed as a different file")
        return

    meta_json = {}
    note_path = config.NOTES_DIR / f"{display_name or pdf_path.stem}.md"
    if not no_llm:
        _progress("metadata", "Extracting metadata (title, authors, year, keywords)...")
        t3 = time.perf_counter()
        meta_json, meta_method = extract_paper_metadata(text)
        t4 = time.perf_counter()
        meta_str = []
        if meta_json.get("title"):
            meta_str.append(f"title: {meta_json['title'][:60]}")
        if meta_json.get("year"):
            meta_str.append(f"year: {meta_json['year']}")
        if meta_json.get("authors"):
            meta_str.append(f"authors: {len(meta_json['authors'])}")
        _progress("metadata", f"Metadata extracted in {t4 - t3:.1f}s [{meta_method}] ({', '.join(meta_str)})")

    note = ""
    if not no_llm:
        est_tokens = len(text) // 3
        _progress("generating_note", f"Generating reading note via {config.LLM_MODEL} (~{est_tokens:,} tokens input)...")
        t5 = time.perf_counter()
        note = generate_note(text)
        t6 = time.perf_counter()
        _progress("generating_note", f"Note generated: {len(note):,} chars in {t6 - t5:.1f}s")

        note_path = config.NOTES_DIR / _make_note_filename(meta_json, display_name or pdf_path.stem)
        note_path.write_text(note, encoding="utf-8")
        _progress("generating_note", f"Saved note: {note_path.name}")

    if not no_index:
        paper_id = normalize_id(display_name or pdf_path.stem)
        _progress("building_index", "Chunking text and building vector index...")

        chunks = chunk_text(text)
        _progress("building_index", f"Created {len(chunks)} chunks, embedding...")
        chunk_embeddings = embed_texts([c["text"] for c in chunks])
        rebuild_chunks_index(paper_id, chunks, chunk_embeddings, meta=meta_json)

        if not no_llm and note and meta_json:
            note_embeddings = embed_texts([note])
            sections_json = _build_section_map(chunks)
            rebuild_notes_index(paper_id, note, note_embeddings[0],
                                meta=meta_json, chunk_count=len(chunks),
                                sections=sections_json, note_file=note_path.name,
                                content_hash=content_hash)
            _progress("building_index", f"Index complete: {len(chunks)} chunks + note")
        else:
            _progress("building_index", f"Index complete: {len(chunks)} chunks")

    _progress("done", note_path.name)


def import_pdfs(paths: list[str], no_llm: bool = False, no_index: bool = False,
                force: bool = False, progress_callback=None):
    """Import PDF files: extract text, cache, optionally generate notes and index.

    If *progress_callback* is provided, called as progress_callback(step, message)
    for each major step (forwarded to import_one).
    """
    if not paths:
        paths = config.DEFAULT_IMPORT_DIRS

    pdf_files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            pdf_files.extend(sorted(path.glob("*.pdf")))
        elif path.suffix.lower() == ".pdf":
            pdf_files.append(path)

    if not pdf_files:
        print("No PDF files found.")
        if progress_callback:
            progress_callback("done", "No PDF files found.")
        return

    indexed_ids = set() if force else get_indexed_paper_ids()
    known_hashes = set() if force else get_indexed_hashes()

    new_files = []
    skipped = 0
    for f in pdf_files:
        if f.stem in indexed_ids:
            skipped += 1
        else:
            new_files.append(f)

    if skipped:
        print(f"Skipping {skipped} already-indexed paper(s) (use --force to re-import).\n")

    if not new_files:
        print("All papers already indexed. Nothing to do.")
        if progress_callback:
            progress_callback("done", "All papers already indexed.")
        return

    print(f"Importing {len(new_files)} paper(s)...\n")

    failed = []
    for pdf_path in new_files:
        try:
            import_one(pdf_path, no_llm, no_index, known_hashes=known_hashes,
                progress_callback=progress_callback)
        except Exception as e:
            failed.append(pdf_path)
            print(f"    [FAILED] {e}\n")

    succeeded = len(new_files) - len(failed)
    print(f"Done. {succeeded} succeeded", end="")
    if failed:
        names = ", ".join(f.name for f in failed)
        print(f", {len(failed)} failed: {names}", end="")
    print(f". {tracker.summary()}")


def _parse_meta_from_note(note_text: str, note_filename: str) -> dict:
    """Extract basic metadata from note content and filename without LLM calls.

    Parses the first H1 heading as title, looks for year patterns.
    """
    import re
    meta = {"title": "", "authors": [], "year": None, "venue": None, "keywords": []}

    # Try to get title from first H1 heading
    for line in note_text.splitlines():
        line = line.strip()
        if line.startswith("# ") and len(line) > 2:
            meta["title"] = line[2:].strip()
            break

    # Try to extract year from filename (e.g. "..._2024.md")
    year_match = re.search(r'_(\d{4})(?:\.md)?$', note_filename)
    if year_match:
        meta["year"] = int(year_match.group(1))

    # Also look for year in the note (e.g. "Published: 2024" or "(2024)")
    if not meta["year"]:
        for pattern in [r'\((\d{4})\)', r'(\d{4})年', r'Published:?\s*(\d{4})',
                        r'[Pp]ublished in (\d{4})']:
            m = re.search(pattern, note_text[:2000])
            if m:
                meta["year"] = int(m.group(1))
                break

    return meta


def repair_orphan_notes(progress_callback=None) -> int:
    """Re-index notes that exist as .md files but are missing from LanceDB.

    Uses existing extracted text and note content — no LLM calls, no API cost.
    Returns the number of papers repaired.
    """
    from .indexer.store import get_indexed_paper_ids, rebuild_notes_index, rebuild_chunks_index
    from .indexer import chunk_text, embed_texts

    # Get both paper_ids and note_file names already in the index
    import lancedb
    db = lancedb.connect(str(config.VECTORS_DIR))
    indexed_ids = set()
    indexed_note_files = set()
    try:
        table = db.open_table("notes_index")
        for r in table.to_arrow().to_pylist():
            indexed_ids.add(r["paper_id"])
            nf = r.get("note_file", "")
            if nf:
                indexed_note_files.add(nf)
    except Exception:
        pass

    orphan_files = []
    for f in sorted(config.NOTES_DIR.glob("*.md")):
        pid = normalize_id(f.stem)
        # Skip if already indexed by paper_id OR note filename
        if pid in indexed_ids or f.name in indexed_note_files:
            continue
        orphan_files.append((pid, f))

    if not orphan_files:
        if progress_callback:
            progress_callback("done", "All notes are already indexed.")
        return 0

    repaired = 0
    for pid, note_path in orphan_files:
        try:
            note_text = note_path.read_text(encoding="utf-8")
        except Exception:
            continue

        # Try to find extracted text for chunking
        extracted_path = config.EXTRACTED_DIR / f"{pid}.md"
        text_for_chunks = ""
        if extracted_path.exists():
            try:
                text_for_chunks = extracted_path.read_text(encoding="utf-8")
            except Exception:
                pass

        if progress_callback:
            progress_callback("status", f"Re-indexing {note_path.name}...")

        # Parse metadata from note content (no LLM cost)
        meta = _parse_meta_from_note(note_text, note_path.name)

        # Build chunks index (from extracted text if available, else from note)
        source_text = text_for_chunks or note_text
        chunks = chunk_text(source_text)
        chunk_embeddings = embed_texts([c["text"] for c in chunks])
        rebuild_chunks_index(pid, chunks, chunk_embeddings, meta=meta)

        # Build notes index
        note_embeddings = embed_texts([note_text])
        sections_json = _build_section_map(chunks)
        rebuild_notes_index(pid, note_text, note_embeddings[0],
                          meta=meta, chunk_count=len(chunks),
                          sections=sections_json, note_file=note_path.name)

        repaired += 1

    if progress_callback:
        progress_callback("done", f"Re-indexed {repaired} orphan note(s).")
    return repaired
