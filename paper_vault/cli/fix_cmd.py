"""Re-extract metadata for indexed papers without re-importing PDFs."""

from __future__ import annotations

from pathlib import Path

from ..config import config
from ..importer import _make_note_filename
from ..notes.generator import extract_paper_metadata
from ..indexer.store import get_indexed_paper_ids, get_paper_meta, update_paper_metadata
from ..usage import tracker


def _find_extracted(paper_id: str) -> Path | None:
    """Find the extracted text file for a paper_id."""
    from ..utils import normalize_id

    # Exact match
    exact = config.EXTRACTED_DIR / f"{paper_id}.md"
    if exact.exists():
        return exact

    # Normalize-match (paper_id is normalized, extracted files use original PDF stem)
    for f in config.EXTRACTED_DIR.glob("*.md"):
        if normalize_id(f.stem) == paper_id:
            return f

    # Broad substring fallback
    for f in config.EXTRACTED_DIR.glob("*.md"):
        if paper_id[:40] in f.stem or f.stem[:40] in paper_id:
            return f

    return None


def fix_metadata(paper_ids: list[str] | None = None, all_papers: bool = False) -> None:
    """Re-extract metadata for specified papers (or all if --all).

    Updates the LanceDB metadata in place and renames the note file
    when the title changes.
    """
    indexed = get_indexed_paper_ids()

    if all_papers:
        targets = sorted(indexed)
    elif paper_ids:
        targets = [pid for pid in paper_ids if pid in indexed]
        missing = [pid for pid in paper_ids if pid not in indexed]
        if missing:
            print(f"Not indexed: {', '.join(missing)}")
    else:
        print("Specify paper IDs or use --all.")
        return

    if not targets:
        print("No matching papers to fix.")
        return

    print(f"Fixing metadata for {len(targets)} paper(s)...\n")

    fixed = 0
    skipped = 0
    for pid in targets:
        path = _find_extracted(pid)
        if not path:
            print(f"  SKIP  {pid} — no extracted text found")
            skipped += 1
            continue

        text = path.read_text(encoding="utf-8")
        meta, method = extract_paper_metadata(text)

        title = meta.get("title") or ""
        authors = meta.get("authors") or []
        year = meta.get("year") or ""

        # Check if note file should be renamed
        old = get_paper_meta(pid)
        old_note_file = (old.get("note_file") or "") if old else ""
        new_note_file = _make_note_filename(
            meta, path.stem
        ) if title else ""

        if old_note_file and new_note_file and old_note_file != new_note_file:
            old_path = config.NOTES_DIR / old_note_file
            new_path = config.NOTES_DIR / new_note_file
            if old_path.exists() and not new_path.exists():
                old_path.rename(new_path)
                print(f"        renamed: {old_note_file} → {new_note_file}")
            else:
                new_note_file = old_note_file  # Keep old if can't rename

        update_paper_metadata(pid, meta, note_file=new_note_file)

        print(f"  FIX   {pid}")
        print(f"        title:   {title[:80] if title else '(not found)'}")
        print(f"        authors: {len(authors)}")
        print(f"        year:    {year or '(not found)'}")
        print(f"        method:  {method}")
        fixed += 1

    print(f"\nDone. {fixed} fixed, {skipped} skipped. {tracker.summary()}")
