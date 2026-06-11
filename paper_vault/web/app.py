"""Paper Vault Web UI — FastAPI backend."""

import json
import queue
import shutil
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ..config import config, save_settings, _load_settings
from ..indexer.store import get_indexed_paper_ids, get_indexed_hashes, build_where_clause, search_chunks, remove_paper, get_duplicate_paper_ids, remove_duplicate_rows
from ..indexer.embedder import embed_texts
from ..rag.qa import ask_stream
from ..usage import tracker
from ..utils import emit_sse, normalize_id, sanitize_part
from ..importer import import_one, import_pdfs, repair_orphan_notes
from ..cli.fix_cmd import fix_metadata
from .. import prompt_store
from ..notes import prompts as note_prompts
from ..rag import prompts as rag_prompts

app = FastAPI(title="Paper Vault", version="0.1.0")

WEB_DIR = Path(__file__).parent


# ── Title to note file mapping (cached) ──────────────

_note_map_cache = None
_note_stems_cache = None


def _build_note_map():
    global _note_map_cache, _note_stems_cache
    if _note_map_cache is not None:
        return _note_map_cache, _note_stems_cache
    mapping = {}
    stems = []
    for f in config.NOTES_DIR.glob("*.md"):
        mapping[f.stem] = f.name
        stems.append((f.stem, f.name))
    _note_map_cache = mapping
    _note_stems_cache = stems
    return mapping, stems


def _invalidate_note_map():
    global _note_map_cache, _note_stems_cache
    _note_map_cache = None
    _note_stems_cache = None


def _find_note_file(paper_id: str, title: str, note_map: dict, note_stems: list) -> str:
    keys = []
    if title:
        keys.append(sanitize_part(title)[:100])
    keys.append(sanitize_part(paper_id)[:100])
    keys.append(paper_id.strip()[:100])

    for key in keys:
        if not key:
            continue
        for stem, name in note_stems:
            if stem.startswith(key) or key in stem:
                return name

    for key in keys:
        if not key:
            continue
        for stem, name in note_stems:
            if key[:60] in stem or stem[:60] in key:
                return name

    return ""


# ── Settings ────────────────────────────────────────

@app.get("/api/settings")
async def get_settings():
    """Return current config values."""
    return {
        "paths": {
            "import_dirs": [str(p) for p in config.DEFAULT_IMPORT_DIRS],
            "notes_dir": str(config.NOTES_DIR),
            "vectors_dir": str(config.VECTORS_DIR),
            "extracted_dir": str(config.EXTRACTED_DIR),
            "vault_dir": str(config.VAULT_DIR),
        },
        "models": {
            "llm_base_url": config.LLM_BASE_URL,
            "llm_model": config.LLM_MODEL,
            "light_model_id": config.LIGHT_MODEL_ID,
            "embedding_model": config.EMBEDDING_MODEL,
        },
        "import_limits": {
            "max_pdf_mb": config.MAX_PDF_SIZE_MB,
            "max_upload_mb": config.MAX_UPLOAD_MB,
        },
        "indexing": {
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
        },
        "metadata": {
            "metadata_snippet_chars": config.METADATA_SNIPPET_CHARS,
            "meta_extract_max_tokens": config.META_EXTRACT_MAX_TOKENS,
        },
        "note_gen": {
            "note_gen_temperature": config.NOTE_GEN_TEMPERATURE,
            "note_gen_max_tokens": config.NOTE_GEN_MAX_TOKENS,
        },
        "rag": {
            "rag_qa_temperature": config.RAG_QA_TEMPERATURE,
            "rag_detail_moderate_divisor": config.RAG_DETAIL_MODERATE_DIVISOR,
            "rag_detail_extensive_divisor": config.RAG_DETAIL_EXTENSIVE_DIVISOR,
            "rag_detail_moderate_min": config.RAG_DETAIL_MODERATE_MIN,
            "rag_detail_extensive_min": config.RAG_DETAIL_EXTENSIVE_MIN,
        },
        "answer_tokens": {
            "answer_tokens_tier_1": config.ANSWER_TOKENS_TIER_1,
            "answer_tokens_tier_2": config.ANSWER_TOKENS_TIER_2,
            "answer_tokens_tier_3": config.ANSWER_TOKENS_TIER_3,
        },
    }


@app.post("/api/settings")
async def update_settings(data: dict):
    """Save web-configurable settings to vault/settings.json."""
    allowed = {
        "import_dirs", "notes_dir", "vectors_dir", "extracted_dir", "vault_dir",
        "llm_base_url", "llm_model", "light_model_id", "embedding_model",
        "max_pdf_mb", "max_upload_mb",
        "chunk_size", "chunk_overlap",
        "metadata_snippet_chars", "meta_extract_max_tokens",
        "note_gen_temperature", "note_gen_max_tokens",
        "rag_qa_temperature", "rag_detail_moderate_divisor",
        "rag_detail_extensive_divisor",
        "rag_detail_moderate_min", "rag_detail_extensive_min",
        "answer_tokens_tier_1", "answer_tokens_tier_2", "answer_tokens_tier_3",
    }
    sanitized = {k: v for k, v in data.items() if k in allowed}
    save_settings(sanitized)
    config.reload_settings()
    return {"ok": True, "saved": sanitized}


# ── Prompts ──────────────────────────────────────────

@app.get("/api/prompts")
async def get_prompts():
    """Return all prompts with their defaults and current overrides."""
    all_defaults = {**note_prompts.DEFAULTS, **rag_prompts.DEFAULTS}
    overrides = prompt_store._OVERRIDES if hasattr(prompt_store, '_OVERRIDES') else {}
    current = dict(all_defaults)
    current.update(overrides)
    return {
        "prompts": current,
        "defaults": all_defaults,
        "overrides": overrides,
    }


@app.post("/api/prompts")
async def save_prompts(data: dict):
    """Save prompt overrides to vault/prompts.json and reload in-process."""
    allowed_keys = set(note_prompts.DEFAULTS.keys()) | set(rag_prompts.DEFAULTS.keys())
    overrides = {k: v for k, v in data.items() if k in allowed_keys and isinstance(v, str) and v.strip()}
    # Remove keys set to empty/default by comparing against defaults
    all_defaults = {**note_prompts.DEFAULTS, **rag_prompts.DEFAULTS}
    cleaned = {k: v for k, v in overrides.items() if v.strip() != all_defaults.get(k, "").strip()}
    prompt_store.save(cleaned)
    note_prompts.reload()
    rag_prompts.reload()
    return {"ok": True, "saved": len(cleaned), "keys": list(cleaned.keys())}


# ── Directory browser ───────────────────────────────

@app.post("/api/browse-native")
async def browse_native():
    """Open the native OS directory picker and return the chosen path."""
    import platform
    import subprocess

    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["osascript", "-e",
                 'tell app "System Events" to activate',
                 "-e",
                 'POSIX path of (choose folder with prompt "Select a folder")'],
                capture_output=True, text=True, timeout=120,
            )
            path = result.stdout.strip()
            if path and not result.returncode:
                return {"path": path}
        elif system == "Linux":
            result = subprocess.run(
                ["zenity", "--file-selection", "--directory", "--title=Select a folder"],
                capture_output=True, text=True, timeout=120,
            )
            path = result.stdout.strip()
            if path and not result.returncode:
                return {"path": path}
        elif system == "Windows":
            ps_script = '''
Add-Type -AssemblyName System.Windows.Forms
$f = New-Object System.Windows.Forms.FolderBrowserDialog
$f.Description = "Select a folder"
if ($f.ShowDialog() -eq "OK") { $f.SelectedPath }
'''
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True, text=True, timeout=120,
            )
            path = result.stdout.strip()
            if path:
                return {"path": path}
    except Exception:
        pass
    return JSONResponse({"error": "Native picker unavailable — use the Browse modal instead"}, status_code=400)


@app.get("/api/browse")
async def browse_directory(path: str = ""):
    """List subdirectories for the folder picker."""
    if not path:
        path = str(Path.home())
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        return JSONResponse({"error": "Not a directory"}, status_code=400)
    items = []
    try:
        for entry in sorted(p.iterdir()):
            if entry.is_dir() and not entry.name.startswith('.'):
                items.append({"name": entry.name, "path": str(entry), "type": "dir"})
    except PermissionError:
        pass
    parent = str(p.parent) if p.parent != p else None
    return {"path": str(p), "parent": parent, "items": items}


# ── Static pages ────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


# ── Paper library ───────────────────────────────────

@app.get("/api/papers")
async def list_papers():
    """List all indexed papers with metadata (from LanceDB notes_index only)."""
    import lancedb
    db = lancedb.connect(str(config.VECTORS_DIR))
    try:
        table = db.open_table("notes_index")
        rows = table.to_arrow().to_pylist()
    except Exception:
        return []

    note_map, note_stems = _build_note_map()
    seen = set()
    papers = []
    for r in rows:
        pid = r["paper_id"]
        if pid in seen:
            continue
        seen.add(pid)
        title = r.get("title", "") or ""
        note_file = r.get("note_file", "") or _find_note_file(pid, title, note_map, note_stems)
        papers.append({
            "paper_id": pid,
            "title": title,
            "authors": r.get("authors", ""),
            "year": r.get("year", 0),
            "keywords": r.get("keywords", ""),
            "chunk_count": r.get("chunk_count", 0),
            "note_file": note_file,
        })
    return sorted(papers, key=lambda p: (p.get("year") or 0), reverse=True)


@app.get("/api/notes/{paper_id}")
async def get_note(paper_id: str, filename: str = ""):
    """Return a paper's full note content."""
    if filename:
        note_path = config.NOTES_DIR / filename
        if note_path.exists():
            return {"paper_id": paper_id, "filename": filename,
                    "content": note_path.read_text(encoding="utf-8")}

    note_map, note_stems = _build_note_map()
    title = ""
    import lancedb
    db = lancedb.connect(str(config.VECTORS_DIR))
    try:
        table = db.open_table("notes_index")
        rows = table.to_arrow().to_pylist()
        for r in rows:
            if r["paper_id"] == paper_id:
                title = r.get("title", "") or ""
                break
    except Exception:
        pass

    note_file = _find_note_file(paper_id, title, note_map, note_stems)
    if note_file:
        note_path = config.NOTES_DIR / note_file
        if note_path.exists():
            return {"paper_id": paper_id, "filename": note_file,
                    "content": note_path.read_text(encoding="utf-8")}

    return JSONResponse({"error": "Note not found"}, status_code=404)


# ── Import ──────────────────────────────────────────

@app.post("/api/import")
async def import_pdf(file: UploadFile = File(...), request: Request = None):
    """Upload a PDF and import it with streaming progress (SSE)."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "Only PDF files are supported"}, status_code=400)

    # Enforce upload size limit
    if config.MAX_UPLOAD_MB:
        content_length = request.headers.get("content-length") if request else None
        if content_length:
            try:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > config.MAX_UPLOAD_MB:
                    return JSONResponse(
                        {"error": f"File too large ({size_mb:.1f} MB). Maximum: {config.MAX_UPLOAD_MB} MB"},
                        status_code=413,
                    )
            except ValueError:
                pass

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    shutil.copyfileobj(file.file, tmp)
    tmp.close()
    pdf_path = Path(tmp.name)
    display_name = normalize_id(Path(file.filename).stem)

    indexed = get_indexed_paper_ids()
    known_hashes = get_indexed_hashes()
    if display_name in indexed:
        Path(tmp.name).unlink(missing_ok=True)

        async def _already_indexed():
            yield emit_sse("status", message="Already indexed, skipping...")
            yield emit_sse("done", paper_id=display_name, status="skipped")
        return StreamingResponse(_already_indexed(), media_type="text/event-stream")

    progress_queue = queue.Queue()

    def _run_import():
        try:
            import_one(pdf_path, no_llm=False, no_index=False,
                        progress_callback=lambda step, msg: progress_queue.put(
                            {"event": "status", "step": step, "message": msg}),
                        display_name=display_name,
                        known_hashes=known_hashes)
        except Exception as e:
            progress_queue.put({"event": "error", "message": str(e)})
        finally:
            progress_queue.put({"event": "_done"})
            try:
                Path(tmp.name).unlink(missing_ok=True)
            except Exception:
                pass

    thread = threading.Thread(target=_run_import)
    thread.start()

    async def _stream_progress():
        note_file = ""
        while True:
            try:
                item = progress_queue.get(timeout=0.1)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue

            if item["event"] == "_done":
                break
            if item["event"] == "error":
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                break
            if item.get("step") == "done":
                note_file = item.get("message", "")
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

        if not note_file:
            candidates = list(config.NOTES_DIR.glob(f"{display_name}*.md"))
            note_file = candidates[0].name if candidates else ""

        # Invalidate cache so next /api/papers reflects the new note file
        _invalidate_note_map()

        yield emit_sse("done", paper_id=display_name, note_file=note_file,
                       usage=tracker.summary(), status="ok")

    return StreamingResponse(_stream_progress(), media_type="text/event-stream")


# ── Scan & import directories ───────────────────────

@app.post("/api/import-scan")
async def import_scan(request: Request):
    """Scan configured import directories and stream import progress (SSE)."""
    force = request.query_params.get("force") == "1"
    progress_queue = queue.Queue()

    def _run():
        try:
            import_pdfs([], no_llm=False, no_index=False, force=force,
                        progress_callback=lambda step, msg: progress_queue.put(
                            {"event": "status", "step": step, "message": msg}))
        except Exception as e:
            progress_queue.put({"event": "error", "message": str(e)})
        finally:
            progress_queue.put({"event": "_done"})

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    async def _stream():
        while True:
            try:
                item = progress_queue.get(timeout=0.1)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if item["event"] == "_done":
                break
            if item["event"] == "error":
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        _invalidate_note_map()
        yield emit_sse("done", usage=tracker.summary(), status="ok")

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ── Clean duplicates ────────────────────────────────

@app.post("/api/clean-duplicates")
async def clean_duplicates():
    """Remove duplicate paper entries from LanceDB tables."""
    dups = get_duplicate_paper_ids()
    if not dups:
        _invalidate_note_map()
        return {"ok": True, "removed": 0, "duplicates": {}}
    removed = remove_duplicate_rows(keep_count=1)
    _invalidate_note_map()
    return {"ok": True, "removed": removed, "duplicates": {k: len(v) for k, v in dups.items()}}


# ── Reindex orphan notes ─────────────────────────────

@app.post("/api/reindex-orphans")
async def reindex_orphans():
    """Re-index note files that exist on disk but are missing from LanceDB."""
    progress_queue = queue.Queue()

    def _run():
        try:
            repair_orphan_notes(
                progress_callback=lambda step, msg: progress_queue.put(
                    {"event": "status" if step != "done" else "done",
                     "step": step, "message": msg}))
        except Exception as e:
            progress_queue.put({"event": "error", "message": str(e)})
        finally:
            progress_queue.put({"event": "_done"})

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    async def _stream():
        while True:
            try:
                item = progress_queue.get(timeout=0.1)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if item["event"] == "_done":
                break
            if item["event"] == "error":
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        _invalidate_note_map()
        yield emit_sse("done", usage=tracker.summary(), status="ok")

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ── Command executor ────────────────────────────────

@app.post("/api/cmd")
async def run_command(cmd: str = Form(...)):
    """Execute a CLI-like command and stream output via SSE."""
    import shlex

    try:
        parts = shlex.split(cmd)
    except ValueError:
        return JSONResponse({"error": "Invalid command syntax"}, status_code=400)

    if not parts:
        return JSONResponse({"error": "Empty command"}, status_code=400)

    async def _run_cmd():
        command = parts[0]
        kwargs = {}
        positional = []
        i = 1
        while i < len(parts):
            p = parts[i]
            if p == "--force":
                kwargs["force"] = True
            elif p == "--all":
                kwargs["all"] = True
            elif p == "--no-llm":
                kwargs["no_llm"] = True
            elif p == "--no-index":
                kwargs["no_index"] = True
            elif p in ("-k",) and i + 1 < len(parts):
                kwargs["top_k"] = int(parts[i + 1])
                i += 1
            elif p in ("-n", "--notes") and i + 1 < len(parts):
                kwargs["n_papers"] = int(parts[i + 1])
                i += 1
            elif p == "-d" and i + 1 < len(parts):
                kwargs["detail"] = parts[i + 1]
                i += 1
            elif p in ("--year-from",) and i + 1 < len(parts):
                kwargs["year_from"] = int(parts[i + 1])
                i += 1
            elif p in ("--year-to",) and i + 1 < len(parts):
                kwargs["year_to"] = int(parts[i + 1])
                i += 1
            elif p == "--author" and i + 1 < len(parts):
                kwargs["author"] = parts[i + 1]
                i += 1
            elif p == "--max-tokens" and i + 1 < len(parts):
                kwargs["max_tokens"] = int(parts[i + 1])
                i += 1
            elif p == "--chunks" and i + 1 < len(parts):
                kwargs["chunks_per_paper"] = int(parts[i + 1])
                i += 1
            elif not p.startswith("-"):
                positional.append(p)
            i += 1

        try:
            if command == "import":
                paths = positional
                import_pdfs(paths, no_llm=kwargs.get("no_llm", False),
                            no_index=kwargs.get("no_index", False),
                            force=kwargs.get("force", False),
                            progress_callback=lambda step, msg: None)
                _invalidate_note_map()
                yield emit_sse("done", message="Import complete", usage=tracker.summary())

            elif command == "search":
                from ..retriever import search_papers
                query = " ".join(positional) if positional else ""
                if not query:
                    yield emit_sse("error", message="Usage: search <query> [-k N] [--year-from Y] [--year-to Y] [--author NAME]")
                    return
                where = build_where_clause(year_from=kwargs.get("year_from"),
                                           year_to=kwargs.get("year_to"),
                                           author=kwargs.get("author"))
                results = search_papers(query, top_k=kwargs.get("top_k", 5), where=where)
                if not results:
                    yield emit_sse("output", text="No results found.")
                for i, r in enumerate(results, 1):
                    paper = r.get("paper_id", "unknown")
                    title = r.get("title", "")[:120]
                    year = r.get("year", "")
                    text = r.get("text", "")[:250].replace("\n", " ")
                    yield emit_sse("output", text=f"{i}. [{paper}] ({year}) {title}")
                    yield emit_sse("output", text=f"   {text}...")
                yield emit_sse("done", message="Search complete")

            elif command == "list":
                ids = get_indexed_paper_ids()
                yield emit_sse("output", text=f"Indexed papers ({len(ids)}):")
                for pid in sorted(ids):
                    yield emit_sse("output", text=f"  {pid}")
                yield emit_sse("done", message="List complete")

            elif command == "fix-metadata":
                paper_ids = positional
                fix_metadata(paper_ids, all_papers=kwargs.get("all", False))
                _invalidate_note_map()
                yield emit_sse("done", message="Metadata fix complete")

            elif command == "remove":
                for pid in positional:
                    if pid in get_indexed_paper_ids():
                        remove_paper(pid)
                        yield emit_sse("output", text=f"Removed: {pid}")
                    else:
                        yield emit_sse("output", text=f"Not found: {pid}")
                _invalidate_note_map()
                yield emit_sse("done", message="Remove complete")

            elif command == "ask":
                query = " ".join(positional) if positional else ""
                if not query:
                    yield emit_sse("error", message="Usage: ask <question> [-n N] [-d 1|2|3|all|auto] [--max-tokens N]")
                    return
                for event in ask_stream(query, n_papers=kwargs.get("n_papers", 5),
                                        detail=kwargs.get("detail", "auto"),
                                        max_tokens=kwargs.get("max_tokens")):
                    yield event

            else:
                yield emit_sse("error", message=f"Unknown command: {command}. Supported: import, search, ask, list, fix-metadata, remove")
        except Exception as e:
            yield emit_sse("error", message=str(e))

    return StreamingResponse(_run_cmd(), media_type="text/event-stream")


# ── Search ──────────────────────────────────────────

@app.post("/api/search")
async def search(query: str = Form(...), top_k: int = Form(10),
                 year_from: int = Form(None), year_to: int = Form(None),
                 author: str = Form(None)):
    """Semantic search across indexed papers."""
    where = build_where_clause(year_from=year_from, year_to=year_to, author=author)
    q_embedding = embed_texts([query], is_query=True)
    results = search_chunks(q_embedding[0], top_k=top_k, where=where)

    seen = {}
    for r in results:
        pid = r.get("paper_id", "")
        if pid not in seen:
            seen[pid] = {
                "paper_id": pid,
                "title": r.get("title", ""),
                "authors": r.get("authors", ""),
                "year": r.get("year", 0),
                "text": r.get("text", "")[:300],
                "_distance": r.get("_distance", 0),
            }

    return sorted(seen.values(), key=lambda x: x["_distance"])


# ── RAG Q&A ─────────────────────────────────────────

@app.post("/api/ask")
async def ask_question(question: str = Form(...), n_papers: int = Form(5),
                       detail: str = Form("auto"), max_tokens: int = Form(None),
                       year_from: int = Form(None), year_to: int = Form(None),
                       author: str = Form(None)):
    """RAG Q&A with streaming progress (SSE)."""
    where = build_where_clause(year_from=year_from, year_to=year_to, author=author)

    async def _stream():
        for event in ask_stream(question, n_papers=n_papers, chunks_per_paper=None,
                                where=where, detail=detail, max_tokens=max_tokens):
            yield event

    return StreamingResponse(_stream(), media_type="text/event-stream")
