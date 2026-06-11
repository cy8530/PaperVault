"""Adaptive RAG Q&A pipeline with streaming and non-streaming modes."""

import json
import math
import queue
import threading

from ..config import config
from ..indexer.embedder import embed_texts
from ..indexer.store import search_notes, search_chunks_for_papers
from ..usage import tracker
from ..utils import get_llm_client, safe_format, parse_llm_json, emit_sse
from . import prompts


# ── Context builders ──────────────────────────────────

def _format_context(items: list[dict], prefix: str, text_key: str) -> str:
    def _sort_key(item):
        return (item.get("paper_id", ""), item.get("chunk_idx", 0))
    sorted_items = sorted(items, key=_sort_key)

    parts = []
    for item in sorted_items:
        pid = item.get("paper_id", "")
        title = item.get("title") or pid
        label = f"{prefix}/{pid} | Paper: {title}"
        chunk_idx = item.get("chunk_idx")
        if chunk_idx is not None:
            label += f" | chunk_{chunk_idx}"
        section = item.get("section", "")
        if section:
            label += f" | Section: {section}"
        parts.append(f"[{label}]\n{item[text_key]}")
    return "\n\n---\n\n".join(parts)


# ── LLM helpers ───────────────────────────────────────

def _call_light_llm(prompt: str, label: str, max_tokens: int = None) -> str:
    if max_tokens is None:
        max_tokens = config.RAG_FILTER_MAX_TOKENS
    response = get_llm_client().chat.completions.create(
        model=config.LIGHT_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    tracker.add(response, label)
    return response.choices[0].message.content.strip()


# ── Pipeline steps ────────────────────────────────────

def _search_and_filter(question: str, n_papers: int, where: str | None,
                       query_vec=None) -> list[dict]:
    """Stage 1: Broad semantic search + LLM relevance filter."""
    if query_vec is None:
        query_vec = embed_texts([question], is_query=True)[0]

    broad_results = search_notes(query_vec, top_k=max(n_papers * 2, config.RAG_SEARCH_BREADTH_MIN), where=where)
    if not broad_results:
        return []

    if len(broad_results) <= n_papers:
        return broad_results

    titles = [f"{i}. {nr.get('title') or nr['paper_id']}" for i, nr in enumerate(broad_results, 1)]
    raw = _call_light_llm(
        safe_format(prompts.FILTER_PAPERS_PROMPT, question=question, papers="\n".join(titles)),
        label="paper_filter",
    )
    indices = parse_llm_json(raw)
    if not isinstance(indices, list):
        return broad_results[:n_papers]

    index_set = {i for i in indices if 1 <= i <= len(broad_results)}
    filtered = [nr for i, nr in enumerate(broad_results, 1) if i in index_set]
    return filtered[:n_papers] if filtered else broad_results[:n_papers]


def _determine_detail(question: str, notes_results: list[dict],
                      detail: str | None) -> tuple[int, bool]:
    """Stage 2: Returns (detail_level, full_text)."""
    if detail == "all":
        return 3, True
    if detail is not None and detail != "auto":
        return int(detail), False

    notes_context = _format_context(notes_results, "Source: notes", "note")
    raw = _call_light_llm(
        safe_format(prompts.NEED_DETAILS_PROMPT, question=question, context=notes_context),
        label="judge",
        max_tokens=config.RAG_JUDGE_MAX_TOKENS,
    )
    for c in raw:
        if c in "123":
            return int(c), False
    return 2, False


def _match_sections_batch(question: str, notes_results: list[dict]) -> dict[str, list[str]]:
    """Match question to relevant sections for all papers in one LLM call."""
    papers_with_sections = []
    for nr in notes_results:
        sections_json = nr.get("sections", "")
        if not sections_json:
            continue
        sections_list = parse_llm_json(sections_json)
        if not sections_list:
            continue
        title = nr.get("title", nr["paper_id"])
        papers_with_sections.append((nr["paper_id"], title, sections_list))

    if not papers_with_sections:
        return {}

    blocks = []
    for pid, title, sections_list in papers_with_sections:
        lines = [f"Paper: {title}"]
        for s in sections_list:
            heading = s.get("heading", "")
            chunk_start = s.get("chunk_start", 0)
            chunk_end = s.get("chunk_end", 0)
            lines.append(f"  [{chunk_start}-{chunk_end}] {heading}")
        blocks.append("\n".join(lines))

    raw = _call_light_llm(
        safe_format(prompts.BATCH_SECTION_MATCH_PROMPT, question=question, papers="\n\n".join(blocks)),
        label="section_match",
        max_tokens=512,
    )
    all_matches = parse_llm_json(raw)
    if not isinstance(all_matches, dict):
        return {}

    title_to_pid = {title: pid for pid, title, _ in papers_with_sections}
    result = {}
    for key, sections in all_matches.items():
        pid = title_to_pid.get(key, key)
        if isinstance(sections, list):
            result[pid] = sections
    return result


def _answer_tokens(n_papers: int, max_tokens: int | None) -> int:
    if max_tokens is not None:
        return max_tokens
    if n_papers <= 1:
        return config.ANSWER_TOKENS_TIER_1
    if n_papers == 2:
        return config.ANSWER_TOKENS_TIER_2
    return config.ANSWER_TOKENS_TIER_3


# ── Core pipeline (stateless) ─────────────────────────

def _gather_context(question: str, n_papers: int, chunks_per_paper: int | None,
                    where: str | None, detail: str | None,
                    max_tokens: int | None,
                    progress=None) -> tuple[list[str], str | None, int]:
    """Stages 1-3: search, filter, determine detail, retrieve context.

    Embeds the question once and reuses the vector across stages.

    If ``progress`` is callable, it's invoked as ``progress(msg)`` for each
    status update so callers can stream intermediate progress.

    Returns (status_msgs, context, answer_tokens).
    context is None when no papers were found.
    """
    def emit(msg):
        status_msgs.append(msg)
        if progress:
            progress(msg)

    status_msgs = []
    emit("Embedding question...")
    q_embedding = embed_texts([question], is_query=True)
    query_vec = q_embedding[0]

    # Stage 1: Search + filter
    emit("Searching papers...")
    notes_results = _search_and_filter(question, n_papers, where, query_vec=query_vec)
    if not notes_results:
        return status_msgs, None, 0

    selected_count = len(notes_results)
    emit(f"Found {selected_count} relevant papers")

    # Stage 2: Determine detail level
    if detail and detail != "auto" and detail != "all":
        detail_level = int(detail)
        full_text = False
    else:
        emit("Determining detail level...")
        detail_level, full_text = _determine_detail(question, notes_results, detail)

    # Stage 3: Retrieve context
    if detail_level == 1 and not full_text:
        context = _format_context(notes_results, "Source: notes", "note")
        emit("Route: notes only")
    else:
        max_cc = max(nr.get("chunk_count", config.RAG_DEFAULT_CHUNK_COUNT) for nr in notes_results)
        if full_text:
            per_paper = max_cc
        elif chunks_per_paper is not None:
            per_paper = chunks_per_paper
        elif detail_level == 2:
            per_paper = max(config.RAG_DETAIL_MODERATE_MIN, math.ceil(max_cc / config.RAG_DETAIL_MODERATE_DIVISOR))
        else:
            per_paper = max(config.RAG_DETAIL_EXTENSIVE_MIN, math.ceil(max_cc / config.RAG_DETAIL_EXTENSIVE_DIVISOR))

        emit(f"Retrieving chunks ({per_paper}/paper)...")
        section_matches = _match_sections_batch(question, notes_results) if not full_text else {}

        all_chunks = []
        for nr in notes_results:
            pid = nr["paper_id"]
            matched = section_matches.get(pid, [])
            chunks = search_chunks_for_papers(
                query_vec, [pid], per_paper=per_paper,
                sections=matched if matched else None, where=where,
            )
            all_chunks.extend(chunks)

        level_label = {2: "moderate", 3: "extensive"}.get(detail_level, "?")
        if full_text:
            level_label = "full text"
        route = f"Context ready: {len(all_chunks)} chunks [{level_label}]"
        if section_matches:
            route += ", section-targeted"
        emit(route)
        context = (_format_context(notes_results, "Source: notes", "note")
                   + "\n\n---\n\n" + _format_context(all_chunks, "Detail", "text"))

    answer_tokens = _answer_tokens(n_papers, max_tokens)
    return status_msgs, context, answer_tokens


# ── Public API ────────────────────────────────────────

def ask(question: str, n_papers: int = 5, chunks_per_paper: int = None,
        where: str = None, detail: str = None, max_tokens: int = None) -> str:
    """Adaptive RAG Q&A — returns complete answer string (CLI)."""
    status_msgs, context, answer_tokens = _gather_context(
        question, n_papers, chunks_per_paper, where, detail, max_tokens)

    for msg in status_msgs:
        print(f"    [RAG] {msg}")

    if context is None:
        return "No relevant papers found to answer this question."

    print("    [RAG] Generating answer...")
    response = get_llm_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": safe_format(prompts.QA_PROMPT,
            context=context, question=question)}],
        temperature=config.RAG_QA_TEMPERATURE,
        max_tokens=answer_tokens,
        stream=False,
    )
    tracker.add(response, "qa_answer")
    return response.choices[0].message.content


def ask_stream(question: str, n_papers: int = 5, chunks_per_paper: int = None,
               where: str = None, detail: str = None, max_tokens: int = None):
    """Streaming RAG Q&A — yields SSE-formatted progress events (Web).

    Runs _gather_context in a background thread so progress messages are
    yielded in real-time instead of buffered until completion.
    """
    result_queue = queue.Queue()

    def _run_gather():
        try:
            msgs, ctx, tokens = _gather_context(
                question, n_papers, chunks_per_paper, where, detail, max_tokens,
                progress=lambda msg: result_queue.put(("status", msg)),
            )
            result_queue.put(("__result__", (msgs, ctx, tokens)))
        except Exception as e:
            result_queue.put(("__error__", str(e)))

    thread = threading.Thread(target=_run_gather, daemon=True)
    thread.start()

    context = None
    answer_tokens = None

    while True:
        try:
            item = result_queue.get(timeout=0.1)
        except queue.Empty:
            yield ": keepalive\n\n"
            continue

        kind = item[0]
        if kind == "status":
            msg = item[1]
            print(f"    [RAG] {msg}")
            yield emit_sse("status", message=msg)
        elif kind == "__result__":
            _, (status_msgs, context, answer_tokens) = item
            break
        elif kind == "__error__":
            yield emit_sse("error", message=item[1])
            return

    thread.join()

    if context is None:
        yield emit_sse("done", answer="No relevant papers found to answer this question.")
        return

    print("    [RAG] Generating answer...")
    yield emit_sse("status", message="Generating answer...")

    response = get_llm_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": safe_format(prompts.QA_PROMPT,
            context=context, question=question)}],
        temperature=config.RAG_QA_TEMPERATURE,
        max_tokens=answer_tokens,
        stream=True,
    )
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield emit_sse("token", text=chunk.choices[0].delta.content)
    yield emit_sse("done", usage=tracker.summary())
