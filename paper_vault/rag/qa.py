"""Adaptive RAG Q&A pipeline with streaming and non-streaming modes."""

import json
import math
import queue
import re
import threading

from ..config import config
from ..indexer.embedder import embed_texts
from ..indexer.store import search_notes, search_chunks_for_papers
from ..usage import tracker
from ..utils import get_llm_client, safe_format, parse_llm_json, emit_sse
from . import prompts
from . import session as session_mod


# ── Text normalization ─────────────────────────────────

def _normalize_text(text: str) -> str:
    """Collapse whitespace for comparison."""
    return re.sub(r'\s+', ' ', text).strip().lower()


def _word_overlap(a: str, b: str) -> float:
    """Fraction of shorter text's words that appear in longer text."""
    wa = set(a.split())
    wb = set(b.split())
    if not wa or not wb:
        return 0.0
    smaller = wa if len(wa) < len(wb) else wb
    return len(smaller & (wb if smaller is wa else wa)) / len(smaller)


# ── Context deduplication ──────────────────────────────

def _deduplicate_items(items: list[dict], text_key: str) -> list[dict]:
    """Remove items whose text is substantially contained in another item.

    Uses substring containment and word-overlap ratio (>85%) to detect
    near-duplicates.  Keeps the longer / richer item when a duplicate is found.
    """
    if len(items) <= 1:
        return items

    texts = [_normalize_text(item[text_key]) for item in items]
    keep = []
    for i, item in enumerate(items):
        is_dup = False
        for j in range(len(items)):
            if i == j:
                continue
            ti, tj = texts[i], texts[j]
            # Substring containment — shorter inside longer
            if len(ti) < len(tj) and ti in tj:
                is_dup = True
                break
            # High word overlap
            if len(ti) > 40 and len(tj) > 40 and _word_overlap(ti, tj) > 0.85:
                is_dup = True
                break
        if not is_dup:
            keep.append(item)
    return keep


# ── Query expansion ────────────────────────────────────

def _preprocess_question(question: str, session) -> tuple[str, str]:
    """Rewrite + expand question in a single LLM call.

    Merges what were previously two serial calls:
      1. Question rewriting (de-reference pronouns, add context)
      2. Query expansion (extract key terms for better search recall)

    Returns (rewritten_question, search_query).  The search_query is only
    used for embedding / vector search, never shown to the final-answer LLM.
    """
    if not session or not session.turns:
        return question, question

    # Build compact context from last 6 turns
    context = ""
    for t in session.turns[-6:]:
        if t.role == "user":
            context += f"Q: {t.question}\n"
        elif t.role == "assistant" and t.summary:
            context += f"A: {t.summary}\n"

    if not context.strip():
        return question, question

    prompt = safe_format(prompts.QUESTION_PREPROCESS_PROMPT,
                         history=context, question=question)
    try:
        response = get_llm_client().chat.completions.create(
            model=config.LIGHT_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        tracker.add(response, "question_preprocess")
        raw = response.choices[0].message.content.strip()
        result = parse_llm_json(raw)
        if isinstance(result, dict):
            rewritten = result.get("rewritten", question) or question
            terms = result.get("search_terms", "")
            search_query = f"{rewritten}  [context: {terms}]" if terms else rewritten
            return rewritten, search_query
    except Exception:
        pass

    return question, question

def _format_context(items: list[dict], prefix: str, text_key: str) -> str:
    def _sort_key(item):
        return (item.get("paper_id", ""), item.get("chunk_idx", 0))
    sorted_items = sorted(items, key=_sort_key)

    parts = []
    for item in sorted_items:
        pid = item.get("paper_id", "")
        title = item.get("title") or pid
        year = item.get("year", "")
        year_str = f" ({year})" if year else ""
        label = f"{prefix} | {title}{year_str}"
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
                       query_vec=None, progress=None,
                       broad_results: list[dict] = None) -> list[dict]:
    """Stage 1: Broad semantic search + LLM relevance filter.

    If *broad_results* is given (from multi-vector search), skip the search
    step and only run the LLM filter.
    """
    def _emit(msg):
        if progress:
            progress(msg)

    if broad_results is None:
        if query_vec is None:
            query_vec = embed_texts([question], is_query=True)[0]
        broad_results = search_notes(
            query_vec, top_k=max(n_papers, config.RAG_SEARCH_BREADTH_MIN),
            where=where, distance_threshold=config.RAG_SEARCH_DISTANCE_THRESHOLD,
        )

    if not broad_results:
        return []

    if len(broad_results) <= n_papers:
        return broad_results

    _emit(f"Filtering {len(broad_results)} candidates → {n_papers}...")
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


def _multi_vector_search(search_text: str, where: str | None,
                         progress_emit) -> list[dict]:
    """Generate query variants, embed all, search, merge by best _distance."""
    n_variants = config.RAG_QUERY_VARIANTS
    if n_variants <= 0:
        return []

    progress_emit(f"Decomposing query into {n_variants} variants...")
    try:
        raw = _call_light_llm(
            safe_format(prompts.QUERY_DECOMPOSE_PROMPT, n=n_variants, question=search_text),
            label="query_decompose", max_tokens=256,
        )
        result = parse_llm_json(raw)
    except Exception:
        return []

    queries = []
    if isinstance(result, dict) and "queries" in result:
        queries = [q for q in result["queries"] if q and q != search_text][:n_variants]

    if not queries:
        return []

    # Embed original + all variants
    all_queries = [search_text] + queries
    all_embeddings = embed_texts(all_queries, is_query=True)

    # Search with each, merge by paper_id (keep lowest _distance)
    seen = {}
    for i, vec in enumerate(all_embeddings):
        results = search_notes(
            vec, top_k=config.RAG_SEARCH_BREADTH_MIN,
            where=where, distance_threshold=config.RAG_SEARCH_DISTANCE_THRESHOLD,
        )
        for r in results:
            pid = r["paper_id"]
            if pid not in seen or r.get("_distance", 999) < seen[pid].get("_distance", 999):
                seen[pid] = r

    merged = sorted(seen.values(), key=lambda x: x.get("_distance", 999))
    progress_emit(f"Multi-vector: {len(queries)} variants, merged {len(merged)} unique papers")
    return merged


# ── Core pipeline (stateless) ─────────────────────────

def _gather_context(question: str, n_papers: int, chunks_per_paper: int | None,
                    where: str | None, detail: str | None,
                    max_tokens: int | None,
                    progress=None,
                    search_query: str = None) -> tuple[list[str], str | None, int]:
    """Stages 1-3: search, filter, determine detail, retrieve context.

    Embeds the question once and reuses the vector across stages.
    Uses multi-vector search (query decomposition) to improve recall.

    If *search_query* is given, it is used for embedding / vector search
    instead of *question*.  This lets callers inject expanded query terms
    without affecting the answer prompt.

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
    search_text = search_query or question
    emit("Embedding question...")
    q_embedding = embed_texts([search_text], is_query=True)
    query_vec = q_embedding[0]

    # Stage 1a: Multi-vector search (query decomposition)
    multi_results = _multi_vector_search(search_text, where, emit)

    # Stage 1b: Search + filter (merges multi-vector results if available)
    emit("Searching papers...")
    notes_results = _search_and_filter(
        question, n_papers, where, query_vec=query_vec, progress=emit,
        broad_results=multi_results if multi_results else None,
    )
    if not notes_results:
        return status_msgs, None, 0, [], []

    selected_count = len(notes_results)
    emit(f"Found {selected_count} relevant papers")

    # Kick off section matching in background while detail judge runs (parallel)
    section_match_result = {}
    section_match_done = threading.Event()
    section_match_error = None

    def _run_section_match():
        nonlocal section_match_result, section_match_error
        try:
            section_match_result = _match_sections_batch(question, notes_results)
        except Exception as e:
            section_match_error = e
        finally:
            section_match_done.set()

    section_thread = threading.Thread(target=_run_section_match, daemon=True)
    section_thread.start()

    # Stage 2: Determine detail level (runs in parallel with section matching)
    if detail and detail != "auto" and detail != "all":
        detail_level = int(detail)
        full_text = False
    else:
        emit("Determining detail level...")
        detail_level, full_text = _determine_detail(question, notes_results, detail)

    # Stage 3: Retrieve context
    all_chunks = []
    if detail_level == 1 and not full_text:
        # Level 1: notes + minimal chunks for factual grounding
        notes_dedup = _deduplicate_items(notes_results, "note")
        context = _format_context(notes_dedup, "Paper", "note")

        min_chunks = config.RAG_LEVEL1_MIN_CHUNKS
        if min_chunks > 0:
            emit(f"Retrieving minimal chunks ({min_chunks}/paper)...")
            all_chunks = []
            for nr in notes_results:
                pid = nr["paper_id"]
                cc = nr.get("chunk_count", 0)
                per = min(min_chunks, max(1, cc))
                chunks = search_chunks_for_papers(
                    query_vec, [pid], per_paper=per,
                    where=where,
                    distance_threshold=config.RAG_SEARCH_DISTANCE_THRESHOLD,
                )
                all_chunks.extend(chunks)
            if all_chunks:
                all_chunks = _deduplicate_items(all_chunks, "text")
                context += "\n\n---\n\n" + _format_context(all_chunks, "Detail", "text")
                emit(f"Route: notes + {len(all_chunks)} minimal chunks")
            else:
                emit("Route: notes only")
        else:
            emit("Route: notes only")

        if len(notes_dedup) < len(notes_results):
            emit(f"Dedup: {len(notes_results) - len(notes_dedup)} note items merged")
    else:
        # Need sections — wait for background thread (already running or done)
        if not full_text:
            section_match_done.wait()
            if section_match_error:
                emit(f"Section match failed: {section_match_error}")

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
        section_matches = section_match_result if not full_text else {}

        all_chunks = []
        for nr in notes_results:
            pid = nr["paper_id"]
            matched = section_matches.get(pid, [])
            chunks = search_chunks_for_papers(
                query_vec, [pid], per_paper=per_paper,
                sections=matched if matched else None, where=where,
                distance_threshold=config.RAG_SEARCH_DISTANCE_THRESHOLD,
            )
            all_chunks.extend(chunks)

        level_label = {2: "moderate", 3: "extensive"}.get(detail_level, "?")
        if full_text:
            level_label = "full text"
        route = f"Context ready: {len(all_chunks)} chunks [{level_label}]"
        if section_matches:
            route += ", section-targeted"

        # Deduplicate chunks (adjacent chunks may overlap due to chunk_overlap)
        chunks_before = len(all_chunks)
        all_chunks = _deduplicate_items(all_chunks, "text")
        if len(all_chunks) < chunks_before:
            route += f", dedup: {chunks_before} → {len(all_chunks)}"

        emit(route)
        context = (_format_context(notes_results, "Paper", "note")
                   + "\n\n---\n\n" + _format_context(all_chunks, "Detail", "text"))

    answer_tokens = _answer_tokens(n_papers, max_tokens)
    return status_msgs, context, answer_tokens, notes_results, all_chunks


# ── Divide and conquer ──────────────────────────────

def _divide_conquer_answer(question: str, notes_results: list[dict],
                           all_chunks: list[dict], history_text: str,
                           answer_tokens: int) -> str:
    """Answer each paper's context separately, then synthesize."""
    # Group chunks by paper_id
    chunks_by_paper = {}
    for c in all_chunks:
        pid = c.get("paper_id", "")
        chunks_by_paper.setdefault(pid, []).append(c)

    # Generate per-paper sub-answers
    per_paper_answers = []
    for nr in notes_results:
        pid = nr["paper_id"]
        title = nr.get("title", pid)
        paper_chunks = chunks_by_paper.get(pid, [])
        paper_context = _format_context([nr], "Paper", "note")
        if paper_chunks:
            paper_context += "\n\n---\n\n" + _format_context(paper_chunks, "Detail", "text")

        sub_max_tokens = min(answer_tokens // max(len(notes_results), 1), 1024)
        print(f"    [RAG] Divide: analyzing {title[:60]}...")
        response = get_llm_client().chat.completions.create(
            model=config.LIGHT_MODEL_ID,
            messages=[{"role": "user", "content": safe_format(
                prompts.DIVIDE_SUB_PROMPT, context=paper_context, question=question)}],
            temperature=0.0,
            max_tokens=sub_max_tokens,
        )
        tracker.add(response, "divide_sub_answer")
        sub_answer = response.choices[0].message.content
        per_paper_answers.append(f"## {title}\n{sub_answer}")

    # Synthesize final answer
    print(f"    [RAG] Synthesizing {len(per_paper_answers)} sub-answers...")
    synthesis = "\n\n---\n\n".join(per_paper_answers)
    prompt = safe_format(prompts.DIVIDE_SYNTHESIS_PROMPT,
                         per_paper_answers=synthesis, question=question)
    if history_text:
        prompt = history_text + "\n" + prompt

    response = get_llm_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.RAG_QA_TEMPERATURE,
        max_tokens=answer_tokens,
        stream=False,
    )
    tracker.add(response, "divide_synthesis")
    return response.choices[0].message.content


# ── Public API ────────────────────────────────────────

def ask(question: str, n_papers: int = 5, chunks_per_paper: int = None,
        where: str = None, detail: str = None, max_tokens: int = None,
        session_id: str = None, divide_conquer: bool = False) -> str:
    """Adaptive RAG Q&A — returns complete answer string (CLI).

    If *session_id* is provided, loads the session for multi-turn context:
    rewrites the question to be self-contained, injects history into the
    answer prompt, and records the exchange after completion.

    If *divide_conquer* is True and multiple papers are found, answers each
    paper separately then synthesizes a final answer.
    """
    session = None
    rewritten_question = question
    search_query = question

    if session_id:
        session = session_mod.get_session(session_id)
        if session:
            rewritten_question, search_query = _preprocess_question(question, session)
            if rewritten_question != question:
                print(f"    [RAG] Rewritten: {rewritten_question}")
            if search_query != rewritten_question:
                print(f"    [RAG] Expanded: {search_query}")

    status_msgs, context, answer_tokens, notes_results, all_chunks = _gather_context(
        rewritten_question, n_papers, chunks_per_paper, where, detail, max_tokens,
        search_query=search_query)

    for msg in status_msgs:
        print(f"    [RAG] {msg}")

    if context is None:
        return ("No relevant papers found in your library for this question. "
                "Suggestions: try broader search terms, reduce year/author filters, "
                "or import more papers on this topic.")

    # Build history section
    history_text = ""
    if session:
        history_text = session_mod.build_history_for_prompt(session)
        if history_text:
            history_text = f"## Conversation history (for context)\n{history_text}\n"
            history_tokens = session_mod.estimate_tokens(history_text)
            if history_tokens > config.CONTEXT_HISTORY_MAX_TOKENS:
                lines = history_text.split("\n")
                history_text = "\n".join(lines[-30:])

    if divide_conquer and len(notes_results) > 1:
        answer = _divide_conquer_answer(
            question, notes_results, all_chunks, history_text, answer_tokens)
    else:
        print("    [RAG] Generating answer...")
        response = get_llm_client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": safe_format(prompts.QA_PROMPT,
                history=history_text, context=context, question=question)}],
            temperature=config.RAG_QA_TEMPERATURE,
            max_tokens=answer_tokens,
            stream=False,
        )
        tracker.add(response, "qa_answer")
        answer = response.choices[0].message.content

    if session:
        session_mod.after_answer(question, rewritten_question, answer, session)

    return answer


def ask_stream(question: str, n_papers: int = 5, chunks_per_paper: int = None,
               where: str = None, detail: str = None, max_tokens: int = None,
               session_id: str = None, divide_conquer: bool = False):
    """Streaming RAG Q&A — yields SSE-formatted progress events (Web).

    Runs _gather_context in a background thread so progress messages are
    yielded in real-time instead of buffered until completion.

    If *session_id* is provided, loads session for multi-turn context.

    If *divide_conquer* is True and multiple papers are found, answers each
    paper separately then streams the final synthesis.
    """
    session = None
    rewritten_question = question
    search_query = question

    if session_id:
        session = session_mod.get_session(session_id)
        if session:
            rewritten_question, search_query = _preprocess_question(question, session)
            if rewritten_question != question:
                yield emit_sse("status", message=f"Rewritten: {rewritten_question}")
            if search_query != rewritten_question:
                yield emit_sse("status", message=f"Expanded: {search_query}")

    result_queue = queue.Queue()

    def _run_gather():
        try:
            msgs, ctx, tokens, notes, chunks = _gather_context(
                rewritten_question, n_papers, chunks_per_paper, where, detail, max_tokens,
                progress=lambda msg: result_queue.put(("status", msg)),
                search_query=search_query,
            )
            result_queue.put(("__result__", (msgs, ctx, tokens, notes, chunks)))
        except Exception as e:
            result_queue.put(("__error__", str(e)))

    thread = threading.Thread(target=_run_gather, daemon=True)
    thread.start()

    context = None
    answer_tokens = None
    notes_results = []
    all_chunks = []

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
            _, (status_msgs, context, answer_tokens, notes_results, all_chunks) = item
            break
        elif kind == "__error__":
            yield emit_sse("error", message=item[1])
            return

    thread.join()

    if context is None:
        yield emit_sse("done", answer=(
            "No relevant papers found in your library for this question. "
            "Suggestions: try broader search terms, reduce year/author filters, "
            "or import more papers on this topic."))
        return

    # Build history section
    history_text = ""
    if session:
        history_text = session_mod.build_history_for_prompt(session)
        if history_text:
            history_text = f"## Conversation history (for context)\n{history_text}\n"
            history_tokens = session_mod.estimate_tokens(history_text)
            if history_tokens > config.CONTEXT_HISTORY_MAX_TOKENS:
                lines = history_text.split("\n")
                history_text = "\n".join(lines[-30:])

    if divide_conquer and len(notes_results) > 1:
        # Divide-and-conquer: per-paper sub-answers, then stream synthesis
        chunks_by_paper = {}
        for c in all_chunks:
            pid = c.get("paper_id", "")
            chunks_by_paper.setdefault(pid, []).append(c)

        per_paper_answers = []
        for idx, nr in enumerate(notes_results):
            pid = nr["paper_id"]
            title = nr.get("title", pid)
            paper_chunks = chunks_by_paper.get(pid, [])
            paper_context = _format_context([nr], "Paper", "note")
            if paper_chunks:
                paper_context += "\n\n---\n\n" + _format_context(paper_chunks, "Detail", "text")

            sub_max = min(answer_tokens // max(len(notes_results), 1), 1024)
            msg = f"Divide: analyzing paper {idx + 1}/{len(notes_results)}: {title[:60]}..."
            print(f"    [RAG] {msg}")
            yield emit_sse("status", message=msg)

            response = get_llm_client().chat.completions.create(
                model=config.LIGHT_MODEL_ID,
                messages=[{"role": "user", "content": safe_format(
                    prompts.DIVIDE_SUB_PROMPT, context=paper_context, question=question)}],
                temperature=0.0,
                max_tokens=sub_max,
            )
            tracker.add(response, "divide_sub_answer")
            per_paper_answers.append(f"## {title}\n{response.choices[0].message.content}")

        synthesis = "\n\n---\n\n".join(per_paper_answers)
        prompt = safe_format(prompts.DIVIDE_SYNTHESIS_PROMPT,
                             per_paper_answers=synthesis, question=question)
        if history_text:
            prompt = history_text + "\n" + prompt

        print(f"    [RAG] Synthesizing {len(per_paper_answers)} sub-answers...")
        yield emit_sse("status", message=f"Synthesizing from {len(per_paper_answers)} papers...")

        answer = ""
        response = get_llm_client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.RAG_QA_TEMPERATURE,
            max_tokens=answer_tokens,
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                answer += chunk.choices[0].delta.content
                yield emit_sse("token", text=chunk.choices[0].delta.content)

        if session:
            session_mod.after_answer(question, rewritten_question, answer, session)

        yield emit_sse("done", usage=tracker.summary())
        return

    # Normal (non-divide-conquer) streaming path
    print("    [RAG] Generating answer...")
    yield emit_sse("status", message="Generating answer...")

    answer = ""
    response = get_llm_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": safe_format(prompts.QA_PROMPT,
            history=history_text, context=context, question=question)}],
        temperature=config.RAG_QA_TEMPERATURE,
        max_tokens=answer_tokens,
        stream=True,
    )
    for chunk in response:
        if chunk.choices[0].delta.content:
            answer += chunk.choices[0].delta.content
            yield emit_sse("token", text=chunk.choices[0].delta.content)

    if session:
        session_mod.after_answer(question, rewritten_question, answer, session)

    yield emit_sse("done", usage=tracker.summary())
