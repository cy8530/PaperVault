"""Generate test questions and reference answers from paper content.

Test data is cached to ``benchmark/data/cache/`` so repeated runs against
the same papers avoid re-generating questions / reference answers.

Question types:
  - single:  one target paper, tagged by difficulty (1=shallow, 2=medium, 3=deep)
  - cross:   survey/comparison across multiple papers
"""
import json
import random
from pathlib import Path

from paper_vault.config import config
from paper_vault.indexer.store import get_indexed_paper_ids
from paper_vault.utils import get_llm_client, safe_format

_CACHE_DIR = Path(__file__).parent / "cache"

# ── Prompts ───────────────────────────────────────────────

_DIFFICULTY_SINGLE_PROMPT = """You are generating test questions for evaluating a RAG (Retrieval-Augmented Generation) system.

Given the following paper note, generate {n} questions that this paper can answer. Distribute across three difficulty levels (roughly equal split):

- 浅层 (shallow, 1): Simple factual recall — specific numbers, datasets, metric values. Answerable from a paper summary / abstract.
- 中等 (medium, 2): Method explanation or architectural reasoning — how does a method work, what is the core innovation, why was a design choice made. Needs some technical detail.
- 深层 (deep, 3): Cross-aspect synthesis or critical analysis — methodology rationale, design trade-offs, limitations, comparison to alternatives. Needs deep understanding of multiple sections.

Requirements:
1. All questions in Chinese
2. Specific: require factual answers, not just yes/no
3. Realistic: similar to what a real researcher would ask

Paper content:
---
{note_content}
---

Return ONLY a JSON array of objects with "question" and "difficulty" (1, 2, or 3):
[{{"question": "...", "difficulty": 1}}, ...]

JSON:"""

_CROSS_PAPER_PROMPT = """You are generating cross-paper survey and comparison questions for evaluating a RAG system's ability to synthesize information across multiple papers.

Below are summaries of {n} papers. Generate 3-5 questions that REQUIRE synthesizing or comparing information from at least TWO papers. Question types:

1. Survey/synthesis: "在 [领域] 中，不同方法分别如何处理 [问题]？"
2. Comparison: "对比 [Paper A] 和 [Paper B] 在 [维度] 上的差异"
3. Trend analysis: "从这些论文来看，[领域] 的发展趋势是什么？"

Paper summaries:
---
{paper_summaries}
---

Return ONLY a JSON array of question strings:
["question 1", "question 2", ...]

JSON:"""

_REFERENCE_ANSWER_PROMPT = """Answer the following question based on the full text provided below. Give a comprehensive, thorough answer, citing specific methods, numbers, and findings.

Question: {question}

Full text:
---
{full_text}
---

Answer (in Chinese, be thorough and complete):"""

_CROSS_REFERENCE_PROMPT = """Answer the following cross-paper question based on the paper contents provided below. Synthesize information across papers and compare where relevant.

Question: {question}

Paper contents:
---
{full_text}
---

Answer (in Chinese, be thorough and complete, cite paper titles when referencing specific findings):"""


# ── Paper access ─────────────────────────────────────────

def _get_papers(n: int) -> list[dict]:
    """Randomly sample ``n`` papers with notes available."""
    all_ids = list(get_indexed_paper_ids())
    if not all_ids:
        raise RuntimeError("No indexed papers found — run 'python pv.py import' first")
    sample = random.sample(all_ids, min(n, len(all_ids)))

    papers = []
    for pid in sample:
        note_path = _find_note_file(pid)
        if not note_path:
            continue
        note_content = note_path.read_text(encoding="utf-8")
        papers.append({
            "paper_id": pid,
            "title": _extract_title(note_content),
            "note_path": str(note_path),
            "note_content": note_content,
        })

    if not papers:
        raise RuntimeError("No papers with note files found")
    return papers


def _find_note_file(paper_id: str) -> Path | None:
    """Find a note file by paper_id (prefix match)."""
    for f in config.NOTES_DIR.glob("*.md"):
        if f.stem.startswith(paper_id[:80]) or paper_id[:80] in f.stem:
            return f
    try:
        from paper_vault.indexer import store
        info = store.get_paper_info(paper_id)
        if info and info.get("note_file"):
            np = config.NOTES_DIR / info["note_file"]
            if np.exists():
                return np
    except Exception:
        pass
    return None


def _find_extracted_file(paper_id: str) -> Path | None:
    """Find the full extracted text for a paper."""
    # Direct match: extracted/{paper_id}.md
    direct = config.EXTRACTED_DIR / f"{paper_id}.md"
    if direct.exists():
        return direct
    # Prefix match (some extracted files use title-based naming)
    for f in config.EXTRACTED_DIR.glob("*.md"):
        if f.stem.startswith(paper_id[:60]) or paper_id[:60] in f.stem:
            return f
    return None


def _extract_title(note_content: str) -> str:
    """Extract title from note (first # heading)."""
    for line in note_content.split("\n"):
        if line.startswith("# ") and len(line) > 3:
            return line[2:].strip()
    return ""


# ── Generators ────────────────────────────────────────────

def generate_questions_with_difficulty(paper: dict, n: int = 5) -> list[dict]:
    """Generate ``n`` difficulty-tagged questions for one paper.

    Returns list of ``{"question": str, "difficulty": int}``.
    """
    prompt = safe_format(
        _DIFFICULTY_SINGLE_PROMPT.replace("{n}", str(n)),
        note_content=paper["note_content"][:8000],
    )
    response = get_llm_client().chat.completions.create(
        model=config.LIGHT_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=768,
    )
    raw = response.choices[0].message.content.strip()

    from paper_vault.utils import parse_llm_json
    result = parse_llm_json(raw)
    if isinstance(result, list) and len(result) > 0:
        items = []
        for r in result[:n]:
            if isinstance(r, dict) and "question" in r:
                diff = r.get("difficulty", 2)
                if isinstance(diff, (int, float)) and 1 <= diff <= 3:
                    items.append({"question": r["question"], "difficulty": int(diff)})
                else:
                    items.append({"question": r["question"], "difficulty": 2})
            elif isinstance(r, str):
                items.append({"question": r, "difficulty": 2})
        return items

    # Fallback
    return [{"question": f"What are the main contributions of {paper['paper_id']}?",
             "difficulty": 2}]


def generate_cross_paper_questions(papers: list[dict], n: int = 4) -> list[str]:
    """Generate survey/comparison questions spanning multiple papers.

    Returns list of question strings.
    """
    summaries = []
    for i, p in enumerate(papers):
        title = p.get("title", p["paper_id"])[:80]
        # Take key sections: title + first 300 chars of note
        snippet = p["note_content"][:400].replace("\n", " ")
        summaries.append(f"[{i+1}] {title}\n   {snippet}...")

    prompt = safe_format(
        _CROSS_PAPER_PROMPT.replace("{n}", str(len(papers))),
        paper_summaries="\n\n".join(summaries),
    )
    response = get_llm_client().chat.completions.create(
        model=config.LIGHT_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()

    from paper_vault.utils import parse_llm_json
    result = parse_llm_json(raw)
    if isinstance(result, list):
        return [q for q in result if isinstance(q, str)][:n]

    import re
    found = re.findall(r'"([^"]+)"', raw)
    return found[:n] if found else ["Compare the main methods across these papers."]


def generate_reference_answer(question: str, paper: dict | None = None,
                               all_papers: list[dict] | None = None) -> str:
    """Generate a reference answer from **full extracted text** (not notes).

    For single-paper questions, pass *paper* — reads extracted/{paper_id}.md.
    For cross-paper questions, pass *all_papers* — concatenates extracted texts.
    Falls back to note_content if extracted file is not found.
    """
    if all_papers:
        # Cross-paper: concatenate extracted texts (up to 8K chars per paper)
        parts = []
        for p in all_papers:
            title = p.get("title", p["paper_id"])[:80]
            ext_path = _find_extracted_file(p["paper_id"])
            if ext_path:
                text = ext_path.read_text(encoding="utf-8")[:8000]
            else:
                text = p["note_content"][:4000]
            parts.append(f"## {title}\n\n{text}")
        full_text = "\n\n---\n\n".join(parts)
        prompt = safe_format(_CROSS_REFERENCE_PROMPT, question=question, full_text=full_text)
        max_tokens = 2048
    elif paper:
        # Single-paper: use extracted full text (up to 30K chars)
        ext_path = _find_extracted_file(paper["paper_id"])
        if ext_path:
            full_text = ext_path.read_text(encoding="utf-8")[:30000]
        else:
            full_text = paper["note_content"][:15000]
        prompt = safe_format(_REFERENCE_ANSWER_PROMPT, question=question,
                            full_text=full_text)
        max_tokens = 1536
    else:
        return ""

    response = get_llm_client().chat.completions.create(
        model=config.LIGHT_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ── Cache management ──────────────────────────────────────

def _cache_path(name: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{name}.json"


def load_cache(name: str) -> dict | None:
    path = _cache_path(name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_cache(name: str, data: dict) -> None:
    _cache_path(name).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Main builder ──────────────────────────────────────────

def build_test_data(n_papers: int = 5, questions_per_paper: int = 5,
                    cross_paper_questions: int = 3,
                    use_cache: bool = True,
                    use_cache_ref: bool = True) -> dict:
    """Build the full test dataset.

    Args:
        use_cache: if False, regenerate questions AND reference answers
        use_cache_ref: if False, regenerate only reference answers (keep cached questions).
                       Only effective when use_cache=True.

    Returns dict with keys:
      - "single": list of single-paper items (paper_id, question, difficulty, reference_answer, note_content)
      - "cross":  list of cross-paper items (question, relevant_paper_ids, reference_answer, all_notes)
      - "papers": list of sampled paper dicts
    """
    papers = _get_papers(n_papers)
    _use_ref_cache = use_cache and use_cache_ref

    # ── Single-paper questions (difficulty-tagged) ───────
    single_items = []
    for paper in papers:
        qkey = f"questions_v2_{paper['paper_id']}"
        cached = load_cache(qkey) if use_cache else None
        if cached:
            questions = cached.get("questions", [])
        else:
            questions = generate_questions_with_difficulty(paper, questions_per_paper)
            save_cache(qkey, {"questions": questions})

        for i, qd in enumerate(questions):
            q_text = qd["question"]
            difficulty = qd.get("difficulty", 2)
            rkey = f"ref_full_{paper['paper_id']}_{i}"
            cached_ref = load_cache(rkey) if _use_ref_cache else None
            if cached_ref:
                ref = cached_ref.get("reference_answer", "")
            else:
                ref = generate_reference_answer(q_text, paper=paper)
                save_cache(rkey, {"reference_answer": ref})

            single_items.append({
                "type": "single",
                "paper_id": paper["paper_id"],
                "question": q_text,
                "difficulty": difficulty,
                "reference_answer": ref,
                "note_content": paper["note_content"],
            })

    # ── Cross-paper questions ────────────────────────────
    cross_items = []
    if cross_paper_questions > 0 and len(papers) >= 2:
        ckey = f"cross_full_{'_'.join(p['paper_id'][:30] for p in papers)}"
        cached_cross = load_cache(ckey) if use_cache else None

        if cached_cross and _use_ref_cache:
            # Full cache hit: questions + references
            cross_questions = cached_cross.get("questions", [])
            cross_refs = cached_cross.get("references", {})
        elif cached_cross:
            # Questions cached but need fresh reference answers
            cross_questions = cached_cross.get("questions", [])
            cross_refs = {}
            for q in cross_questions:
                cross_refs[q] = generate_reference_answer(q, all_papers=papers)
            save_cache(ckey, {"questions": cross_questions, "references": cross_refs})
        else:
            # Regenerate everything
            cross_questions = generate_cross_paper_questions(papers, cross_paper_questions)
            cross_refs = {}
            for q in cross_questions:
                cross_refs[q] = generate_reference_answer(q, all_papers=papers)
            save_cache(ckey, {"questions": cross_questions, "references": cross_refs})

        for q_text in cross_questions:
            ref = cross_refs.get(q_text, "")
            cross_items.append({
                "type": "cross",
                "question": q_text,
                "relevant_paper_ids": [p["paper_id"] for p in papers],
                "reference_answer": ref,
                # Collect all notes for context
                "all_notes": {p["paper_id"]: p["note_content"] for p in papers},
            })

    return {
        "single": single_items,
        "cross": cross_items,
        "papers": papers,
    }


def clear_cache() -> int:
    """Delete all cached test data. Returns number of files removed."""
    if not _CACHE_DIR.exists():
        return 0
    count = 0
    for f in _CACHE_DIR.glob("*.json"):
        f.unlink()
        count += 1
    return count
