"""RAG Q&A prompts — filter, judge, section matching, and final answer.

Defaults are defined here. Overrides stored in vault/prompts.json take
precedence. Call ``reload()`` after saving overrides to apply them without
a server restart.
"""

from ..prompt_store import get as _get_prompt

# ── Defaults ───────────────────────────────────────────

_DEFAULT_QA_PROMPT = """You are a research assistant. Answer the question based ONLY on the provided paper excerpts. Each excerpt has a source label in [brackets].

{history}

Rules:
- Answer in Chinese
- Use [Paper: ...] excerpts for background and high-level understanding
- Use [Detail: ...] excerpts for specific numbers, formulas, and implementation facts — prefer these for factual claims
- Cite sources with format: 【Title (Year)】
- If comparing multiple papers, use a table when helpful
- If the excerpts don't contain enough information to answer, say so explicitly: "当前知识库中的论文未涉及此问题的足够信息"
- After your answer, add a "## Sources" section listing each paper you cited with its title and year
- Self-check: every factual claim in your answer must be traceable to an excerpt above

Paper excerpts:
---
{context}
---

Question: {question}

Answer:"""

_DEFAULT_NEED_DETAILS_PROMPT = """You are evaluating how much additional technical detail is needed to answer a question.

Question: {question}

Provided notes:
---
{context}
---

How much additional detail from the full paper text is needed?
1 = notes are sufficient, no extra detail needed
2 = need key details only (core formulas, main experimental numbers, critical implementation specifics)
3 = need extensive detail (full methodology breakdown, all experiments, complete derivations)

Reply with ONLY one digit: 1, 2, or 3."""

_DEFAULT_BATCH_SECTION_MATCH_PROMPT = """Given a research question and multiple papers' section structures, identify which sections of each paper are most likely to contain the answer.

Question: {question}

Papers:
{papers}

Return a JSON object keyed by paper title, each value is an array of section heading names that are relevant. Return empty array for a paper if all sections seem relevant or if uncertain.

JSON:"""

# Combined rewrite + search-term extraction (replaces two separate LLM calls)
_DEFAULT_QUESTION_PREPROCESS_PROMPT = """Given conversation history and a new question:

1. Rewrite the question to be self-contained (resolve pronouns like "it", "this", "they", "the paper").
2. Extract 3-5 key technical terms or concepts from the history that would improve semantic search retrieval.

History:
{history}

New question: {question}

Return JSON: {{"rewritten": "...", "search_terms": "term1, term2, ..."}}"""

_DEFAULT_QUESTION_REWRITE_PROMPT = """Given a conversation history and a new question, rewrite the question to be self-contained.

Resolve all pronouns, implicit references, and deictic expressions (e.g. "it", "this method", "the second paper", "their approach") by replacing them with the specific entity names from the history.

If the new question is completely unrelated to the history, return the original question unchanged. If it IS related, rewrite it so it can be understood without the history.

History:
---
{history}
---

New question: {question}

Rewritten question:"""

_DEFAULT_ROUND_SUMMARY_PROMPT = """Summarize this Q&A exchange in 1-2 concise sentences in Chinese. Capture what was asked and what was answered (key papers, methods, or findings discussed).

Question: {question}

Answer:
---
{answer}
---

Summary (1-2 sentences):"""

_DEFAULT_HISTORY_FULL_COMPACT_PROMPT = """Summarize this conversation history in a compact Chinese paragraph. Preserve:
1. The main research topics discussed
2. Which papers were referenced and what about them
3. Key conclusions reached

History:
---
{history}
---

Compact summary:"""

_DEFAULT_FILTER_PAPERS_PROMPT = """Given a question and a list of candidate papers, select which papers are relevant to answering the question.

Question: {question}

Papers:
{papers}

Return ONLY a JSON array of indices (1-based) of the papers that are relevant to the question. Include a paper only if it directly addresses the question's topic. Example: [1, 3, 5]

JSON:"""

# ── Public API ─────────────────────────────────────────

QA_PROMPT = _get_prompt("qa_prompt", _DEFAULT_QA_PROMPT)
NEED_DETAILS_PROMPT = _get_prompt("need_details_prompt", _DEFAULT_NEED_DETAILS_PROMPT)
BATCH_SECTION_MATCH_PROMPT = _get_prompt("batch_section_match_prompt", _DEFAULT_BATCH_SECTION_MATCH_PROMPT)
FILTER_PAPERS_PROMPT = _get_prompt("filter_papers_prompt", _DEFAULT_FILTER_PAPERS_PROMPT)
QUESTION_PREPROCESS_PROMPT = _get_prompt("question_preprocess_prompt", _DEFAULT_QUESTION_PREPROCESS_PROMPT)
QUESTION_REWRITE_PROMPT = _get_prompt("question_rewrite_prompt", _DEFAULT_QUESTION_REWRITE_PROMPT)
ROUND_SUMMARY_PROMPT = _get_prompt("round_summary_prompt", _DEFAULT_ROUND_SUMMARY_PROMPT)
HISTORY_FULL_COMPACT_PROMPT = _get_prompt("history_full_compact_prompt", _DEFAULT_HISTORY_FULL_COMPACT_PROMPT)

DEFAULTS = {
    "qa_prompt": _DEFAULT_QA_PROMPT,
    "need_details_prompt": _DEFAULT_NEED_DETAILS_PROMPT,
    "batch_section_match_prompt": _DEFAULT_BATCH_SECTION_MATCH_PROMPT,
    "filter_papers_prompt": _DEFAULT_FILTER_PAPERS_PROMPT,
    "question_preprocess_prompt": _DEFAULT_QUESTION_PREPROCESS_PROMPT,
    "question_rewrite_prompt": _DEFAULT_QUESTION_REWRITE_PROMPT,
    "round_summary_prompt": _DEFAULT_ROUND_SUMMARY_PROMPT,
    "history_full_compact_prompt": _DEFAULT_HISTORY_FULL_COMPACT_PROMPT,
}


def reload() -> None:
    """Re-read prompt overrides from vault/prompts.json (no restart needed)."""
    global QA_PROMPT, NEED_DETAILS_PROMPT, BATCH_SECTION_MATCH_PROMPT, FILTER_PAPERS_PROMPT
    global QUESTION_PREPROCESS_PROMPT, QUESTION_REWRITE_PROMPT, ROUND_SUMMARY_PROMPT, HISTORY_FULL_COMPACT_PROMPT
    from ..prompt_store import _OVERRIDES, _OVERRIDE_PATH
    if _OVERRIDE_PATH and _OVERRIDE_PATH.exists():
        try:
            import json
            _OVERRIDES.clear()
            _OVERRIDES.update(json.loads(_OVERRIDE_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    QA_PROMPT = _get_prompt("qa_prompt", _DEFAULT_QA_PROMPT)
    NEED_DETAILS_PROMPT = _get_prompt("need_details_prompt", _DEFAULT_NEED_DETAILS_PROMPT)

    BATCH_SECTION_MATCH_PROMPT = _get_prompt("batch_section_match_prompt", _DEFAULT_BATCH_SECTION_MATCH_PROMPT)
    FILTER_PAPERS_PROMPT = _get_prompt("filter_papers_prompt", _DEFAULT_FILTER_PAPERS_PROMPT)
    QUESTION_PREPROCESS_PROMPT = _get_prompt("question_preprocess_prompt", _DEFAULT_QUESTION_PREPROCESS_PROMPT)
    QUESTION_REWRITE_PROMPT = _get_prompt("question_rewrite_prompt", _DEFAULT_QUESTION_REWRITE_PROMPT)
    ROUND_SUMMARY_PROMPT = _get_prompt("round_summary_prompt", _DEFAULT_ROUND_SUMMARY_PROMPT)
    HISTORY_FULL_COMPACT_PROMPT = _get_prompt("history_full_compact_prompt", _DEFAULT_HISTORY_FULL_COMPACT_PROMPT)
