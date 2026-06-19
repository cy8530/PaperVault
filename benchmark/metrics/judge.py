"""LLM-as-judge evaluation — faithfulness, coverage, context relevance.

Uses the project's configured LLM (LIGHT_MODEL_ID) to judge RAG answer quality
by comparing the answer against retrieved contexts and a reference answer.

Zero new dependencies — uses the same OpenAI client and config as the project.
"""
from paper_vault.config import config
from paper_vault.utils import get_llm_client, safe_format
from paper_vault.rag.qa import _search_and_filter, _format_context, _determine_detail
from paper_vault.indexer.store import search_chunks_for_papers
from paper_vault.indexer.embedder import embed_texts

# ── Judge prompts ─────────────────────────────────────────

_FAITHFULNESS_PROMPT = """You are evaluating whether an AI's answer is faithful to the provided source texts.

Task: For each factual claim in the answer, determine whether it is supported by the source texts below. Then compute an overall faithfulness score.

Source texts (these are the ONLY sources the AI had access to):
---
{contexts}
---

Answer to evaluate:
---
{answer}
---

Analysis format (in Chinese):

**逐条检查：**
列出答案中的每条事实性断言，一行一条，标记 [有依据] 或 [无依据]，并说明对应的出处文本或为什么找不到支撑。

**总结：**
有依据的断言数 / 总断言数 = X/N

**综合评分 (0-1):**

Score:"""

_COVERAGE_PROMPT = """You are evaluating how well an AI answer covers the key information compared to a comprehensive reference answer.

Reference answer (generated from the FULL paper text — this is the gold standard):
---
{reference}
---

AI answer (generated via RAG retrieval from partial paper excerpts):
---
{answer}
---

Analysis format (in Chinese):

**参考回答中的关键信息点：**
列出参考回答中的关键信息点（逐条编号）

**覆盖检查：**
对每条关键信息点，标记 [已覆盖] 或 [未覆盖]，简要说明原因

**额外信息：**
AI 回答中是否有参考回答中未提及的信息？如有，列出（可能是补充见解，也可能偏离主题）

**综合评分 (0-1):**

Score:"""

_CONTEXT_RELEVANCE_PROMPT = """You are evaluating whether retrieved text passages are relevant to answering a question.

Question: {question}

Retrieved passages:
---
{contexts}
---

For each passage, rate its relevance to the question:
- 2 = 高度相关 — 直接有助于回答问题
- 1 = 部分相关 — 提供背景信息或间接相关
- 0 = 不相关

Analysis format (in Chinese):
对每条 passage 给出评分和一句话理由，最后给出平均分。

**综合评分 (0-1):**

Score:"""


def _call_judge(prompt: str) -> str:
    """Call the lightweight LLM for judging."""
    try:
        response = get_llm_client().chat.completions.create(
            model=config.LIGHT_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"JUDGE_ERROR: {e}"


def _parse_judge_score(raw: str) -> tuple[float, str]:
    """Extract a 0-1 score from judge output, plus the full analysis text."""
    score = 0.0
    # Try to find score after "Score:" or at end of "综合评分"
    import re
    # Match patterns like "Score: 0.85", "综合评分: 0.85", "**综合评分 (0-1):** 0.85"
    patterns = [
        r'\*?\*?综合评分\*?\*?\s*\(?0-?1\)?\s*[:：]\s*(\d+\.?\d*)',
        r'Score\s*[:：]\s*(\d+\.?\d*)',
        r'(\d+\.\d+)\s*$',
    ]
    for pat in patterns:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            try:
                score = float(m.group(1))
                if 0 <= score <= 1:
                    return score, raw
            except ValueError:
                pass
    return score, raw


def evaluate_faithfulness(answer: str, contexts: list[str]) -> dict:
    """Check if answer claims are supported by the source contexts.

    Returns: {score: float, analysis: str}
    """
    contexts_text = "\n\n---\n\n".join(
        f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)
    )
    prompt = safe_format(
        _FAITHFULNESS_PROMPT,
        contexts=contexts_text,
        answer=answer,
    )
    raw = _call_judge(prompt)
    score, analysis = _parse_judge_score(raw)
    return {"score": score, "analysis": analysis}


def evaluate_coverage(answer: str, reference: str) -> dict:
    """Check how well the RAG answer covers key points from the reference answer.

    Returns: {score: float, analysis: str}
    """
    prompt = safe_format(
        _COVERAGE_PROMPT,
        reference=reference,
        answer=answer,
    )
    raw = _call_judge(prompt)
    score, analysis = _parse_judge_score(raw)
    return {"score": score, "analysis": analysis}


def evaluate_context_relevance(question: str, contexts: list[str]) -> dict:
    """Check how relevant each retrieved context passage is to the question.

    Returns: {score: float, analysis: str}
    """
    contexts_text = "\n\n---\n\n".join(
        f"[Passage {i+1}]\n{c[:1000]}" for i, c in enumerate(contexts)
    )
    prompt = safe_format(
        _CONTEXT_RELEVANCE_PROMPT,
        question=question,
        contexts=contexts_text,
    )
    raw = _call_judge(prompt)
    score, analysis = _parse_judge_score(raw)
    return {"score": score, "analysis": analysis}


def get_rag_context(question: str, n_papers: int = 5,
                    detail: str = "auto") -> tuple[str, list[str], list[str]]:
    """Run the RAG pipeline and collect answer + raw contexts.

    Args:
        question: The question to answer.
        n_papers: Number of papers to retrieve.
        detail: Detail level — "auto", "1", "2", "3", or "all".

    Returns (answer, note_contexts, chunk_contexts).
    """
    # Use _search_and_filter to get the raw notes data
    notes_results = _search_and_filter(question, n_papers, where=None)

    if not notes_results:
        return "", [], []

    note_texts = [nr.get("note", "") for nr in notes_results if nr.get("note")]

    # Determine detail and search chunks
    detail_level, full_text = _determine_detail(question, notes_results, detail)
    chunk_texts = []

    if detail_level > 1 or full_text:
        q_vec = embed_texts([question], is_query=True)[0]
        for nr in notes_results:
            chunks = search_chunks_for_papers(
                q_vec, [nr["paper_id"]],
                per_paper=None if full_text else 5,
            )
            for c in chunks:
                chunk_texts.append(c.get("text", ""))

    # Build the answer
    from paper_vault.rag.qa import _format_context
    notes_ctx = _format_context(notes_results, "Source: notes", "note")
    full_context = notes_ctx
    if chunk_texts:
        chunks_as_dicts = [{"paper_id": "", "chunk_idx": i, "text": t, "title": ""}
                           for i, t in enumerate(chunk_texts)]
        full_context += "\n\n---\n\n" + _format_context(chunks_as_dicts, "Detail", "text")

    # Generate answer
    from paper_vault.rag import prompts as rag_prompts
    prompt = safe_format(
        rag_prompts.QA_PROMPT,
        history="",
        context=full_context,
        question=question,
    )
    response = get_llm_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.RAG_QA_TEMPERATURE,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content.strip()

    return answer, note_texts, chunk_texts
