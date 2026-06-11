"""Note generation and metadata extraction prompts.

Defaults are defined here. Overrides stored in vault/prompts.json take
precedence. Call ``reload()`` after saving overrides to apply them without
a server restart.
"""

from ..prompt_store import get as _get_prompt

# ── Defaults ───────────────────────────────────────────

_DEFAULT_NOTE_PROMPT = """You are a research assistant. Read the following paper content extracted from a PDF and produce a structured reading note in Markdown. This is a summary note, not an exhaustive transcription — focus on the core ideas and key results.

Requirements:
- Write in Chinese
- Title, authors, publication year/venue (must include if visible in the text)
- Core problem statement: what gap does this paper fill? Why is it important? (2-4 sentences)
- Proposed method/approach: describe each component and how they connect. Include 1-2 key formulas (using `$...$` or `$$...$$`) inline where they help explain the method, with brief explanation of variables. (5-8 bullet points)
- Key implementation details: architecture, hyperparameters, training setup, datasets (3-5 bullet points)
- Key findings and results: specific numbers/metrics from experiments. Mention important table results inline rather than describing every table separately. (3-5 bullet points)
- Limitations mentioned by the authors (if any, 1-2 sentences)
- Keep the note concise — aim for quality over quantity
- Output the note directly. Do NOT wrap your response in code fences (```markdown or ```). Start with the title/heading, not a code block.

Paper content:
---
{text}
---

Structured reading note:"""

_DEFAULT_META_PROMPT = """Extract structured metadata from this paper text. Return ONLY a JSON object, no other text.

{{
  "title": "full paper title",
  "authors": ["Author Name 1", "Author Name 2"],
  "year": 2024,
  "venue": "conference or journal name",
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}

Rules:
- title: the ACTUAL paper title (usually the first large bold text on page 1). If you cannot find a real paper title, set to null. Do NOT use page headers, footers, journal metadata (e.g. "Vol. XX", "DOI: XX"), or running titles as the paper title.
- year: integer, extract from publication info if visible, otherwise null
- venue: conference (e.g. NeurIPS, ICML, CVPR) or journal name if visible, otherwise null
- authors: list of full names, limit to first 6
- keywords: 3-5 key topics/techniques mentioned
- If the text appears to be a table of contents, cover page, or reference list rather than a paper body, set title to null

Paper text (first portion):
---
{text}
---

JSON:"""

# ── Public API ─────────────────────────────────────────

NOTE_PROMPT = _get_prompt("note_prompt", _DEFAULT_NOTE_PROMPT)
META_PROMPT = _get_prompt("meta_prompt", _DEFAULT_META_PROMPT)

DEFAULTS = {
    "note_prompt": _DEFAULT_NOTE_PROMPT,
    "meta_prompt": _DEFAULT_META_PROMPT,
}


def reload() -> None:
    """Re-read prompt overrides from vault/prompts.json (no restart needed)."""
    global NOTE_PROMPT, META_PROMPT
    from ..prompt_store import _OVERRIDES, _OVERRIDE_PATH
    if _OVERRIDE_PATH and _OVERRIDE_PATH.exists():
        try:
            import json
            _OVERRIDES.clear()
            _OVERRIDES.update(json.loads(_OVERRIDE_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    NOTE_PROMPT = _get_prompt("note_prompt", _DEFAULT_NOTE_PROMPT)
    META_PROMPT = _get_prompt("meta_prompt", _DEFAULT_META_PROMPT)
