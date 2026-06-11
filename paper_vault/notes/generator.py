from ..config import config
from ..usage import tracker
from ..utils import get_llm_client, safe_format, parse_llm_json, strip_code_fences
from . import prompts


def generate_note(text: str) -> str:
    """Generate a structured reading note from extracted paper text."""
    response = get_llm_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": safe_format(prompts.NOTE_PROMPT, text=text)}],
        temperature=config.NOTE_GEN_TEMPERATURE,
        max_tokens=config.NOTE_GEN_MAX_TOKENS,
    )
    tracker.add(response, "note_generation")
    content = response.choices[0].message.content.strip()
    return strip_code_fences(content)


def extract_paper_metadata(text: str) -> tuple[dict, str]:
    """Extract title, authors, year, venue, keywords via LLM.

    Returns (metadata_dict, method) where method is always 'llm'.
    On repeated failure, returns empty dict with a logged warning.
    """
    snippet = text[:config.METADATA_SNIPPET_CHARS]

    for attempt in (1, 2):
        temperature = 0.0 if attempt == 1 else 0.3
        response = get_llm_client().chat.completions.create(
            model=config.LIGHT_MODEL_ID,
            messages=[{"role": "user", "content": safe_format(prompts.META_PROMPT, text=snippet)}],
            temperature=temperature,
            max_tokens=config.META_EXTRACT_MAX_TOKENS,
        )
        tracker.add(response, "metadata_extraction")
        raw = response.choices[0].message.content.strip()
        result = parse_llm_json(raw)
        if result is not None:
            return result, "llm"

    print(f"  [WARN] LLM metadata extraction failed after 2 attempts — title/authors will be empty")
    return {"title": "", "authors": [], "year": None, "venue": None, "keywords": []}, "llm"
