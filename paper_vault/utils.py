"""Shared utilities used across paper_vault modules."""

import json
import re

import httpx
from openai import OpenAI
from .config import config


# ── LLM client ───────────────────────────────────────

_LLM_CLIENT = None

# Total timeout 120s, connect timeout 15s — prevents indefinite hangs
# when the API endpoint is unreachable or unresponsive.
_CLIENT_TIMEOUT = httpx.Timeout(120.0, connect=15.0)


def get_llm_client() -> OpenAI:
    """Lazy singleton OpenAI client shared across the project."""
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            timeout=_CLIENT_TIMEOUT,
        )
    return _LLM_CLIENT


# ── String utilities ─────────────────────────────────

def safe_format(template: str, **kwargs) -> str:
    """Format a string template by replacing {key} placeholders with values.

    Uses str.replace() instead of str.format() so LaTeX/JSON braces in values
    are never interpreted as format placeholders.
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def strip_code_fences(text: str) -> str:
    """Strip opening/closing ``` fences from LLM output.

    Handles ```lang ... ``` and bare ``` ... ``` patterns.
    """
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence line (may include language tag e.g. ```json)
        text = text.split("\n", 1)[-1]
        # Remove closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def parse_llm_json(raw: str):
    """Parse JSON from an LLM response, stripping ``` fences if present.

    Returns the parsed object, or None if parsing fails.
    """
    raw = strip_code_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def emit_sse(event: str, **kwargs) -> str:
    """Format a Server-Sent Events line with JSON payload."""
    return f"data: {json.dumps({'event': event, **kwargs}, ensure_ascii=False)}\n\n"


def normalize_id(name: str) -> str:
    """Normalize a paper filename stem into a safe, consistent paper_id.

    - Replaces whitespace and hyphens with underscores
    - Removes characters unsafe for SQL/filenames
    - Collapses repeated underscores
    - Truncates to 120 chars
    """
    s = name.strip()
    s = re.sub(r'[\s\-]+', '_', s)
    s = re.sub(r'[^a-zA-Z0-9_.]', '', s)
    s = re.sub(r'_+', '_', s).strip('_')
    if len(s) > 120:
        s = s[:120].rstrip('_')
    return s or "paper"


def sanitize_part(s: str) -> str:
    """Remove characters invalid in filenames and collapse whitespace."""
    s = re.sub(r'[/\\:*?"<>|]', '', s)
    s = re.sub(r'\s+', '_', s)
    return re.sub(r'_+', '_', s).strip('_')


