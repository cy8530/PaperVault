"""Persistent prompt overrides — read/write vault/prompts.json.

Prompts can be edited via the Web UI Settings page. Changes take effect
immediately without server restart.
"""

from __future__ import annotations

import json
from pathlib import Path

_OVERRIDE_PATH: Path | None = None
_OVERRIDES: dict[str, str] = {}


def init(prompts_path: Path) -> None:
    global _OVERRIDE_PATH, _OVERRIDES
    _OVERRIDE_PATH = prompts_path
    _OVERRIDES = {}
    if _OVERRIDE_PATH.exists():
        try:
            _OVERRIDES = json.loads(_OVERRIDE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _OVERRIDES = {}


def get(name: str, default: str) -> str:
    """Return the prompt string for *name*, or *default* if no override exists."""
    return _OVERRIDES.get(name, default)


def get_all(defaults: dict[str, str]) -> dict[str, str]:
    """Return all prompts with overrides applied."""
    result = dict(defaults)
    result.update(_OVERRIDES)
    return result


def save(overrides: dict[str, str]) -> None:
    """Persist overrides to vault/prompts.json and update in-memory cache."""
    global _OVERRIDES
    if _OVERRIDE_PATH is None:
        return
    _OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OVERRIDE_PATH.write_text(json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8")
    _OVERRIDES = dict(overrides)
