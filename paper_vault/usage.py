"""Token usage tracking — LLM API tokens + estimated text tokens."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 3 chars for English, ~1 per 1.5 for Chinese."""
    if not text:
        return 0
    cjk = len(re.findall(r'[一-鿿㐀-䶿]', text))
    other = len(text) - cjk
    return max(cjk // 2 + other // 3, 1)


@dataclass
class UsageTracker:
    """Accumulate token usage across a session."""

    # LLM API tokens (from OpenAI response.usage)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    # Estimated input tokens (chunks + notes + history + question)
    estimated_input_tokens: int = 0

    _label: str = ""

    def add(self, response: Any, label: str = "") -> None:
        """Record usage from an OpenAI API response."""
        usage = response.usage
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.calls += 1

    def add_input_text(self, text: str) -> None:
        """Estimate tokens for text content fed into the LLM context."""
        self.estimated_input_tokens += estimate_tokens(text)

    @property
    def total_llm_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def total_tokens(self) -> int:
        """Total estimated consumption: LLM API tokens + input text tokens."""
        return self.total_llm_tokens + self.estimated_input_tokens

    def summary(self) -> str:
        parts = [f"[Usage] {self.calls} LLM calls"]
        if self.estimated_input_tokens:
            parts.append(
                f"context ~{self.estimated_input_tokens:,} tokens"
            )
        parts.append(
            f"LLM: {self.prompt_tokens:,} in + {self.completion_tokens:,} out "
            f"= {self.total_llm_tokens:,} tokens"
        )
        if self.estimated_input_tokens:
            parts.append(f"total ~{self.total_tokens:,} tokens (estimated)")
        return ", ".join(parts)

    def reset(self) -> None:
        """Reset all counters to zero (for per-query tracking)."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.estimated_input_tokens = 0


# Global tracker for this session
tracker = UsageTracker()
