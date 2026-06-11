"""Token usage tracking."""

import time
from dataclasses import dataclass, field


@dataclass
class UsageTracker:
    """Accumulate token usage across a session."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    _label: str = ""

    def add(self, response, label: str = ""):
        """Record usage from an OpenAI API response."""
        usage = response.usage
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> str:
        return (
            f"[Usage] {self.calls} calls, "
            f"{self.prompt_tokens:,} in + {self.completion_tokens:,} out "
            f"= {self.total_tokens:,} tokens"
        )


# Global tracker for this session
tracker = UsageTracker()
