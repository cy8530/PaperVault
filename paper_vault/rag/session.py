"""Multi-turn conversation session management with progressive compaction."""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..config import config
from ..utils import get_llm_client, safe_format
from . import prompts as rag_prompts


# ── Data model ────────────────────────────────────────

@dataclass
class Turn:
    role: str                     # "user" | "assistant"
    question: str                 # original user question (empty for assistant)
    rewritten_question: str       # de-referenced version (empty if not rewritten)
    answer: str                   # assistant answer (empty for user)
    summary: str                  # 1-2 sentence summary of this exchange
    cited_papers: list[str] = field(default_factory=list)
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class Session:
    id: str
    name: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    turns: list[Turn] = field(default_factory=list)
    compact_count: int = 0        # how many times this session has been compacted

    def __post_init__(self):
        now = time.time()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


# ── Paths ─────────────────────────────────────────────

def _sessions_dir() -> Path:
    d = config.VAULT_DIR / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_path(session_id: str) -> Path:
    return _sessions_dir() / f"{session_id}.json"


# ── Serialization ──────────────────────────────────────

def _session_to_dict(s: Session) -> dict:
    return {
        "id": s.id,
        "name": s.name,
        "created_at": s.created_at,
        "updated_at": s.updated_at,
        "turns": [asdict(t) for t in s.turns],
        "compact_count": s.compact_count,
    }


def _session_from_dict(d: dict) -> Session:
    turns = [Turn(**t) for t in d.get("turns", [])]
    return Session(
        id=d["id"],
        name=d.get("name", ""),
        created_at=d.get("created_at", 0.0),
        updated_at=d.get("updated_at", 0.0),
        turns=turns,
        compact_count=d.get("compact_count", 0),
    )


# ── CRUD ───────────────────────────────────────────────

def create_session(name: str = "") -> Session:
    session_id = f"session_{int(time.time())}"
    s = Session(id=session_id, name=name)
    save_session(s)
    return s


def get_session(session_id: str) -> Session | None:
    path = _session_path(session_id)
    if not path.exists():
        return None
    try:
        return _session_from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def save_session(session: Session) -> None:
    session.updated_at = time.time()
    _session_path(session.id).write_text(
        json.dumps(_session_to_dict(session), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def delete_session(session_id: str) -> bool:
    path = _session_path(session_id)
    if path.exists():
        path.unlink()
        return True
    return False


def rename_session(session_id: str, name: str) -> bool:
    session = get_session(session_id)
    if not session:
        return False
    session.name = name.strip()
    save_session(session)
    return True


def list_sessions() -> list[dict]:
    sessions = []
    for path in sorted(_sessions_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            sessions.append({
                "id": d["id"],
                "name": d.get("name", ""),
                "turns": len(d.get("turns", [])),
                "created_at": d.get("created_at", 0),
                "updated_at": d.get("updated_at", 0),
                "compact_count": d.get("compact_count", 0),
            })
        except Exception:
            pass
    return sessions


# ── Token estimation ────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 3 chars for English, ~1 per 1.5 for Chinese."""
    return max(len(text) // 3, 1)


# ── Progressive compaction ──────────────────────────────

def _compress_to_summary(session: Session, keep_full: int) -> str | None:
    """Compress session history, keeping last *keep_full* rounds in full form.

    Returns a summary string if compaction happened, or None if no action needed.
    """
    # Identify assistant turns with answers (non-compacted)
    full_indices = [i for i, t in enumerate(session.turns)
                    if t.role == "assistant" and t.answer and not _is_compacted(t)]

    if len(full_indices) <= keep_full:
        return None

    # Old turns = all full assistant turns except the last keep_full
    old_indices = full_indices[:-keep_full]
    old_ids = set(old_indices)

    # Build context from old turns for LLM summary
    old_blocks = []
    for idx in old_indices:
        t = session.turns[idx]
        user_turn = _find_prev_user(session.turns, idx)
        question = user_turn.question if user_turn else ""
        old_blocks.append(f"Q: {question}\nA: {t.summary or t.answer[:300]}")
    old_text = "\n\n".join(old_blocks)

    try:
        response = get_llm_client().chat.completions.create(
            model=config.LIGHT_MODEL_ID,
            messages=[{"role": "user", "content": safe_format(
                rag_prompts.HISTORY_FULL_COMPACT_PROMPT, history=old_text
            )}],
            temperature=0.0,
            max_tokens=512,
        )
        summary = response.choices[0].message.content.strip()
    except Exception:
        summary = "\n".join(b.split("\n")[0] for b in old_blocks)[:500]

    # Replace old answers with compacted text, keep message order intact
    for idx in old_ids:
        session.turns[idx].answer = ""
        session.turns[idx].summary = summary
        session.turns[idx].cited_papers = []

    session.compact_count += 1
    return summary


def _is_compacted(turn: Turn) -> bool:
    """A turn is compacted if it's an assistant turn with summary but no answer."""
    return turn.role == "assistant" and not turn.answer and bool(turn.summary)


def _find_prev_user(turns: list[Turn], idx: int) -> Turn | None:
    for i in range(idx - 1, -1, -1):
        if turns[i].role == "user":
            return turns[i]
    return None


def build_history_for_prompt(session: Session) -> str:
    """Build conversation history text for injection into the RAG answer prompt.

    - Recent N rounds: full Q&A text
    - Older rounds: summary only (compacted marker has answer="" and summary non-empty)
    """
    if not session.turns:
        return ""

    parts = []
    for t in session.turns:
        if t.role == "user":
            current_question = t.rewritten_question or t.question
            parts.append(f"## User\n{current_question}")
        elif t.role == "assistant":
            if not t.answer and t.summary:
                parts.append(f"## Earlier conversation (summarized)\n{t.summary}")
            else:
                parts.append(f"## Assistant\n{t.answer}")
    return "\n\n".join(parts)


# ── Question rewriting ──────────────────────────────────

def rewrite_question(question: str, session: Session) -> tuple[str, str | None]:
    """Rewrite a new question into a self-contained form using session context.

    Returns (rewritten_question, summary_or_none).
    """
    if not session.turns:
        return question, None

    history_text = build_history_for_prompt(session)
    if not history_text.strip():
        return question, None

    # Build a compact context for the LLM
    context = ""
    for t in session.turns[-6:]:  # last 3 exchanges at most
        if t.role == "user":
            context += f"Q: {t.question}\n"
        elif t.role == "assistant" and t.summary:
            context += f"A: {t.summary}\n"

    if not context.strip():
        return question, None

    try:
        response = get_llm_client().chat.completions.create(
            model=config.LIGHT_MODEL_ID,
            messages=[{"role": "user", "content": safe_format(
                rag_prompts.QUESTION_REWRITE_PROMPT, history=context, question=question
            )}],
            temperature=0.0,
            max_tokens=256,
        )
        rewritten = response.choices[0].message.content.strip()
        # If LLM returns the same or empty, keep original
        if not rewritten or rewritten == question:
            return question, None
        return rewritten, None
    except Exception:
        return question, None


# ── Post-answer processing ──────────────────────────────

def after_answer(question: str, rewritten_question: str, answer: str, session: Session) -> None:
    """Record a Q&A exchange into the session and generate a summary of this round."""
    # Generate summary for this round
    try:
        response = get_llm_client().chat.completions.create(
            model=config.LIGHT_MODEL_ID,
            messages=[{"role": "user", "content": safe_format(
                rag_prompts.ROUND_SUMMARY_PROMPT, question=question, answer=answer[:1500]
            )}],
            temperature=0.0,
            max_tokens=128,
        )
        round_summary = response.choices[0].message.content.strip()
    except Exception:
        round_summary = question[:60]

    now = time.time()
    session.turns.append(Turn(
        role="user",
        question=question,
        rewritten_question=rewritten_question if rewritten_question != question else "",
        answer="",
        summary="",
        timestamp=now - 0.1,
    ))
    session.turns.append(Turn(
        role="assistant",
        question="",
        rewritten_question="",
        answer=answer,
        summary=round_summary,
        timestamp=now,
    ))

    # Auto-compact if needed
    _compress_to_summary(session, config.SESSION_KEEP_FULL_ROUNDS)
    save_session(session)
