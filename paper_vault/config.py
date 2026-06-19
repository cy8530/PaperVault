import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Pin HuggingFace cache to local vault before any model imports
_vault_dir = Path(os.environ.get("PAPER_VAULT_DIR", "vault"))
os.environ.setdefault("HF_HOME", str(_vault_dir / "models" / "huggingface"))

_SETTINGS_PATH = _vault_dir / "settings.json"


def _load_settings() -> dict:
    """Load web-persisted settings from vault/settings.json."""
    try:
        if _SETTINGS_PATH.exists():
            return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_settings(data: dict) -> None:
    """Persist settings to vault/settings.json."""
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# Module-level cache, loaded once at import. Config class body and reload_settings()
# both read from this, but via differently-scoped accessors.
_settings = _load_settings()


def _get(key: str, env: str, default: str) -> str:
    """Read from settings.json first, then env var, then default.
    Only uses settings value when key exists AND value is not None (JSON null).
    """
    val = _settings.get(key)
    if val is not None:
        return str(val)
    return os.environ.get(env, default)


def _get_int(key: str, env: str, default: str) -> int:
    return int(_get(key, env, default))


def _get_float(key: str, env: str, default: str) -> float:
    return float(_get(key, env, default))


class Config:
    _settings = _load_settings()

    # ── Filesystem ──────────────────────────────────────
    VAULT_DIR = Path(_get("vault_dir", "PAPER_VAULT_DIR", "vault"))
    IMPORT_DIRS_RAW = _get("import_dirs", "PAPER_VAULT_IMPORT_DIRS", "./papers")
    DEFAULT_IMPORT_DIRS = (
        [Path(p) for p in IMPORT_DIRS_RAW.split(":")]
        if isinstance(IMPORT_DIRS_RAW, str) else
        [Path(p) for p in IMPORT_DIRS_RAW]
    )
    EXTRACTED_DIR = Path(_get("extracted_dir", "", str(VAULT_DIR / "extracted")))
    NOTES_DIR = Path(_get("notes_dir", "", str(VAULT_DIR / "notes")))
    VECTORS_DIR = Path(_get("vectors_dir", "", str(VAULT_DIR / "vectors")))
    MODELS_DIR = Path(_get("models_dir", "", str(VAULT_DIR / "models")))

    # ── LLM ────────────────────────────────────────────
    LLM_BASE_URL = _get("llm_base_url", "OPENAI_BASE_URL", "")
    LLM_API_KEY = _get("llm_api_key", "OPENAI_API_KEY", "")
    LLM_MODEL = _get("llm_model", "MODEL_ID", "")
    LIGHT_MODEL_ID = _get("light_model_id", "LIGHT_MODEL_ID", LLM_MODEL)
    EMBEDDING_MODEL = _get("embedding_model", "EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

    # ── Import ──────────────────────────────────────────
    MAX_PDF_SIZE_MB = _get_int("max_pdf_mb", "PAPER_VAULT_MAX_PDF_MB", "0") or None
    MAX_UPLOAD_MB = _get_int("max_upload_mb", "PAPER_VAULT_MAX_UPLOAD_MB", "100")

    # ── Chunking ────────────────────────────────────────
    CHUNK_SIZE = _get_int("chunk_size", "PAPER_VAULT_CHUNK_SIZE", "800")
    CHUNK_OVERLAP = _get_int("chunk_overlap", "PAPER_VAULT_CHUNK_OVERLAP", "100")

    # ── Metadata extraction ─────────────────────────────
    METADATA_SNIPPET_CHARS = _get_int("metadata_snippet_chars", "PAPER_VAULT_METADATA_SNIPPET_CHARS", "6000")
    META_EXTRACT_MAX_TOKENS = _get_int("meta_extract_max_tokens", "PAPER_VAULT_META_EXTRACT_MAX_TOKENS", "512")

    # ── Note generation ─────────────────────────────────
    NOTE_GEN_MAX_TOKENS = _get_int("note_gen_max_tokens", "PAPER_VAULT_NOTE_GEN_MAX_TOKENS", "4096")
    NOTE_GEN_TEMPERATURE = _get_float("note_gen_temperature", "PAPER_VAULT_NOTE_GEN_TEMPERATURE", "0.3")
    NOTE_FILENAME_MAX_LEN = _get_int("note_filename_max_len", "PAPER_VAULT_NOTE_FILENAME_MAX_LEN", "150")

    # ── Dedup ───────────────────────────────────────────
    DEDUP_HASH_CHARS = _get_int("dedup_hash_chars", "PAPER_VAULT_DEDUP_HASH_CHARS", "5000")

    # ── RAG pipeline ────────────────────────────────────
    RAG_FILTER_MAX_TOKENS = _get_int("rag_filter_max_tokens", "PAPER_VAULT_RAG_FILTER_MAX_TOKENS", "256")
    RAG_JUDGE_MAX_TOKENS = _get_int("rag_judge_max_tokens", "PAPER_VAULT_RAG_JUDGE_MAX_TOKENS", "10")
    RAG_QA_TEMPERATURE = _get_float("rag_qa_temperature", "PAPER_VAULT_RAG_QA_TEMPERATURE", "0.3")
    RAG_SEARCH_BREADTH_MIN = _get_int("rag_search_breadth_min", "PAPER_VAULT_RAG_SEARCH_BREADTH_MIN", "10")
    RAG_DETAIL_MODERATE_DIVISOR = _get_int("rag_detail_moderate_divisor", "PAPER_VAULT_RAG_DETAIL_MODERATE_DIVISOR", "8")
    RAG_DETAIL_EXTENSIVE_DIVISOR = _get_int("rag_detail_extensive_divisor", "PAPER_VAULT_RAG_DETAIL_EXTENSIVE_DIVISOR", "3")
    RAG_DETAIL_MODERATE_MIN = _get_int("rag_detail_moderate_min", "PAPER_VAULT_RAG_DETAIL_MODERATE_MIN", "5")
    RAG_DETAIL_EXTENSIVE_MIN = _get_int("rag_detail_extensive_min", "PAPER_VAULT_RAG_DETAIL_EXTENSIVE_MIN", "15")
    RAG_DEFAULT_CHUNK_COUNT = _get_int("rag_default_chunk_count", "PAPER_VAULT_RAG_DEFAULT_CHUNK_COUNT", "50")
    RAG_SEARCH_DISTANCE_THRESHOLD = _get_float("rag_search_distance_threshold", "PAPER_VAULT_RAG_SEARCH_DISTANCE_THRESHOLD", "2.0")
    RAG_LEVEL1_MIN_CHUNKS = _get_int("rag_level1_min_chunks", "PAPER_VAULT_RAG_LEVEL1_MIN_CHUNKS", "3")
    RAG_QUERY_VARIANTS = _get_int("rag_query_variants", "PAPER_VAULT_RAG_QUERY_VARIANTS", "3")

    # ── Session / Multi-turn ──────────────────────────
    SESSION_KEEP_FULL_ROUNDS = _get_int("session_keep_full_rounds", "PAPER_VAULT_SESSION_KEEP_FULL", "3")
    CONTEXT_HISTORY_MAX_TOKENS = _get_int("context_history_max_tokens", "PAPER_VAULT_HISTORY_MAX_TOKENS", "2000")

    # ── Answer token budget tiers ───────────────────────
    ANSWER_TOKENS_TIER_1 = _get_int("answer_tokens_tier_1", "PAPER_VAULT_ANSWER_TOKENS_1", "1024")
    ANSWER_TOKENS_TIER_2 = _get_int("answer_tokens_tier_2", "PAPER_VAULT_ANSWER_TOKENS_2", "2048")
    ANSWER_TOKENS_TIER_3 = _get_int("answer_tokens_tier_3", "PAPER_VAULT_ANSWER_TOKENS_3", "3072")

    def __init__(self):
        for d in [self.VAULT_DIR, self.EXTRACTED_DIR, self.NOTES_DIR,
                   self.VECTORS_DIR, self.MODELS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        from .prompt_store import init as _init_prompts
        _init_prompts(self.VAULT_DIR / "prompts.json")

    @classmethod
    def _s(cls, key: str, env: str, default: str) -> str:
        """Same as module-level _get but reads from cls._settings."""
        val = cls._settings.get(key)
        if val is not None:
            return str(val)
        return os.environ.get(env, default)

    @classmethod
    def reload_settings(cls):
        """Re-read settings.json (called after web settings save)."""
        cls._settings = _load_settings()
        s = cls._s  # shorthand
        cls.VAULT_DIR = Path(s("vault_dir", "PAPER_VAULT_DIR", "vault"))
        raw = s("import_dirs", "PAPER_VAULT_IMPORT_DIRS", "./papers")
        cls.IMPORT_DIRS_RAW = raw
        cls.DEFAULT_IMPORT_DIRS = [Path(p) for p in raw.split(":")] if isinstance(raw, str) else [Path(p) for p in raw]
        cls.EXTRACTED_DIR = Path(s("extracted_dir", "", str(cls.VAULT_DIR / "extracted")))
        cls.NOTES_DIR = Path(s("notes_dir", "", str(cls.VAULT_DIR / "notes")))
        cls.VECTORS_DIR = Path(s("vectors_dir", "", str(cls.VAULT_DIR / "vectors")))
        cls.MODELS_DIR = Path(s("models_dir", "", str(cls.VAULT_DIR / "models")))
        cls.LLM_BASE_URL = s("llm_base_url", "OPENAI_BASE_URL", "")
        cls.LLM_API_KEY = s("llm_api_key", "OPENAI_API_KEY", "")
        cls.LLM_MODEL = s("llm_model", "MODEL_ID", "")
        cls.LIGHT_MODEL_ID = s("light_model_id", "LIGHT_MODEL_ID", cls.LLM_MODEL)
        cls.EMBEDDING_MODEL = s("embedding_model", "EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
        cls.MAX_PDF_SIZE_MB = int(s("max_pdf_mb", "PAPER_VAULT_MAX_PDF_MB", "0")) or None
        cls.MAX_UPLOAD_MB = int(s("max_upload_mb", "PAPER_VAULT_MAX_UPLOAD_MB", "100"))
        cls.CHUNK_SIZE = int(s("chunk_size", "PAPER_VAULT_CHUNK_SIZE", "800"))
        cls.CHUNK_OVERLAP = int(s("chunk_overlap", "PAPER_VAULT_CHUNK_OVERLAP", "100"))
        cls.METADATA_SNIPPET_CHARS = int(s("metadata_snippet_chars", "PAPER_VAULT_METADATA_SNIPPET_CHARS", "6000"))
        cls.META_EXTRACT_MAX_TOKENS = int(s("meta_extract_max_tokens", "PAPER_VAULT_META_EXTRACT_MAX_TOKENS", "512"))
        cls.NOTE_GEN_MAX_TOKENS = int(s("note_gen_max_tokens", "PAPER_VAULT_NOTE_GEN_MAX_TOKENS", "4096"))
        cls.NOTE_GEN_TEMPERATURE = float(s("note_gen_temperature", "PAPER_VAULT_NOTE_GEN_TEMPERATURE", "0.3"))
        cls.RAG_QA_TEMPERATURE = float(s("rag_qa_temperature", "PAPER_VAULT_RAG_QA_TEMPERATURE", "0.3"))
        cls.RAG_DETAIL_MODERATE_DIVISOR = int(s("rag_detail_moderate_divisor", "PAPER_VAULT_RAG_DETAIL_MODERATE_DIVISOR", "8"))
        cls.RAG_DETAIL_EXTENSIVE_DIVISOR = int(s("rag_detail_extensive_divisor", "PAPER_VAULT_RAG_DETAIL_EXTENSIVE_DIVISOR", "3"))
        cls.RAG_DETAIL_MODERATE_MIN = int(s("rag_detail_moderate_min", "PAPER_VAULT_RAG_DETAIL_MODERATE_MIN", "5"))
        cls.RAG_DETAIL_EXTENSIVE_MIN = int(s("rag_detail_extensive_min", "PAPER_VAULT_RAG_DETAIL_EXTENSIVE_MIN", "15"))
        cls.RAG_SEARCH_DISTANCE_THRESHOLD = float(s("rag_search_distance_threshold", "PAPER_VAULT_RAG_SEARCH_DISTANCE_THRESHOLD", "2.0"))
        cls.RAG_LEVEL1_MIN_CHUNKS = int(s("rag_level1_min_chunks", "PAPER_VAULT_RAG_LEVEL1_MIN_CHUNKS", "3"))
        cls.RAG_QUERY_VARIANTS = int(s("rag_query_variants", "PAPER_VAULT_RAG_QUERY_VARIANTS", "3"))
        cls.ANSWER_TOKENS_TIER_1 = int(s("answer_tokens_tier_1", "PAPER_VAULT_ANSWER_TOKENS_1", "1024"))
        cls.ANSWER_TOKENS_TIER_2 = int(s("answer_tokens_tier_2", "PAPER_VAULT_ANSWER_TOKENS_2", "2048"))
        cls.ANSWER_TOKENS_TIER_3 = int(s("answer_tokens_tier_3", "PAPER_VAULT_ANSWER_TOKENS_3", "3072"))
        cls.SESSION_KEEP_FULL_ROUNDS = int(s("session_keep_full_rounds", "PAPER_VAULT_SESSION_KEEP_FULL", "3"))
        cls.CONTEXT_HISTORY_MAX_TOKENS = int(s("context_history_max_tokens", "PAPER_VAULT_HISTORY_MAX_TOKENS", "2000"))


config = Config()
