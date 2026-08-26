from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from ..config import config

_MODEL = None
_MODEL_NAME = None
_EMBEDDING_DIM = None
_NEEDS_PREFIX = None


def _get_model() -> SentenceTransformer:
    global _MODEL, _MODEL_NAME
    if _MODEL is None or _MODEL_NAME != config.EMBEDDING_MODEL:
        _MODEL = SentenceTransformer(config.EMBEDDING_MODEL)
        _MODEL_NAME = config.EMBEDDING_MODEL
    return _MODEL


def _needs_prefix() -> bool:
    global _NEEDS_PREFIX
    if _NEEDS_PREFIX is None:
        _NEEDS_PREFIX = config.EMBEDDING_MODEL.startswith("intfloat/")
    return _NEEDS_PREFIX


def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Generate embeddings for a list of texts.

    Set is_query=True for search queries (adds 'query: ' prefix for e5 models).
    Documents/chunks should use is_query=False (adds 'passage: ' prefix for e5).
    """
    model = _get_model()
    if _needs_prefix():
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + t for t in texts]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings)


def embedding_dim() -> int:
    global _EMBEDDING_DIM
    if _EMBEDDING_DIM is None:
        model = _get_model()
        _EMBEDDING_DIM = model.get_sentence_embedding_dimension()
    return _EMBEDDING_DIM
