"""Shared text embedding utilities and sentence-transformer model loading."""

import logging
from typing import Any, List
import numpy as np

# Ensure PyTorch OpenMP runtime initializes before scikit-learn on Windows
try:
    import torch  # noqa: F401
except ImportError:
    pass

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Load and return a SentenceTransformer embedding model."""
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def embed_texts(texts: List[str], model: Any) -> np.ndarray:
    """Encode a collection of texts into dense embeddings using the provided model."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(embeddings, dtype=np.float32)
