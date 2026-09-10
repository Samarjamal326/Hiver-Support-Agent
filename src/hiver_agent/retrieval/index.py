"""NearestNeighbors index construction and top-k dialogue retrieval for customer inquiries."""

import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import yaml

# Ensure src root is in sys.path when executed directly as a script
_src_root = Path(__file__).resolve().parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from hiver_agent.embeddings import (
    DEFAULT_MODEL_NAME,
    embed_texts,
    load_embedding_model,
)

logger = logging.getLogger(__name__)

DEFAULT_BRAND = "SpotifyCares"
DEFAULT_K = 3


def build_index(embeddings: np.ndarray) -> NearestNeighbors:
    """Build and fit an sklearn NearestNeighbors index using cosine metric."""
    if len(embeddings) == 0:
        raise ValueError("Cannot build index on empty embeddings array.")

    n_neighbors = min(50, len(embeddings))
    index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    index.fit(embeddings)
    return index


def retrieve_top_k(
    query_embedding: np.ndarray,
    index: Any,
    pairs_df: pd.DataFrame,
    k: int = DEFAULT_K,
) -> List[Dict[str, Any]]:
    """Retrieve the k nearest customer-reply matches for a query embedding sorted by similarity descending."""
    if pairs_df.empty or len(pairs_df) == 0:
        return []

    # Ensure query embedding is 2-dimensional (1, D)
    if query_embedding.ndim == 1:
        query_vec = query_embedding.reshape(1, -1)
    else:
        query_vec = query_embedding

    effective_k = min(k, len(pairs_df))
    distances, indices = index.kneighbors(query_vec, n_neighbors=effective_k)

    results: List[Dict[str, Any]] = []
    for dist, idx in zip(distances[0], indices[0]):
        # Cosine distance d = 1 - cos(theta) => Cosine similarity = 1.0 - d
        similarity = float(1.0 - dist)
        row = pairs_df.iloc[idx]
        results.append(
            {
                "customer_text": str(row["customer_text"]),
                "reply_text": str(row["reply_text"]),
                "reply_tweet_id": str(row["reply_tweet_id"]),
                "similarity_score": round(similarity, 4),
            }
        )

    # Sort descending by similarity score
    results.sort(key=lambda item: item["similarity_score"], reverse=True)
    return results[:k]


def main(
    config_path: Optional[str] = None,
    k: int = DEFAULT_K,
) -> None:
    """Load index, run sample queries from smoke sample report, and display top-k retrieved matches."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    cfg_file = Path(config_path) if config_path else root_dir / "configs" / "config.yaml"

    brand = DEFAULT_BRAND
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            brand = cfg.get("data", {}).get("brand", cfg.get("brand", DEFAULT_BRAND))

    processed_dir = root_dir / "data" / "processed"
    pairs_csv = processed_dir / f"{brand}_retrieval_pairs.csv"
    embeddings_npy = processed_dir / f"{brand}_retrieval_embeddings.npy"

    if not pairs_csv.exists() or not embeddings_npy.exists():
        logger.info("Artifacts not found on disk. Invoking embed generation first...")
        from hiver_agent.retrieval.embed import main as embed_main

        embed_main(config_path=str(cfg_file))

    logger.info("Loading retrieval pairs from %s", pairs_csv)
    pairs_df = pd.read_csv(pairs_csv, dtype=str)

    logger.info("Loading embeddings from %s", embeddings_npy)
    embeddings = np.load(embeddings_npy)

    logger.info("Building NearestNeighbors index for %d vectors...", len(embeddings))
    index = build_index(embeddings)

    logger.info("Loading embedding model '%s'...", DEFAULT_MODEL_NAME)
    model = load_embedding_model(DEFAULT_MODEL_NAME)

    # Sample 3 test queries from reports/classifier_smoke_sample.csv (rows 7, 9, 20)
    smoke_sample_path = root_dir / "reports" / "classifier_smoke_sample.csv"
    sample_queries: List[str] = []

    if smoke_sample_path.exists():
        smoke_df = pd.read_csv(smoke_sample_path, dtype=str)
        text_col = "text_clean" if "text_clean" in smoke_df.columns else "text"
        target_indices = [6, 8, 19]  # 0-indexed rows 7, 9, 20
        for idx in target_indices:
            if idx < len(smoke_df):
                sample_queries.append(str(smoke_df.iloc[idx][text_col]))

    if not sample_queries:
        sample_queries = [
            "What's going on with the #WindowsPhone app? It's been like a week since it stopped working.",
            "why can’t we pay the premium student discount with paypal???",
            "ummm why is Take Care suddenly non existent...",
        ]

    print("\n" + "=" * 80)
    print("                    Retrieval Top-K Quality Review")
    print("=" * 80)

    for q_idx, query_text in enumerate(sample_queries, 1):
        safe_query = query_text.encode("ascii", errors="replace").decode("ascii")
        print(f"\n[Query {q_idx}]: \"{safe_query}\"")
        print("-" * 80)

        query_emb = embed_texts([query_text], model)
        matches = retrieve_top_k(query_emb, index, pairs_df, k=k)

        for m_idx, match in enumerate(matches, 1):
            cust_preview = match["customer_text"].encode("ascii", errors="replace").decode("ascii")
            reply_preview = match["reply_text"].encode("ascii", errors="replace").decode("ascii")
            score = match["similarity_score"]
            tid = match["reply_tweet_id"]

            print(f"  Match {m_idx} (Score: {score:.4f} | Reply ID: {tid}):")
            print(f"    Similar Past Customer: \"{cust_preview}\"")
            print(f"    Brand Historical Reply: \"{reply_preview}\"")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
