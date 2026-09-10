"""Build and embed historical customer-reply pairs for nearest-neighbor response retrieval."""

import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
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


def build_customer_reply_pairs(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """Extract customer-reply pairs where the brand replied to a customer message in the dataset."""
    if df.empty:
        return pd.DataFrame(
            columns=["customer_tweet_id", "customer_text", "reply_tweet_id", "reply_text"]
        )

    text_col = "text_clean" if "text_clean" in df.columns else "text"

    # Build unique lookup mapping for tweet_id -> text
    df_lookup = df.dropna(subset=["tweet_id"]).drop_duplicates(subset=["tweet_id"])
    id_to_text: Dict[str, str] = {}
    for tid, txt in zip(df_lookup["tweet_id"], df_lookup[text_col]):
        s_id = str(tid).strip()
        if s_id.endswith(".0"):
            s_id = s_id[:-2]
        if s_id and not pd.isna(txt):
            id_to_text[s_id] = str(txt)

    # Filter to tweets authored by the brand
    brand_mask = df["author_id"].astype(str).str.strip() == str(brand).strip()
    brand_replies = df[brand_mask].copy()

    records = []
    for _, row in brand_replies.iterrows():
        parent_raw = row.get("in_response_to_tweet_id")
        if parent_raw is None or pd.isna(parent_raw):
            continue

        parent_id = str(parent_raw).strip()
        if parent_id.endswith(".0"):
            parent_id = parent_id[:-2]

        if not parent_id or parent_id.lower() in ("nan", "none", "<na>"):
            continue

        if parent_id in id_to_text:
            reply_id = str(row.get("tweet_id", "")).strip()
            if reply_id.endswith(".0"):
                reply_id = reply_id[:-2]

            reply_text = str(row.get(text_col, ""))
            customer_text = id_to_text[parent_id]

            records.append(
                {
                    "customer_tweet_id": parent_id,
                    "customer_text": customer_text,
                    "reply_tweet_id": reply_id,
                    "reply_text": reply_text,
                }
            )

    return pd.DataFrame(
        records,
        columns=["customer_tweet_id", "customer_text", "reply_tweet_id", "reply_text"],
    )


def embed_customer_messages(
    pairs_df: pd.DataFrame,
    model: Optional[Any] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> np.ndarray:
    """Encode the customer_text column of paired dialogues using the embedding model."""
    if pairs_df.empty or "customer_text" not in pairs_df.columns:
        return np.empty((0, 384), dtype=np.float32)

    if model is None:
        model = load_embedding_model(model_name)

    texts = pairs_df["customer_text"].astype(str).tolist()
    return embed_texts(texts, model)


def main(
    config_path: Optional[str] = None,
    force_recompute: bool = False,
) -> None:
    """Build paired customer-reply dataset, embed customer inquiries, and save to disk."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    cfg_file = Path(config_path) if config_path else root_dir / "configs" / "config.yaml"

    brand = DEFAULT_BRAND
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            brand = cfg.get("data", {}).get("brand", cfg.get("brand", DEFAULT_BRAND))

    processed_dir = root_dir / "data" / "processed"
    clean_csv_path = processed_dir / f"{brand}_clean.csv"
    pairs_csv_path = processed_dir / f"{brand}_retrieval_pairs.csv"
    embeddings_npy_path = processed_dir / f"{brand}_retrieval_embeddings.npy"

    # Skip recomputation if cached artifacts already exist
    if pairs_csv_path.exists() and embeddings_npy_path.exists() and not force_recompute:
        print(f"Cached retrieval artifacts found:\n  - {pairs_csv_path}\n  - {embeddings_npy_path}\nSkipping recomputation.")
        return

    logger.info("Loading cleaned brand dataset from %s", clean_csv_path)
    df = pd.read_csv(
        clean_csv_path,
        dtype={
            "tweet_id": str,
            "author_id": str,
            "in_response_to_tweet_id": str,
            "response_tweet_id": str,
            "thread_id": str,
        },
    )

    logger.info("Extracting customer-reply pairs for brand '%s'...", brand)
    pairs_df = build_customer_reply_pairs(df, brand=brand)
    logger.info("Built %d customer-reply pairs. Saving to %s", len(pairs_df), pairs_csv_path)
    pairs_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(pairs_csv_path, index=False)

    logger.info("Loading embedding model '%s'...", DEFAULT_MODEL_NAME)
    model = load_embedding_model(DEFAULT_MODEL_NAME)

    logger.info("Embedding %d customer messages...", len(pairs_df))
    t0 = time.time()
    embeddings = embed_customer_messages(pairs_df, model=model)
    embedding_wall_time = time.time() - t0

    logger.info(
        "Embedding completed in %.2f seconds (%.1f messages/sec)",
        embedding_wall_time,
        len(pairs_df) / embedding_wall_time if embedding_wall_time > 0 else 0,
    )

    logger.info("Saving embeddings array of shape %s to %s", embeddings.shape, embeddings_npy_path)
    np.save(embeddings_npy_path, embeddings)

    print("\n" + "=" * 60)
    print("        Retrieval Embedding Generation Summary")
    print("=" * 60)
    print(f" Brand:                   {brand}")
    print(f" Pairs Extracted:         {len(pairs_df)}")
    print(f" Embedding Dimensions:    {embeddings.shape}")
    print(f" Wall-clock Time:         {embedding_wall_time:.2f} s")
    print(f" Pairs CSV:               {pairs_csv_path}")
    print(f" Embeddings NPY:          {embeddings_npy_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
