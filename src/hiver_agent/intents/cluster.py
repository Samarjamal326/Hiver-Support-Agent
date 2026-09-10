"""Unsupervised intent discovery and clustering of customer support inquiries."""

import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

# Ensure PyTorch OpenMP runtime initializes before scikit-learn on Windows
try:
    import torch  # noqa: F401
except ImportError:
    pass

# Ensure src root is in sys.path when executed directly as a script
_src_root = Path(__file__).resolve().parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import yaml

from hiver_agent.embeddings import (
    DEFAULT_MODEL_NAME,
    embed_texts,
    load_embedding_model,
)

logger = logging.getLogger(__name__)

K_RANGE = range(8, 15)
RANDOM_STATE = 42
MAX_SAMPLE_SIZE = 1000
EXAMPLES_PER_CLUSTER = 8
TOP_TERMS_PER_CLUSTER = 8


def load_thread_openers(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to inbound thread-opening messages without prior parent tweets."""
    if df.empty:
        return df.copy()

    # Match inbound messages whether stored as boolean or string
    if df["inbound"].dtype == bool:
        inbound_mask = df["inbound"]
    else:
        inbound_mask = df["inbound"].astype(str).str.strip().str.lower().isin(["true", "1"])

    # Match messages that do not respond to an earlier tweet
    if "in_response_to_tweet_id" in df.columns:
        opener_mask = df["in_response_to_tweet_id"].isna() | (
            df["in_response_to_tweet_id"].astype(str).str.strip().isin(["", "nan", "none", "<na>"])
        )
    else:
        opener_mask = pd.Series(True, index=df.index)

    return df[inbound_mask & opener_mask].copy()


def embed_messages(
    texts: List[str], model_name: str = DEFAULT_MODEL_NAME
) -> np.ndarray:
    """Encode a list of text messages into dense sentence embeddings using SentenceTransformer."""
    model = load_embedding_model(model_name)
    return embed_texts(texts, model)


def sweep_k(
    embeddings: np.ndarray,
    k_range: Union[List[int], range] = K_RANGE,
    random_state: int = RANDOM_STATE,
) -> Dict[int, float]:
    """Run KMeans clustering across candidate k values and return silhouette scores."""
    scores: Dict[int, float] = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores[k] = float(score)
    return scores


def fit_clusters(
    embeddings: np.ndarray, k: int, random_state: int = RANDOM_STATE
) -> np.ndarray:
    """Fit final KMeans clustering for chosen k and return 1D array of cluster labels."""
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return np.asarray(labels, dtype=int)


def top_terms_per_cluster(
    texts: List[str],
    labels: np.ndarray,
    top_n: int = TOP_TERMS_PER_CLUSTER,
) -> Dict[int, List[str]]:
    """Extract top distinguishing terms per cluster using TF-IDF across aggregated cluster texts."""
    unique_labels = sorted(int(c) for c in np.unique(labels))
    cluster_docs = [
        " ".join(texts[i] for i, lbl in enumerate(labels) if lbl == c)
        for c in unique_labels
    ]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
        )
        tfidf_matrix = vectorizer.fit_transform(cluster_docs)
    except ValueError:
        try:
            vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tfidf_matrix = vectorizer.fit_transform(cluster_docs)
        except ValueError:
            return {c: [] for c in unique_labels}

    feature_names = np.array(vectorizer.get_feature_names_out())
    top_terms: Dict[int, List[str]] = {}

    for idx, c in enumerate(unique_labels):
        row = tfidf_matrix[idx].toarray().ravel()
        positive_indices = np.where(row > 0)[0]
        if len(positive_indices) == 0:
            top_terms[c] = []
            continue
        sorted_indices = positive_indices[np.argsort(row[positive_indices])[::-1]]
        top_terms[c] = feature_names[sorted_indices[:top_n]].tolist()

    return top_terms


def sample_examples_per_cluster(
    texts: List[str],
    labels: np.ndarray,
    n: int = EXAMPLES_PER_CLUSTER,
    random_state: int = RANDOM_STATE,
) -> Dict[int, List[str]]:
    """Sample representative message examples from each cluster."""
    rng = np.random.RandomState(random_state)
    unique_labels = sorted(int(c) for c in np.unique(labels))
    examples: Dict[int, List[str]] = {}

    for c in unique_labels:
        indices = [i for i, lbl in enumerate(labels) if lbl == c]
        if not indices:
            examples[c] = []
            continue
        if len(indices) <= n:
            chosen = indices
        else:
            chosen = rng.choice(indices, size=n, replace=False).tolist()
        examples[c] = [texts[i] for i in chosen]

    return examples


def build_review_report(
    cluster_sizes: Dict[int, int],
    top_terms: Dict[int, List[str]],
    examples: Dict[int, List[str]],
) -> str:
    """Build a Markdown review report summarizing cluster sizes, distinguishing terms, and sample messages."""
    total_messages = sum(cluster_sizes.values())
    lines: List[str] = [
        "# Intent Clusters Review Report",
        "",
        "This report summarizes unsupervised KMeans clustering on customer support thread openers.",
        "Review the distinguishing terms and sample inquiries below to determine candidate intent taxonomy labels.",
        "",
        "## Summary Overview",
        "",
        "| Cluster ID | Size | Percentage | Top Distinguishing Terms |",
        "|---|---|---|---|",
    ]

    for c in sorted(cluster_sizes.keys()):
        size = cluster_sizes[c]
        pct = (size / total_messages * 100.0) if total_messages > 0 else 0.0
        terms_str = ", ".join(f"`{t}`" for t in top_terms.get(c, [])) if top_terms.get(c) else "_none_"
        lines.append(f"| Cluster {c} | {size} | {pct:.1f}% | {terms_str} |")

    lines.extend(["", "## Detailed Cluster Profiles", ""])

    for c in sorted(cluster_sizes.keys()):
        size = cluster_sizes[c]
        pct = (size / total_messages * 100.0) if total_messages > 0 else 0.0
        terms_list = top_terms.get(c, [])
        terms_display = ", ".join(f"`{t}`" for t in terms_list) if terms_list else "None identified"

        lines.append(f"### Cluster {c}")
        lines.append(f"- **Size**: {size} messages ({pct:.1f}%)")
        lines.append(f"- **Distinguishing Keywords**: {terms_display}")
        lines.append("- **Sample Inquiries**:")

        cluster_examples = examples.get(c, [])
        if cluster_examples:
            for idx, ex in enumerate(cluster_examples, 1):
                clean_ex = ex.replace("\n", " ").strip()
                lines.append(f"  {idx}. \"{clean_ex}\"")
        else:
            lines.append("  - _No samples available_")

        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main(
    csv_path: Optional[str] = None,
    output_path: Optional[str] = None,
    k_override: Optional[int] = None,
) -> None:
    """Run unsupervised clustering pipeline on thread openers and write the review report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 1. Resolve paths
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    config_path = root_dir / "configs" / "config.yaml"

    if csv_path is None:
        brand = "SpotifyCares"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                brand = cfg.get("data", {}).get("brand", cfg.get("brand", "SpotifyCares"))
        csv_path = str(root_dir / "data" / "processed" / f"{brand}_clean.csv")

    if output_path is None:
        output_path = str(root_dir / "reports" / "intent_clusters_review.md")

    logger.info("Reading processed data from %s", csv_path)
    df = pd.read_csv(
        csv_path,
        dtype={
            "tweet_id": str,
            "in_response_to_tweet_id": str,
            "response_tweet_id": str,
            "thread_id": str,
        },
    )

    # 2. Extract thread openers
    openers_df = load_thread_openers(df)
    total_openers = len(openers_df)
    logger.info("Identified %d thread openers out of %d total tweets", total_openers, len(df))

    # 3. Sample up to MAX_SAMPLE_SIZE
    if total_openers > MAX_SAMPLE_SIZE:
        sampled_df = openers_df.sample(n=MAX_SAMPLE_SIZE, random_state=RANDOM_STATE)
        logger.info(
            "Sampled %d thread openers (MAX_SAMPLE_SIZE=%d, RANDOM_STATE=%d)",
            len(sampled_df),
            MAX_SAMPLE_SIZE,
            RANDOM_STATE,
        )
    else:
        sampled_df = openers_df
        logger.info("Using all %d thread openers (<= MAX_SAMPLE_SIZE=%d)", total_openers, MAX_SAMPLE_SIZE)

    text_col = "text_clean" if "text_clean" in sampled_df.columns else "text"
    texts = sampled_df[text_col].dropna().astype(str).tolist()

    # 4. Generate embeddings
    logger.info("Generating embeddings for %d messages using %s...", len(texts), DEFAULT_MODEL_NAME)
    embeddings = embed_messages(texts, model_name=DEFAULT_MODEL_NAME)

    # 5. Sweep k to compute silhouette scores
    k_list = list(K_RANGE)
    logger.info("Sweeping k over range %s...", k_list)
    scores = sweep_k(embeddings, k_range=k_list, random_state=RANDOM_STATE)

    print("\n" + "=" * 36)
    print("      k -> Silhouette Score Table")
    print("=" * 36)
    print(f" {'k':<6} | {'Silhouette Score':<20}")
    print("-" * 36)
    for k in k_list:
        print(f" {k:<6} | {scores[k]:<20.4f}")
    print("=" * 36 + "\n")

    best_k = max(scores, key=scores.get)
    chosen_k = k_override if k_override is not None else best_k
    logger.info("Best k by silhouette score: %d (score: %.4f). Fitting with k=%d.", best_k, scores[best_k], chosen_k)

    # 6. Fit clusters with chosen k
    labels = fit_clusters(embeddings, k=chosen_k, random_state=RANDOM_STATE)

    # 7. Extract cluster summaries, terms, and samples
    cluster_sizes = {int(c): int(np.sum(labels == c)) for c in sorted(np.unique(labels))}
    top_terms = top_terms_per_cluster(texts, labels, top_n=TOP_TERMS_PER_CLUSTER)
    examples = sample_examples_per_cluster(texts, labels, n=EXAMPLES_PER_CLUSTER, random_state=RANDOM_STATE)

    # 8. Build and save review report
    report_content = build_review_report(cluster_sizes, top_terms, examples)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report_content, encoding="utf-8")
    logger.info("Wrote intent clusters review report to %s", output_path)


if __name__ == "__main__":
    main()
