"""Unit tests for unsupervised intent discovery and clustering routines."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hiver_agent.intents.cluster import (
    build_review_report,
    embed_messages,
    fit_clusters,
    load_thread_openers,
    sample_examples_per_cluster,
    sweep_k,
    top_terms_per_cluster,
)

SYNTHETIC_TEXTS = [
    "Cannot log in to my account, password reset link is broken.",
    "Help with login, getting incorrect password on account.",
    "Reset password email never arrived for my account.",
    "My account is locked due to too many failed login attempts.",
    "Charged twice for my monthly premium subscription invoice.",
    "Double billing charge appeared on my credit card statement.",
    "Subscription fee was deducted but premium features are inactive.",
    "How to cancel auto-renewal billing on my music subscription?",
    "Songs keep stopping and skipping during audio playback.",
    "Music player pauses automatically whenever screen turns off.",
    "Playback error 404 when trying to stream any song on playlist.",
    "Audio stuttering and lag when listening over bluetooth speaker.",
    "Offline download not working for my favorite saved playlist.",
    "Downloaded songs disappear when device is in offline mode.",
    "Cannot download tracks to SD card for offline listening.",
]


def test_load_thread_openers_filters_correctly() -> None:
    """Verify that load_thread_openers retains only inbound thread openers without parent tweets."""
    data = {
        "tweet_id": ["1", "2", "3", "4", "5", "6"],
        "inbound": [True, False, True, "True", True, True],
        "in_response_to_tweet_id": [None, None, "100", "", "nan", "200"],
        "text_clean": [
            "Need help logging in",
            "Glad to assist you!",
            "Following up on ticket",
            "Payment issue with card",
            "App crashing on start",
            "Still waiting for reply",
        ],
    }
    df = pd.DataFrame(data)
    result = load_thread_openers(df)

    # Expected openers:
    # 1: inbound=True, in_response_to_tweet_id=None -> keep
    # 2: inbound=False -> drop
    # 3: inbound=True, in_response_to_tweet_id='100' -> drop (reply)
    # 4: inbound='True', in_response_to_tweet_id='' -> keep
    # 5: inbound=True, in_response_to_tweet_id='nan' -> keep
    # 6: inbound=True, in_response_to_tweet_id='200' -> drop (reply)
    assert len(result) == 3
    assert list(result["tweet_id"]) == ["1", "4", "5"]


def test_embed_messages_shape_and_type() -> None:
    """Verify that embed_messages returns a 2D float32 numpy array with expected dimensions."""
    embeddings = embed_messages(SYNTHETIC_TEXTS)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(SYNTHETIC_TEXTS), 384)
    assert embeddings.dtype == np.float32


def test_sweep_k_returns_valid_silhouette_scores() -> None:
    """Verify that sweep_k returns valid float silhouette scores between -1.0 and 1.0 for each k."""
    embeddings = embed_messages(SYNTHETIC_TEXTS)
    k_range = [2, 3]
    scores = sweep_k(embeddings, k_range=k_range, random_state=42)

    assert isinstance(scores, dict)
    assert set(scores.keys()) == {2, 3}
    for k, score in scores.items():
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


def test_fit_clusters_returns_valid_labels() -> None:
    """Verify that fit_clusters returns an integer label array matching input sample length."""
    embeddings = embed_messages(SYNTHETIC_TEXTS)
    k = 3
    labels = fit_clusters(embeddings, k=k, random_state=42)

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (len(SYNTHETIC_TEXTS),)
    assert issubclass(labels.dtype.type, np.integer)
    assert set(np.unique(labels)).issubset({0, 1, 2})


def test_top_terms_per_cluster_independent() -> None:
    """Verify top_terms_per_cluster extracts distinguishing terms for predefined fixed labels."""
    texts = [
        "login password reset account credentials",
        "forgot password unable to login account",
        "billing charge invoice credit card payment",
        "refund payment monthly subscription charge",
    ]
    labels = np.array([0, 0, 1, 1])
    top_terms = top_terms_per_cluster(texts, labels, top_n=3)

    assert isinstance(top_terms, dict)
    assert set(top_terms.keys()) == {0, 1}
    assert len(top_terms[0]) <= 3
    assert len(top_terms[1]) <= 3

    # Cluster 0 should contain login/password related terms
    assert any(term in top_terms[0] for term in ["login", "password", "account", "reset"])
    # Cluster 1 should contain billing/payment related terms
    assert any(term in top_terms[1] for term in ["billing", "charge", "payment", "invoice", "refund"])


def test_sample_examples_per_cluster_independent() -> None:
    """Verify sample_examples_per_cluster samples requested count from assigned clusters deterministically."""
    texts = [
        "cluster0 message A",
        "cluster0 message B",
        "cluster0 message C",
        "cluster1 message D",
        "cluster1 message E",
    ]
    labels = np.array([0, 0, 0, 1, 1])
    n_samples = 2

    samples = sample_examples_per_cluster(texts, labels, n=n_samples, random_state=42)

    assert isinstance(samples, dict)
    assert set(samples.keys()) == {0, 1}
    assert len(samples[0]) == 2
    assert len(samples[1]) == 2

    for s in samples[0]:
        assert s in texts[:3]
    for s in samples[1]:
        assert s in texts[3:]


def test_build_review_report_generates_markdown() -> None:
    """Verify build_review_report creates structured Markdown with table and cluster sections."""
    cluster_sizes = {0: 25, 1: 15}
    top_terms = {
        0: ["login", "password", "reset"],
        1: ["billing", "invoice", "charge"],
    }
    examples = {
        0: ["Cannot log into account", "Forgot password"],
        1: ["Double charged for subscription", "Billing inquiry"],
    }

    report = build_review_report(cluster_sizes, top_terms, examples)

    assert isinstance(report, str)
    assert "# Intent Clusters Review Report" in report
    assert "| Cluster 0 | 25 | 62.5% |" in report
    assert "| Cluster 1 | 15 | 37.5% |" in report
    assert "### Cluster 0" in report
    assert "### Cluster 1" in report
    assert "`login`" in report
    assert "`billing`" in report
    assert "Cannot log into account" in report
    assert "Double charged for subscription" in report
