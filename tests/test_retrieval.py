"""Unit tests for customer-reply pair generation, index construction, and top-k retrieval."""

import numpy as np
import pandas as pd
import pytest

from hiver_agent.retrieval.embed import build_customer_reply_pairs
from hiver_agent.retrieval.index import build_index, retrieve_top_k


def test_build_customer_reply_pairs_matches_and_skips_unfindable_parents() -> None:
    """Verify build_customer_reply_pairs pairs valid parent-reply tweets and skips orphan replies."""
    data = {
        "tweet_id": ["100", "101", "102", "103", "104"],
        "author_id": ["user1", "SpotifyCares", "user2", "SpotifyCares", "OtherBrand"],
        "in_response_to_tweet_id": [None, "100", None, "999", "102"],
        "text_clean": [
            "Cannot log in to my account",
            "Hey! Send us a DM with your account email.",
            "Songs are buffering constantly",
            "Try restarting your network router.",
            "Unrelated brand response",
        ],
    }
    df = pd.DataFrame(data)

    pairs = build_customer_reply_pairs(df, brand="SpotifyCares")

    # Only row 101 is a SpotifyCares reply with an existing parent (100)
    # Row 103 has parent 999 which does not exist in df -> skipped
    # Row 104 is authored by OtherBrand -> skipped
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["customer_tweet_id"] == "100"
    assert row["customer_text"] == "Cannot log in to my account"
    assert row["reply_tweet_id"] == "101"
    assert row["reply_text"] == "Hey! Send us a DM with your account email."


def test_build_index_and_retrieve_top_k_mathematical_order() -> None:
    """Verify build_index and retrieve_top_k return mathematically correct nearest neighbors in exact descending similarity."""
    # Hand-constructed 3D vectors with known cosine similarities against query [1.0, 0.0, 0.0]
    # v0: dot product with q = 1.0 (exact match)
    # v1: dot product with q = 0.8
    # v2: dot product with q = 0.6
    # v3: dot product with q = 0.0 (orthogonal)
    # v4: dot product with q = -1.0 (opposite)
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.6, 0.8, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    pairs_df = pd.DataFrame(
        {
            "customer_tweet_id": ["c0", "c1", "c2", "c3", "c4"],
            "customer_text": ["text 0", "text 1", "text 2", "text 3", "text 4"],
            "reply_tweet_id": ["r0", "r1", "r2", "r3", "r4"],
            "reply_text": ["reply 0", "reply 1", "reply 2", "reply 3", "reply 4"],
        }
    )

    index = build_index(embeddings)
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    results = retrieve_top_k(query_vec, index, pairs_df, k=3)

    assert len(results) == 3
    # Expected ranking: v0 (1.0) -> v1 (0.8) -> v2 (0.6)
    assert results[0]["reply_tweet_id"] == "r0"
    assert pytest.approx(results[0]["similarity_score"], abs=1e-3) == 1.0

    assert results[1]["reply_tweet_id"] == "r1"
    assert pytest.approx(results[1]["similarity_score"], abs=1e-3) == 0.8

    assert results[2]["reply_tweet_id"] == "r2"
    assert pytest.approx(results[2]["similarity_score"], abs=1e-3) == 0.6


def test_retrieve_top_k_respects_k_parameter() -> None:
    """Verify retrieve_top_k returns exactly the requested number of nearest neighbors."""
    embeddings = np.eye(5, dtype=np.float32)
    pairs_df = pd.DataFrame(
        {
            "customer_tweet_id": [f"c{i}" for i in range(5)],
            "customer_text": [f"text {i}" for i in range(5)],
            "reply_tweet_id": [f"r{i}" for i in range(5)],
            "reply_text": [f"reply {i}" for i in range(5)],
        }
    )

    index = build_index(embeddings)
    query = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Request k = 2
    res_k2 = retrieve_top_k(query, index, pairs_df, k=2)
    assert len(res_k2) == 2

    # Request k = 4
    res_k4 = retrieve_top_k(query, index, pairs_df, k=4)
    assert len(res_k4) == 4

    # Request k larger than dataset size (k=10 on 5 rows) -> should clamp to 5
    res_clamp = retrieve_top_k(query, index, pairs_df, k=10)
    assert len(res_clamp) == 5


def test_retrieve_top_k_excludes_self_match() -> None:
    """Verify retrieve_top_k removes the query's own tweet_id when in index and still returns full k neighbors."""
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],  # c0: exact match with query (sim = 1.0)
            [0.8, 0.6, 0.0],  # c1: sim = 0.8
            [0.6, 0.8, 0.0],  # c2: sim = 0.6
            [0.0, 1.0, 0.0],  # c3: sim = 0.0
            [-1.0, 0.0, 0.0],  # c4: sim = -1.0
        ],
        dtype=np.float32,
    )

    pairs_df = pd.DataFrame(
        {
            "customer_tweet_id": ["c0", "c1", "c2", "c3", "c4"],
            "customer_text": ["text 0", "text 1", "text 2", "text 3", "text 4"],
            "reply_tweet_id": ["r0", "r1", "r2", "r3", "r4"],
            "reply_text": ["reply 0", "reply 1", "reply 2", "reply 3", "reply 4"],
        }
    )

    index = build_index(embeddings)
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Without exclusion, c0 is top match at similarity 1.0
    res_unfiltered = retrieve_top_k(query_vec, index, pairs_df, k=3)
    assert res_unfiltered[0]["reply_tweet_id"] == "r0"
    assert pytest.approx(res_unfiltered[0]["similarity_score"], abs=1e-3) == 1.0

    # With exclude_tweet_id="c0", c0 is filtered out, but full k=3 results are returned (c1, c2, c3)
    res_filtered = retrieve_top_k(query_vec, index, pairs_df, k=3, exclude_tweet_id="c0")
    assert len(res_filtered) == 3
    assert all(r["reply_tweet_id"] != "r0" for r in res_filtered)
    assert res_filtered[0]["reply_tweet_id"] == "r1"
    assert pytest.approx(res_filtered[0]["similarity_score"], abs=1e-3) == 0.8
    assert res_filtered[1]["reply_tweet_id"] == "r2"
    assert pytest.approx(res_filtered[1]["similarity_score"], abs=1e-3) == 0.6
    assert res_filtered[2]["reply_tweet_id"] == "r3"
    assert pytest.approx(res_filtered[2]["similarity_score"], abs=1e-3) == 0.0


def test_retrieve_top_k_excludes_duplicate_customer_tweet_id() -> None:
    """Verify exclude_tweet_id removes all rows sharing the excluded customer_tweet_id and still returns full k."""
    # 5 candidate vectors: c0 appears in two rows (branch replies) with identical embeddings
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],  # c0, branch reply A (similarity 1.0)
            [1.0, 0.0, 0.0],  # c0, branch reply B (similarity 1.0)
            [0.8, 0.6, 0.0],  # c1, reply 1 (similarity 0.8)
            [0.6, 0.8, 0.0],  # c2, reply 2 (similarity 0.6)
            [0.0, 1.0, 0.0],  # c3, reply 3 (similarity 0.0)
        ],
        dtype=np.float32,
    )

    pairs_df = pd.DataFrame(
        {
            "customer_tweet_id": ["c0", "c0", "c1", "c2", "c3"],
            "customer_text": [
                "Customer message 0",
                "Customer message 0",
                "Customer message 1",
                "Customer message 2",
                "Customer message 3",
            ],
            "reply_tweet_id": ["r0_a", "r0_b", "r1", "r2", "r3"],
            "reply_text": [
                "First branch reply to c0",
                "Second branch reply to c0",
                "Reply to c1",
                "Reply to c2",
                "Reply to c3",
            ],
        }
    )

    index = build_index(embeddings)
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Exclude "c0" and request full k=3
    results = retrieve_top_k(query_vec, index, pairs_df, k=3, exclude_tweet_id="c0")

    # Verify both rows corresponding to c0 were removed
    assert len(results) == 3
    returned_reply_ids = [r["reply_tweet_id"] for r in results]
    assert "r0_a" not in returned_reply_ids
    assert "r0_b" not in returned_reply_ids

    # Verify the remaining 3 distinct vectors are returned in descending similarity order
    assert returned_reply_ids == ["r1", "r2", "r3"]
    assert pytest.approx(results[0]["similarity_score"], abs=1e-3) == 0.8
    assert pytest.approx(results[1]["similarity_score"], abs=1e-3) == 0.6
    assert pytest.approx(results[2]["similarity_score"], abs=1e-3) == 0.0


