"""Smoke tests for the customer support agent pipeline."""

import sys
from pathlib import Path
import pandas as pd

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hiver_agent.data.clean import clean_dataframe
from hiver_agent.data.sample_brand import extract_brand_slice


def test_placeholder():
    """Placeholder smoke test."""
    assert True


def test_data_pipeline_end_to_end_smoke():
    """End-to-end smoke test on synthetic data: extract brand slice and clean dataset."""
    synthetic_rows = [
        # Thread 1: Spotify conversation (Customer Opener -> Brand Reply -> Customer Follow-up -> Brand Resolution)
        {
            "tweet_id": "1001",
            "author_id": "customer_alpha",
            "inbound": True,
            "created_at": "Tue Oct 31 22:00:00 +0000 2017",
            "text": "Hey @SpotifyCares my desktop app crashes whenever I start a podcast https://t.co/abc",
            "response_tweet_id": "1002",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "1002",
            "author_id": "SpotifyCares",
            "inbound": False,
            "created_at": "Tue Oct 31 22:05:00 +0000 2017",
            "text": "@customer_alpha We are here to help! Could you share your OS version and app build number?",
            "response_tweet_id": "1003",
            "in_response_to_tweet_id": "1001",
        },
        {
            "tweet_id": "1003",
            "author_id": "customer_alpha",
            "inbound": True,
            "created_at": "Tue Oct 31 22:10:00 +0000 2017",
            "text": "@SpotifyCares I am on Windows 11 with Spotify build 1.2.14",
            "response_tweet_id": "1004",
            "in_response_to_tweet_id": "1002",
        },
        {
            "tweet_id": "1004",
            "author_id": "SpotifyCares",
            "inbound": False,
            "created_at": "Tue Oct 31 22:15:00 +0000 2017",
            "text": "@customer_alpha Thanks! Please run a clean reinstall following the steps in our FAQ.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "1003",
        },
        # Thread 2: Unrelated brand thread (AppleSupport)
        {
            "tweet_id": "2001",
            "author_id": "customer_beta",
            "inbound": True,
            "created_at": "Tue Oct 31 22:00:00 +0000 2017",
            "text": "My iPhone battery drains overnight @AppleSupport",
            "response_tweet_id": "2002",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "2002",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": "Tue Oct 31 22:05:00 +0000 2017",
            "text": "@customer_beta Please send us a DM with your iOS version.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "2001",
        },
        # Row 7: Short reply in Thread 1 to verify cleaning interaction
        {
            "tweet_id": "1005",
            "author_id": "customer_alpha",
            "inbound": True,
            "created_at": "Tue Oct 31 22:20:00 +0000 2017",
            "text": "ok",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "1004",
        },
    ]
    df_raw = pd.DataFrame(synthetic_rows)

    # 1. Run brand slice extraction
    slice_df = extract_brand_slice(df_raw, brand="SpotifyCares", max_depth=20)
    assert not slice_df.empty
    assert "thread_id" in slice_df.columns
    # Ensure all extracted Spotify rows belong to thread 1001
    assert set(slice_df["thread_id"].unique()) == {"1001"}
    # Ensure AppleSupport rows were excluded
    assert "2001" not in slice_df["tweet_id"].values
    assert "2002" not in slice_df["tweet_id"].values

    # 2. Run clean dataframe
    cleaned_df = clean_dataframe(slice_df, min_len=3, verbose=False)
    assert not cleaned_df.empty

    # Assert expected columns
    for expected_col in [
        "tweet_id",
        "author_id",
        "inbound",
        "text_raw",
        "text_clean",
        "thread_id",
    ]:
        assert expected_col in cleaned_df.columns

    # Assert cleaning behavior on end-to-end output
    # - Row 1001 had URL stripped and leading mention retained / cleaned
    row_1001 = cleaned_df[cleaned_df["tweet_id"] == "1001"].iloc[0]
    assert "https://t.co/abc" not in row_1001["text_clean"]
    assert "crashes whenever I start a podcast" in row_1001["text_clean"]

    # - Row 1002 had leading mention @customer_alpha stripped
    row_1002 = cleaned_df[cleaned_df["tweet_id"] == "1002"].iloc[0]
    assert not row_1002["text_clean"].startswith("@customer_alpha")
    assert row_1002["text_clean"].startswith("We are here to help!")

    # - Row 1005 ("ok") was dropped due to min length < 3
    assert "1005" not in cleaned_df["tweet_id"].values
