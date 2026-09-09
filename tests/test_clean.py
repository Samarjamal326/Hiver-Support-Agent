"""Unit tests for customer support data cleaning routines."""

import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hiver_agent.data.clean import (
    clean_dataframe,
    clean_text,
    is_english_heuristic,
    mask_exact_duplicates,
    mask_min_length,
    mask_english,
    strip_leading_mentions,
    strip_urls,
)


def test_placeholder():
    """Placeholder test ensuring test runner health."""
    assert True


def test_strip_urls_removes_url_and_keeps_surrounding_text():
    """Verify that URL stripping removes URLs completely while preserving surrounding text."""
    raw = "Please check https://support.spotify.com/help or www.spotify.com for assistance."
    stripped = strip_urls(raw)
    assert "https://support.spotify.com/help" not in stripped
    assert "www.spotify.com" not in stripped
    assert "Please check" in stripped
    assert "for assistance." in stripped

    # clean_text additionally normalizes whitespace
    expected_clean = "Please check or for assistance."
    assert clean_text(raw) == expected_clean

    # Test via clean_dataframe
    df = pd.DataFrame([{"text": "Visit https://t.co/xyz123 today"}])
    result = clean_dataframe(df, verbose=False)
    assert result["text_clean"].iloc[0] == "Visit today"


def test_strip_leading_mentions_keeps_mid_sentence_mentions():
    """Verify that leading threading @mentions are stripped while mid-sentence mentions are kept."""
    raw = "@123456 @SpotifyCares I really love listening to @taylorswift13 on Spotify!"
    expected = "I really love listening to @taylorswift13 on Spotify!"
    assert strip_leading_mentions(raw) == expected
    assert clean_text(raw) == expected

    # Test that a text without leading mentions is unaltered
    raw_mid_only = "Hey @SpotifyCares why is the music paused?"
    assert strip_leading_mentions(raw_mid_only) == raw_mid_only


def test_duplicate_removal_drops_exact_repeats_keeps_first():
    """Verify that exact duplicate text_raw rows are dropped and first occurrence is kept."""
    df = pd.DataFrame(
        [
            {"text_raw": "I cannot login to my account", "author_id": "cust_1"},
            {"text_raw": "I cannot login to my account", "author_id": "cust_2"},
            {"text_raw": "My playlist disappeared", "author_id": "cust_3"},
            {"text_raw": "I cannot login to my account", "author_id": "cust_4"},
        ]
    )
    dup_mask = mask_exact_duplicates(df)
    assert dup_mask.tolist() == [True, False, True, False]

    cleaned = clean_dataframe(df, verbose=False)
    assert len(cleaned) == 2
    assert cleaned["author_id"].tolist() == ["cust_1", "cust_3"]
    assert cleaned["text_raw"].tolist() == [
        "I cannot login to my account",
        "My playlist disappeared",
    ]


def test_empty_and_short_after_cleaning_rows_are_dropped():
    """Verify that rows empty or under 3 characters after cleaning are dropped."""
    df = pd.DataFrame(
        [
            {"text": "https://t.co/onlyurl"},  # empty after URL strip
            {"text": "@105843 "},  # empty after leading mention strip
            {"text": "ok"},  # 2 characters (< 3)
            {"text": "   "},  # whitespace only
            {"text": "App keeps crashing on startup"},  # valid >= 3 chars
        ]
    )
    cleaned = clean_dataframe(df, min_len=3, verbose=False)
    assert len(cleaned) == 1
    assert cleaned["text_clean"].iloc[0] == "App keeps crashing on startup"


def test_non_english_heuristic_flags_non_english_and_passes_english():
    """Verify that ASCII ratio heuristic flags clearly non-English text and passes English text."""
    english_text = "Can someone please help me fix my playlist? It stops playing after one song."
    russian_text = "Привет, помогите пожалуйста восстановить мой аккаунт"
    japanese_text = "音楽が再生されません。助けてください。"
    arabic_text = "مرحبا، أحتاج إلى مساعدة في حسابي"

    assert is_english_heuristic(english_text, threshold=0.6) is True
    assert is_english_heuristic(russian_text, threshold=0.6) is False
    assert is_english_heuristic(japanese_text, threshold=0.6) is False
    assert is_english_heuristic(arabic_text, threshold=0.6) is False

    df = pd.DataFrame(
        [
            {"text": english_text, "lang": "en"},
            {"text": russian_text, "lang": "ru"},
            {"text": japanese_text, "lang": "ja"},
            {"text": arabic_text, "lang": "ar"},
        ]
    )
    cleaned = clean_dataframe(df, english_threshold=0.6, verbose=False)
    assert len(cleaned) == 1
    assert cleaned["lang"].iloc[0] == "en"
    assert cleaned["text_clean"].iloc[0] == english_text
