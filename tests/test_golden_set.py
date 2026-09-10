"""Unit tests for golden dataset candidate sampling, exclusion logic, stratification, and thread context."""

import pandas as pd
import pytest

from hiver_agent.eval.golden_set import (
    build_thread_context,
    filter_candidate_pool,
    generate_golden_set_candidates,
    rough_intent_bucket,
    split_openers_and_followups,
    stratified_sample_openers,
)


def test_exclusion_logic_filters_smoke_and_few_shot() -> None:
    """Verify filter_candidate_pool drops smoke-test tweet_ids, few-shot exact texts, and non-inbound rows."""
    few_shot_texts = {
        "reset the password & tells me I can't do that",
        "If I'm paying for premium, why do I still see advertisements?",
    }
    smoke_tweet_ids = {"99001", "99002"}

    data = {
        "tweet_id": ["1", "99001", "2", "3", "4"],
        "author_id": ["user1", "user2", "user3", "SpotifyCares", "user5"],
        "inbound": ["True", "True", "True", "False", "True"],
        "text_clean": [
            "Normal customer question about playlists",
            "Smoke sample question about login",  # ID 99001 in smoke_tweet_ids
            "reset the password & tells me I can't do that",  # matches few_shot_texts
            "Brand response tweet",  # inbound == False
            "Another legitimate customer question",
        ],
    }
    df = pd.DataFrame(data)

    filtered = filter_candidate_pool(df, few_shot_texts=few_shot_texts, smoke_tweet_ids=smoke_tweet_ids)

    # Only rows 1 and 4 (tweet_ids "1" and "4") should be kept
    assert len(filtered) == 2
    assert set(filtered["tweet_id"].tolist()) == {"1", "4"}


def test_opener_followup_split() -> None:
    """Verify split_openers_and_followups partitions on null/empty parent tweet IDs."""
    data = {
        "tweet_id": ["10", "11", "12", "13", "14"],
        "in_response_to_tweet_id": [None, "", "nan", "10", "11"],
        "text_clean": ["Opener 1", "Opener 2", "Opener 3", "Follow-up 1", "Follow-up 2"],
    }
    df = pd.DataFrame(data)

    openers, follow_ups = split_openers_and_followups(df)

    assert len(openers) == 3
    assert set(openers["tweet_id"].tolist()) == {"10", "11", "12"}

    assert len(follow_ups) == 2
    assert set(follow_ups["tweet_id"].tolist()) == {"13", "14"}


def test_stratified_sampling_minimum_floor() -> None:
    """Verify stratified_sample_openers respects minimum floors and fills remainder to target count."""
    # Construct a synthetic pool with known bucket assignments
    records = []
    # 15 items for Family Plan Management
    for i in range(15):
        records.append({"tweet_id": f"fam_{i}", "text_clean": f"how to add member to family plan {i}"})
    # 7 items for Advertisement Complaints (fewer than minimum of 10)
    for i in range(7):
        records.append({"tweet_id": f"ad_{i}", "text_clean": f"why do I see ads on premium {i}"})
    # 25 items for Billing & Subscription
    for i in range(25):
        records.append({"tweet_id": f"bill_{i}", "text_clean": f"charged twice on my subscription {i}"})
    # 20 items that are Unclear
    for i in range(20):
        records.append({"tweet_id": f"unc_{i}", "text_clean": f"just wondering about something {i}"})

    openers_df = pd.DataFrame(records)

    target_total = 35
    min_floor = 10

    sampled, counts = stratified_sample_openers(
        openers_df,
        target_total=target_total,
        min_per_bucket=min_floor,
        random_state=123,
    )

    # 1. Total count must exactly equal target_total
    assert len(sampled) == target_total

    # 2. Buckets with >= min_floor items must have at least min_floor in the sample
    assert counts.get("Family Plan Management", 0) >= min_floor
    assert counts.get("Billing & Subscription", 0) >= min_floor

    # 3. Buckets with fewer than min_floor must contribute all available items
    assert counts.get("Advertisement Complaints", 0) == 7


def test_build_thread_context_chronological_order() -> None:
    """Verify build_thread_context produces correctly ordered turn-by-turn text for a multi-turn thread."""
    thread_data = {
        "tweet_id": ["t1", "t2", "t3"],
        "thread_id": ["t1", "t1", "t1"],
        "author_id": ["user_100", "SpotifyCares", "user_100"],
        "inbound": ["True", "False", "True"],
        "created_at": [
            "2023-01-01 10:00:00",
            "2023-01-01 10:05:00",
            "2023-01-01 10:10:00",
        ],
        "in_response_to_tweet_id": [None, "t1", "t2"],
        "text_clean": [
            "My playlist has completely vanished",
            "Have you tried logging out and back in?",
            "Yes, did that and it is still gone!",
        ],
    }
    full_df = pd.DataFrame(thread_data)

    # Opener (t1) should have empty thread context
    opener_row = full_df.iloc[0]
    assert build_thread_context(opener_row, full_df) == ""

    # Follow-up (t3) should pull t1 and t2 in chronological order
    follow_up_row = full_df.iloc[2]
    context = build_thread_context(follow_up_row, full_df)

    expected_context = (
        "Customer: My playlist has completely vanished\n"
        "Agent: Have you tried logging out and back in?"
    )
    assert context == expected_context


def test_rough_intent_bucket_mappings() -> None:
    """Verify rough_intent_bucket assigns expected categories for representative keywords."""
    assert rough_intent_bucket("can't login to my account, password reset fails") == "Account Access/Login"
    assert rough_intent_bucket("charged 4 times for my student discount subscription") == "Billing & Subscription"
    assert rough_intent_bucket("why do I see audio ads when I pay for premium") == "Advertisement Complaints"
    assert rough_intent_bucket("songs keep pausing and buffering on offline mode") == "Playback & App Bugs"
    assert rough_intent_bucket("desktop app crashes on windows 10") == "Platform-Specific Technical"
    assert rough_intent_bucket("please add the new album by this artist to spotify") == "Content & Feature Requests"
    assert rough_intent_bucket("how do I invite a member to my family plan") == "Family Plan Management"
    assert rough_intent_bucket("app says not available in my country while traveling") == "Regional/Country Restriction"
    assert rough_intent_bucket("worst customer service ever, absolutely terrible") == "General Complaint/Frustration"
    assert rough_intent_bucket("thanks so much for the quick help, love you guys") == "Praise/Off-topic"
    assert rough_intent_bucket("random unclassifiable text without keywords") == "Unclear"


def test_generate_golden_set_candidates_pool_counts_and_historical_replies(tmp_path) -> None:
    """Verify generate_golden_set_candidates writes raw pool counts and attaches historical_reply_text to debug CSV."""
    clean_csv = tmp_path / "clean.csv"
    smoke_csv = tmp_path / "smoke.csv"
    out_candidates = tmp_path / "candidates.csv"
    out_debug = tmp_path / "debug.csv"
    out_pool_counts = tmp_path / "pool_counts.csv"

    # Synthetic interaction data:
    # 2 customer openers (101, 102), 1 brand reply (103 replying to 101), 1 customer follow-up (104)
    data = {
        "tweet_id": ["101", "102", "103", "104"],
        "author_id": ["user_1", "user_2", "SpotifyCares", "user_1"],
        "inbound": ["True", "True", "False", "True"],
        "created_at": ["2023-01-01 10:00:00", "2023-01-01 10:01:00", "2023-01-01 10:05:00", "2023-01-01 10:10:00"],
        "in_response_to_tweet_id": [None, None, "101", "103"],
        "text_clean": [
            "Need help logging in to my account",
            "Why is my playlist buffering constantly",
            "Hey! Send us a DM with your account email.",
            "Thanks, DM sent",
        ],
        "thread_id": ["101", "102", "101", "101"],
    }
    pd.DataFrame(data).to_csv(clean_csv, index=False)
    pd.DataFrame({"tweet_id": []}).to_csv(smoke_csv, index=False)

    stats = generate_golden_set_candidates(
        clean_csv_path=clean_csv,
        smoke_sample_path=smoke_csv,
        output_candidates_path=out_candidates,
        output_debug_path=out_debug,
        output_pool_counts_path=out_pool_counts,
        openers_target=2,
        followups_target=1,
        min_per_bucket=1,
    )

    # 1. Verify pool counts CSV exists and contains expected counts
    assert out_pool_counts.exists()
    pool_counts_df = pd.read_csv(out_pool_counts)
    assert set(pool_counts_df.columns) == {"rough_intent_bucket", "count"}
    assert pool_counts_df["count"].sum() == 2

    # 2. Verify debug CSV contains historical_reply_text
    assert out_debug.exists()
    debug_df = pd.read_csv(out_debug, dtype=str)
    assert "historical_reply_text" in debug_df.columns
    row_101 = debug_df[debug_df["tweet_id"] == "101"]
    assert len(row_101) == 1
    assert row_101.iloc[0]["historical_reply_text"] == "Hey! Send us a DM with your account email."

    # 3. Verify candidates CSV retains blinded 9-column schema without historical_reply_text
    cand_df = pd.read_csv(out_candidates, dtype=str)
    assert "historical_reply_text" not in cand_df.columns
    assert len(cand_df.columns) == 9

