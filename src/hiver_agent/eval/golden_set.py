"""Golden set candidate sampling and formatting pipeline for human annotation.

CRITICAL ARCHITECTURAL CONSTRAINT:
This module must NEVER invoke an LLM classifier (e.g. classify_message,
classify_dataframe, or OllamaClient) to predict or pre-fill intent, difficulty,
or escalation labels. Its only responsibility is sampling candidate customer
messages, formatting thread contexts, and preparing a blinded sheet for human
ground-truth annotation.
"""

import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import yaml

# Ensure src root is in sys.path when executed directly as a script
_src_root = Path(__file__).resolve().parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from hiver_agent.intents.classifier_llm import (
    FEW_SHOT_EXAMPLES,
    INTENT_TAXONOMY,
)

logger = logging.getLogger(__name__)

DEFAULT_BRAND = "SpotifyCares"
DEFAULT_OPENERS_TARGET = 140
DEFAULT_FOLLOWUPS_TARGET = 60
DEFAULT_MIN_PER_BUCKET = 10
DEFAULT_RANDOM_STATE = 123


def rough_intent_bucket(text: str) -> str:
    """Map customer text to one of the 10 canonical taxonomy intents or 'Unclear'.

    CRITICAL NOTICE:
    This function is an APPROXIMATE regex/keyword heuristic used solely to balance
    stratified sampling across diverse support topics so that minority categories
    are adequately represented for human annotators.
    It is NEVER shown to human labelers as a suggested or pre-filled answer, and
    it is NEVER claimed or used as accurate classification.
    """
    t = str(text).lower()

    # 1. Family Plan Management (checked early to avoid generic plan/account collisions)
    if re.search(r"\b(family\s*plan|family\s*member|family\s*account|add\s*member|remove\s*member|invite\s*member|invitation|head\s*of\s*household)\b", t) or re.search(r"\bfamily\b", t):
        return "Family Plan Management"

    # 2. Advertisement Complaints
    if re.search(r"\b(ads?|advertisements?|commercials?|audio\s*ads?|pop[\s-]?ups?|video\s*ads?|skip\s*ads?)\b", t):
        return "Advertisement Complaints"

    # 3. Regional/Country Restriction
    if re.search(r"\b(country|region|regional|abroad|travel|licensing|territory|overseas|zip\s*code|foreign|location)\b", t):
        return "Regional/Country Restriction"

    # 4. Account Access/Login
    if re.search(
        r"\b(login|log\s*in|logging\s*in|logged\s*out|password|passwords|compromis|hacked|locked\s*out|verification\s*code|reset\s*password|cant\s*log|can't\s*log|cannot\s*log|sign\s*in|signing\s*in|sign-in|auth|authenticate|credentials)\b",
        t,
    ):
        return "Account Access/Login"

    # 5. Billing & Subscription
    if re.search(
        r"\b(charged?|charges?|charging|refund|refunds|discount|student|payment|payments|billing|subscription|subscriptions|premium|receipt|invoice|credit\s*card|debit|paypal|pay\b|paid|price|pricing|cancel|canceling|cancelled|renew|renewal|overcharged)\b",
        t,
    ):
        return "Billing & Subscription"

    # 6. Platform-Specific Technical
    if re.search(
        r"\b(iphone|ipad|android|windows|mac\b|macos|desktop|alexa|carplay|roku|chromecast|bluetooth|browser|web\s*player|update|updates|updated|ios|app\s*store|play\s*store|crash|crashes|crashing)\b",
        t,
    ):
        return "Platform-Specific Technical"

    # 7. Content & Feature Requests
    if re.search(
        r"\b(add\s+song|put\s+song|album|albums|artist|artists|song\s*request|feature|features|suggest|suggestion|bring\s*back|lyrics|discography|release|releases|catalog|missing\s*song|remove\s*song)\b",
        t,
    ):
        return "Content & Feature Requests"

    # 8. Playback & App Bugs
    if re.search(
        r"\b(play|playback|pause|buffering?|skip|skipping|queue|offline|freeze|freezing|stuck|glitch|bug|bugs|blank|black\s*screen|stopped|volume|shuffle|repeat|stutter|disappeared)\b",
        t,
    ):
        return "Playback & App Bugs"

    # 9. Praise/Off-topic
    if re.search(
        r"\b(thank|thanks|thankyou|love\s*you|awesome|kudos|shoutout|great|dm\s*sent|sent\s*dm|haha|lol|cool|good\s*job|appreciate)\b",
        t,
    ):
        return "Praise/Off-topic"

    # 10. General Complaint/Frustration
    if re.search(
        r"\b(worst|terrible|hate|sucks?|annoying|useless|unhelpful|frustrat|garbage|trash|horrible|angry|pissed|ridiculous|disgusted|unacceptable|ditching|bye\s*spotify)\b",
        t,
    ):
        return "General Complaint/Frustration"

    return "Unclear"


def filter_candidate_pool(
    df: pd.DataFrame,
    few_shot_texts: Set[str],
    smoke_tweet_ids: Set[str],
) -> pd.DataFrame:
    """Filter clean data to customer-authored tweets excluding few-shot and smoke-test examples.

    Args:
        df: Raw cleaned interactions dataframe.
        few_shot_texts: Set of exact text_clean strings used in few-shot exemplars.
        smoke_tweet_ids: Set of tweet_ids evaluated in the classifier smoke test.

    Returns:
        Filtered candidate DataFrame containing customer-authored inquiries.
    """
    if df.empty:
        return df.copy()

    text_col = "text_clean" if "text_clean" in df.columns else "text"

    # 1. Filter to customer-authored messages (inbound == True)
    inbound_mask = df["inbound"].astype(str).str.strip().str.lower() == "true"
    candidates = df[inbound_mask].copy()

    # 2. Exclude rows matching smoke test tweet_ids
    norm_smoke_ids = {
        str(tid).strip()[:-2] if str(tid).strip().endswith(".0") else str(tid).strip()
        for tid in smoke_tweet_ids
    }
    tid_series = candidates["tweet_id"].astype(str).str.strip()
    tid_series = tid_series.apply(lambda s: s[:-2] if s.endswith(".0") else s)
    candidates = candidates[~tid_series.isin(norm_smoke_ids)]

    # 3. Exclude rows whose text_clean exactly matches any few-shot exemplar text
    text_series = candidates[text_col].astype(str).str.strip()
    candidates = candidates[~text_series.isin(few_shot_texts)]

    return candidates


def split_openers_and_followups(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split candidate pool into conversation openers and multi-turn follow-ups."""
    if df.empty:
        return df.copy(), df.copy()

    def _is_empty(val: Any) -> bool:
        if val is None or pd.isna(val):
            return True
        s = str(val).strip().lower()
        return s in ("", "nan", "none", "<na>")

    is_opener = df["in_response_to_tweet_id"].apply(_is_empty)
    openers = df[is_opener].copy()
    follow_ups = df[~is_opener].copy()
    return openers, follow_ups


def stratified_sample_openers(
    openers_df: pd.DataFrame,
    target_total: int = DEFAULT_OPENERS_TARGET,
    min_per_bucket: int = DEFAULT_MIN_PER_BUCKET,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Sample thread openers with a minimum floor per named taxonomy bucket.

    Args:
        openers_df: Pool of thread opener messages.
        target_total: Total number of openers to sample (default 140).
        min_per_bucket: Minimum floor of samples per named taxonomy category (default 10).
        random_state: Seed for deterministic sampling.

    Returns:
        Tuple of (sampled openers DataFrame, dictionary of bucket counts).
    """
    if openers_df.empty:
        return openers_df.copy(), {}

    text_col = "text_clean" if "text_clean" in openers_df.columns else "text"
    df = openers_df.copy()
    df["rough_intent_bucket"] = df[text_col].astype(str).apply(rough_intent_bucket)

    selected_dfs: List[pd.DataFrame] = []
    sampled_indices: Set[Any] = set()

    # Phase 1: Minimum floor per named taxonomy bucket
    for intent in INTENT_TAXONOMY:
        bucket_subset = df[df["rough_intent_bucket"] == intent]
        n_sample = min(min_per_bucket, len(bucket_subset))
        if n_sample > 0:
            sample = bucket_subset.sample(n=n_sample, random_state=random_state)
            selected_dfs.append(sample)
            sampled_indices.update(sample.index)

    initial_selected = pd.concat(selected_dfs) if selected_dfs else pd.DataFrame(columns=df.columns)
    remaining_needed = target_total - len(initial_selected)

    # Phase 2: Fill remaining quota from the leftover pool (including unpicked and Unclear)
    leftover_pool = df[~df.index.isin(sampled_indices)]
    if remaining_needed > 0 and len(leftover_pool) > 0:
        fill_count = min(remaining_needed, len(leftover_pool))
        fill_sample = leftover_pool.sample(n=fill_count, random_state=random_state)
        final_openers = pd.concat([initial_selected, fill_sample])
    else:
        final_openers = initial_selected

    bucket_counts = final_openers["rough_intent_bucket"].value_counts().to_dict()
    return final_openers, bucket_counts


def sample_follow_ups(
    follow_ups_df: pd.DataFrame,
    target_total: int = DEFAULT_FOLLOWUPS_TARGET,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Randomly sample follow-up messages from the candidate pool."""
    if follow_ups_df.empty:
        return follow_ups_df.copy()

    n_sample = min(target_total, len(follow_ups_df))
    return follow_ups_df.sample(n=n_sample, random_state=random_state).copy()


def build_thread_context(
    follow_up_row: Any,
    full_df: pd.DataFrame,
) -> str:
    """Construct turn-by-turn conversational history for a message within its thread.

    Retrieves all prior messages in the thread in ascending chronological order
    up to and excluding the current message. Returns an empty string for openers.

    Args:
        follow_up_row: Row representing the candidate message (Series or Dict).
        full_df: Full conversation DataFrame containing all thread messages.

    Returns:
        Formatted multi-turn dialogue string (e.g. 'Customer: ...\\nAgent: ...').
    """
    parent_raw = (
        follow_up_row.get("in_response_to_tweet_id")
        if hasattr(follow_up_row, "get")
        else getattr(follow_up_row, "in_response_to_tweet_id", None)
    )
    if parent_raw is None or pd.isna(parent_raw):
        return ""
    if str(parent_raw).strip().lower() in ("", "nan", "none", "<na>"):
        return ""

    thid = str(
        follow_up_row.get("thread_id")
        if hasattr(follow_up_row, "get")
        else getattr(follow_up_row, "thread_id", "")
    ).strip()
    if not thid or thid.lower() in ("nan", "none", "<na>"):
        return ""

    curr_tid = str(
        follow_up_row.get("tweet_id")
        if hasattr(follow_up_row, "get")
        else getattr(follow_up_row, "tweet_id", "")
    ).strip()
    if curr_tid.endswith(".0"):
        curr_tid = curr_tid[:-2]

    # Pull all messages sharing the same thread_id
    t_mask = full_df["thread_id"].astype(str).str.strip() == thid
    thread_messages = full_df[t_mask].copy()
    if thread_messages.empty:
        return ""

    # Normalize tweet IDs
    clean_tids = thread_messages["tweet_id"].astype(str).str.strip().apply(
        lambda s: s[:-2] if s.endswith(".0") else s
    )
    # Exclude the follow-up message itself
    priors = thread_messages[clean_tids != curr_tid].copy()
    if priors.empty:
        return ""

    # Parse timestamps for chronological sorting
    priors["_dt"] = pd.to_datetime(priors["created_at"], errors="coerce", format="mixed")
    curr_created = (
        follow_up_row.get("created_at")
        if hasattr(follow_up_row, "get")
        else getattr(follow_up_row, "created_at", None)
    )
    curr_dt = pd.to_datetime(curr_created, errors="coerce", format="mixed")

    if pd.notna(curr_dt):
        # Keep strictly antecedent turns
        priors = priors[priors["_dt"] <= curr_dt]

    priors = priors.sort_values(by=["_dt", "tweet_id"], ascending=[True, True])

    text_col = "text_clean" if "text_clean" in priors.columns else "text"
    lines: List[str] = []
    for _, msg in priors.iterrows():
        is_inbound = str(msg.get("inbound", "")).strip().lower() == "true"
        speaker = "Customer" if is_inbound else "Agent"
        msg_text = str(msg.get(text_col, msg.get("text", ""))).strip()
        lines.append(f"{speaker}: {msg_text}")

    return "\n".join(lines)


def generate_golden_set_candidates(
    clean_csv_path: Path,
    smoke_sample_path: Path,
    output_candidates_path: Path,
    output_debug_path: Optional[Path] = None,
    openers_target: int = DEFAULT_OPENERS_TARGET,
    followups_target: int = DEFAULT_FOLLOWUPS_TARGET,
    min_per_bucket: int = DEFAULT_MIN_PER_BUCKET,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """Execute end-to-end golden set sampling pipeline and write candidates sheet."""
    logger.info("Loading cleaned dataset from %s", clean_csv_path)
    full_df = pd.read_csv(clean_csv_path, dtype=str)
    total_raw_rows = len(full_df)

    # Prepare exclusion sets
    few_shot_texts: Set[str] = {txt.strip() for _, txt in FEW_SHOT_EXAMPLES}
    smoke_tweet_ids: Set[str] = set()
    if smoke_sample_path.exists():
        smoke_df = pd.read_csv(smoke_sample_path, dtype=str)
        smoke_tweet_ids = {str(tid).strip() for tid in smoke_df["tweet_id"].dropna()}

    logger.info("Filtering candidate pool (inbound=True, excluding few-shot and smoke samples)...")
    candidate_pool = filter_candidate_pool(full_df, few_shot_texts, smoke_tweet_ids)
    candidate_pool_size = len(candidate_pool)

    # Split openers vs follow-ups
    openers_pool, followups_pool = split_openers_and_followups(candidate_pool)
    logger.info(
        "Candidate pool size: %d (Openers: %d, Follow-ups: %d)",
        candidate_pool_size,
        len(openers_pool),
        len(followups_pool),
    )

    # Sample openers with minimum floor stratification
    sampled_openers, bucket_counts = stratified_sample_openers(
        openers_pool,
        target_total=openers_target,
        min_per_bucket=min_per_bucket,
        random_state=random_state,
    )
    sampled_openers["position"] = "opener"
    sampled_openers["thread_context"] = ""

    # Sample follow-ups
    sampled_followups = sample_follow_ups(
        followups_pool,
        target_total=followups_target,
        random_state=random_state,
    )
    sampled_followups["position"] = "follow_up"
    sampled_followups["rough_intent_bucket"] = sampled_followups["text_clean"].apply(rough_intent_bucket)

    logger.info("Building conversational thread contexts for %d follow-ups...", len(sampled_followups))
    contexts: List[str] = []
    for _, row in sampled_followups.iterrows():
        ctx = build_thread_context(row, full_df)
        contexts.append(ctx)
    sampled_followups["thread_context"] = contexts

    # Combine sampled sets
    combined = pd.concat([sampled_openers, sampled_followups], ignore_index=True)

    text_col = "text_clean" if "text_clean" in combined.columns else "text"
    combined["message_text"] = combined[text_col]

    # Initialize human annotation columns (strictly blank)
    combined["human_intent"] = ""
    combined["human_difficulty"] = ""
    combined["human_escalate"] = ""
    combined["notes"] = ""

    # Define strict output schema without rough_intent_bucket (blinded labeling)
    output_columns = [
        "tweet_id",
        "thread_id",
        "position",
        "thread_context",
        "message_text",
        "human_intent",
        "human_difficulty",
        "human_escalate",
        "notes",
    ]
    candidates_df = combined[output_columns].copy()

    output_candidates_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(output_candidates_path, index=False, encoding="utf-8")
    logger.info("Wrote %d golden set candidates to %s", len(candidates_df), output_candidates_path)

    # Optional debug sheet for auditing bucket allocations
    if output_debug_path:
        debug_columns = [
            "tweet_id",
            "thread_id",
            "position",
            "rough_intent_bucket",
            "message_text",
        ]
        debug_df = combined[debug_columns].copy()
        output_debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_df.to_csv(output_debug_path, index=False, encoding="utf-8")
        logger.info("Wrote debug bucket allocations to %s", output_debug_path)

    return {
        "total_raw_rows": total_raw_rows,
        "candidate_pool_size": candidate_pool_size,
        "openers_pool_size": len(openers_pool),
        "followups_pool_size": len(followups_pool),
        "sampled_openers_count": len(sampled_openers),
        "sampled_followups_count": len(sampled_followups),
        "total_sampled": len(candidates_df),
        "opener_bucket_counts": bucket_counts,
    }


def main(config_path: Optional[str] = None) -> None:
    """CLI entry point to run golden set candidate sampling."""
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

    clean_csv_path = root_dir / "data" / "processed" / f"{brand}_clean.csv"
    smoke_sample_path = root_dir / "reports" / "classifier_smoke_sample.csv"
    output_candidates_path = root_dir / "reports" / "golden_set_candidates.csv"
    output_debug_path = root_dir / "reports" / "golden_set_bucket_debug.csv"

    stats = generate_golden_set_candidates(
        clean_csv_path=clean_csv_path,
        smoke_sample_path=smoke_sample_path,
        output_candidates_path=output_candidates_path,
        output_debug_path=output_debug_path,
    )

    print("\n" + "=" * 80)
    print("                 Golden Set Candidate Sampling Summary")
    print("=" * 80)
    print(f"Total Raw Rows in Cleaned Data: {stats['total_raw_rows']:,}")
    print(f"Candidate Pool (Inbound, Exclusions Applied): {stats['candidate_pool_size']:,}")
    print(f"  - Available Openers: {stats['openers_pool_size']:,}")
    print(f"  - Available Follow-ups: {stats['followups_pool_size']:,}")
    print(f"Sampled Openers: {stats['sampled_openers_count']}")
    print(f"Sampled Follow-ups: {stats['sampled_followups_count']}")
    print(f"Total Golden Set Candidates: {stats['total_sampled']}")
    print("-" * 80)
    print("Opener Stratification Distribution (by rough_intent_bucket):")
    for bucket, cnt in sorted(stats["opener_bucket_counts"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {bucket:<32}: {cnt} items")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
