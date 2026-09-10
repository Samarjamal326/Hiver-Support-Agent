"""Sampling and filtering routines to extract brand-specific customer support dialogues."""

import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, Set
import pandas as pd
import yaml


def _normalize_id(val: Any) -> Optional[str]:
    """Normalize a tweet ID to a stripped string without floating point formatting."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "<na>"):
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


def extract_brand_slice(
    df: pd.DataFrame, brand: str, max_depth: int = 20
) -> pd.DataFrame:
    """Extract full conversation threads containing at least one reply from the brand.

    Args:
        df: Raw tweets DataFrame matching twcs.csv schema.
        brand: Brand name / author_id to sample (e.g. 'SpotifyCares').
        max_depth: Maximum traversal depth for ancestor and descendant walks.

    Returns:
        Filtered DataFrame containing full conversation threads with a derived
        'thread_id' column indicating the earliest ancestor of each thread.
    """
    df = df.copy()

    # Normalize ID columns to strings
    for col in [
        "tweet_id",
        "author_id",
        "in_response_to_tweet_id",
        "response_tweet_id",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(_normalize_id)

    # Normalize inbound column to boolean
    if "inbound" in df.columns:
        if df["inbound"].dtype != bool:
            df["inbound"] = (
                df["inbound"].astype(str).str.lower().isin(["true", "1", "t"])
            )

    # Build lookup dict: tweet_id -> row info for O(1) traversal
    lookup: Dict[str, Dict[str, Any]] = {
        tid: {
            "in_response_to_tweet_id": in_rep,
            "response_tweet_id": resp,
            "author_id": auth,
        }
        for tid, in_rep, resp, auth in zip(
            df["tweet_id"],
            df["in_response_to_tweet_id"],
            df["response_tweet_id"],
            df["author_id"],
        )
        if tid is not None
    }

    # Step A: find all tweet_ids where author_id == brand
    brand_reply_ids: Set[str] = set(
        df[df["author_id"] == brand]["tweet_id"].dropna()
    )

    # Step B: backward walk up in_response_to_tweet_id iteratively
    ancestors: Set[str] = set()
    for bid in brand_reply_ids:
        curr = bid
        depth = 0
        visited_in_walk: Set[str] = {curr}
        while depth < max_depth:
            info = lookup.get(curr)
            if not info:
                break
            parent = info.get("in_response_to_tweet_id")
            if not parent:
                break
            parent_id = _normalize_id(parent)
            if (
                not parent_id
                or parent_id in visited_in_walk
                or parent_id not in lookup
            ):
                break
            visited_in_walk.add(parent_id)
            ancestors.add(parent_id)
            curr = parent_id
            depth += 1

    # Step C: forward walk down response_tweet_id iteratively (supports comma-separated IDs)
    descendants: Set[str] = set()
    for bid in brand_reply_ids:
        queue = [(bid, 0)]
        visited_in_walk: Set[str] = {bid}
        while queue:
            curr, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            info = lookup.get(curr)
            if not info:
                continue
            resp = info.get("response_tweet_id")
            if not resp:
                continue
            for child in str(resp).split(","):
                child_id = _normalize_id(child)
                if (
                    child_id
                    and child_id not in visited_in_walk
                    and child_id in lookup
                ):
                    visited_in_walk.add(child_id)
                    descendants.add(child_id)
                    queue.append((child_id, depth + 1))

    # Union all IDs: brand replies + ancestors + descendants
    all_thread_tweet_ids = brand_reply_ids | ancestors | descendants
    slice_df = df[df["tweet_id"].isin(all_thread_tweet_ids)].copy()

    # Add derived column 'thread_id': earliest ancestor tweet_id in its chain
    root_memo: Dict[str, str] = {}

    def find_thread_root(tid: str) -> str:
        curr = tid
        depth = 0
        path = []
        visited = set()
        while depth < max_depth:
            if curr in root_memo:
                root = root_memo[curr]
                for node in path:
                    root_memo[node] = root
                return root

            info = lookup.get(curr)
            if not info:
                break
            parent = info.get("in_response_to_tweet_id")
            if not parent:
                break
            parent_id = _normalize_id(parent)
            if not parent_id or parent_id in visited or parent_id not in lookup:
                break
            visited.add(curr)
            path.append(curr)
            curr = parent_id
            depth += 1

        root = curr
        for node in path:
            root_memo[node] = root
        root_memo[tid] = root
        return root

    slice_df["thread_id"] = slice_df["tweet_id"].apply(find_thread_root)
    return slice_df


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load yaml configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/config.yaml") -> None:
    """Read config, run extraction, and save brand raw slice."""
    start_time = time.time()
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})

    raw_path = data_cfg.get(
        "raw_path", cfg.get("paths", {}).get("raw_data", "data/raw/twcs.csv")
    )
    processed_dir = data_cfg.get(
        "processed_dir",
        cfg.get("paths", {}).get("processed_dir", "data/processed"),
    )
    brand = data_cfg.get("brand", cfg.get("brand", "SpotifyCares"))
    max_depth = int(data_cfg.get("max_thread_traverse_depth", 20))

    print(f"Reading raw dataset from: {raw_path}")
    df = pd.read_csv(
        raw_path,
        dtype={
            "tweet_id": str,
            "author_id": str,
            "in_response_to_tweet_id": str,
            "response_tweet_id": str,
        },
    )
    total_raw_rows = len(df)

    print(f"Extracting conversation slice for brand: {brand} (max depth: {max_depth})")
    slice_df = extract_brand_slice(df, brand=brand, max_depth=max_depth)

    rows_in_slice = len(slice_df)
    unique_threads = slice_df["thread_id"].nunique()

    # Log required summary statistics to stdout
    print(f"total raw rows read: {total_raw_rows:,}")
    print(f"rows in brand slice: {rows_in_slice:,}")
    print(f"unique threads found: {unique_threads:,}")

    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(processed_dir) / f"{brand}_raw_slice.csv"
    slice_df.to_csv(out_path, index=False)
    print(f"Saved raw slice checkpoint to: {out_path}")

    elapsed_time = time.time() - start_time
    print(f"[sample_brand] Execution completed in: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")


if __name__ == "__main__":
    main()
