"""Data cleaning and preprocessing routines for customer support tweets."""

import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, Optional
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
LEADING_MENTION_PATTERN = re.compile(r"^(?:@\w+\s*)+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def strip_urls(text: str) -> str:
    """Remove HTTP/HTTPS and www URLs from text completely without placeholder tokens."""
    if not isinstance(text, str):
        return ""
    return URL_PATTERN.sub("", text)


def strip_leading_mentions(text: str) -> str:
    """Remove leading @mentions used for threading while preserving mid-sentence mentions."""
    if not isinstance(text, str):
        return ""
    return LEADING_MENTION_PATTERN.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace and newlines into a single space and trim edges."""
    if not isinstance(text, str):
        return ""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Apply the ordered text cleaning pipeline: URLs -> leading mentions -> whitespace.

    Leaves existing masked PII tokens (__email__, __phone__, etc.) as-is.
    """
    text = strip_urls(text)
    text = strip_leading_mentions(text)
    text = normalize_whitespace(text)
    return text


def is_english_heuristic(text: str, threshold: float = 0.6) -> bool:
    """Heuristic to detect English text based on ASCII alphabetic character ratio.

    Calculates the ratio of ASCII alphabetic characters (a-z, A-Z) to total characters.
    """
    if not isinstance(text, str) or not text:
        return False
    total_chars = len(text)
    if total_chars == 0:
        return False
    ascii_letters = sum(1 for c in text if ("a" <= c <= "z" or "A" <= c <= "Z"))
    return (ascii_letters / total_chars) >= threshold


def mask_exact_duplicates(df: pd.DataFrame) -> pd.Series:
    """Boolean mask returning True for first occurrences of unique text_raw values."""
    col = "text_raw" if "text_raw" in df.columns else "text"
    return ~df.duplicated(subset=[col], keep="first")


def mask_min_length(df: pd.DataFrame, min_len: int = 3) -> pd.Series:
    """Boolean mask returning True for rows where text_clean length is >= min_len."""
    col = "text_clean" if "text_clean" in df.columns else "text"
    return df[col].fillna("").astype(str).str.len() >= min_len


def mask_english(df: pd.DataFrame, threshold: float = 0.6) -> pd.Series:
    """Boolean mask returning True for rows passing the ASCII English heuristic."""
    col = "text_clean" if "text_clean" in df.columns else "text"
    return df[col].apply(lambda t: is_english_heuristic(t, threshold=threshold))


def get_cleaning_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    step_counts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return dictionary with cleaning report metrics and row counts per step."""
    initial_count = len(df_before)
    final_count = len(df_after)
    total_dropped = initial_count - final_count
    retention_rate = (final_count / initial_count) if initial_count > 0 else 0.0

    report = {
        "initial_rows": initial_count,
        "final_rows": final_count,
        "total_dropped": total_dropped,
        "retention_rate": round(retention_rate, 4),
    }
    if step_counts:
        report.update(step_counts)
    return report


def clean_dataframe(
    df: pd.DataFrame,
    min_len: int = 3,
    english_threshold: float = 0.6,
    verbose: bool = True,
) -> pd.DataFrame:
    """Clean the brand slice into a modeling-ready dataset.

    Preserves text_raw and generates text_clean, logging before/after counts
    for each audit-tracked filtering step.
    """
    df = df.copy()

    # Preserve untouched original in text_raw
    if "text_raw" not in df.columns:
        df["text_raw"] = df["text"].astype(str) if "text" in df.columns else ""

    # Generate processed text_clean
    df["text_clean"] = df["text_raw"].apply(clean_text)

    initial_len = len(df)
    if verbose:
        print(f"[clean_dataframe] Starting cleaning: {initial_len:,} initial rows")

    # Filter 1: Drop exact duplicate text_raw values (keep first)
    dup_mask = mask_exact_duplicates(df)
    dropped_dups = int((~dup_mask).sum())
    df = df[dup_mask].copy()
    if verbose:
        print(
            f"[Filter 1: Deduplication] Kept {len(df):,} rows (dropped {dropped_dups:,} exact duplicates)"
        )

    # Filter 2: Drop rows where text_clean is empty or under min_len characters
    len_mask = mask_min_length(df, min_len=min_len)
    dropped_short = int((~len_mask).sum())
    df = df[len_mask].copy()
    if verbose:
        print(
            f"[Filter 2: Min Length] Kept {len(df):,} rows (dropped {dropped_short:,} rows < {min_len} chars)"
        )

    # Filter 3: Drop non-English rows via ASCII ratio heuristic
    eng_mask = mask_english(df, threshold=english_threshold)
    dropped_non_eng = int((~eng_mask).sum())
    df = df[eng_mask].copy()
    if verbose:
        print(
            f"[Filter 3: English Heuristic] Kept {len(df):,} rows (dropped {dropped_non_eng:,} non-English rows)"
        )

    return df


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/config.yaml") -> None:
    """Read raw slice CSV, clean it, write clean CSV, and print report."""
    start_time = time.time()
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})

    processed_dir = data_cfg.get(
        "processed_dir",
        cfg.get("paths", {}).get("processed_dir", "data/processed"),
    )
    brand = data_cfg.get("brand", cfg.get("brand", "SpotifyCares"))

    raw_slice_path = Path(processed_dir) / f"{brand}_raw_slice.csv"
    if not raw_slice_path.exists():
        raise FileNotFoundError(
            f"Raw slice CSV not found at {raw_slice_path}. Run sample_brand.py first."
        )

    print(f"Reading raw slice from: {raw_slice_path}")
    raw_df = pd.read_csv(raw_slice_path, dtype=str)

    cleaned_df = clean_dataframe(raw_df, verbose=True)

    out_clean_path = Path(processed_dir) / f"{brand}_clean.csv"
    cleaned_df.to_csv(out_clean_path, index=False)
    print(f"Saved cleaned dataset to: {out_clean_path}")

    # Generate and print cleaning report
    report = get_cleaning_report(raw_df, cleaned_df)
    print("\n--- Cleaning Report ---")
    for k, v in report.items():
        print(f"  {k}: {v}")

    elapsed_time = time.time() - start_time
    print(f"\n[clean] Execution completed in: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")


if __name__ == "__main__":
    main()
