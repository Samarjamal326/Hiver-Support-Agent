"""Grounded reply drafting using retrieved historical customer-reply exemplars.

ANTI-FABRICATION GUARDRAIL: The prompt explicitly forbids the model from inventing
specific refund amounts, dates, or policies that do not appear in the exemplars.
grounding_tweet_ids in the structured JSON output enforces auditability — if a reply
cites IDs that do not appear in the retrieved exemplars or cites nothing while using
exemplar content, it signals free-generation rather than grounded retrieval.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# Ensure src root is in sys.path when executed directly as a script
_src_root = Path(__file__).resolve().parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from hiver_agent.llm.client import OllamaClient
from hiver_agent.embeddings import DEFAULT_MODEL_NAME, embed_texts, load_embedding_model
from hiver_agent.retrieval.index import build_index, retrieve_top_k

logger = logging.getLogger(__name__)

# Maximum exemplars to show in the prompt
MAX_EXEMPLARS: int = 3

# Fallback returned when JSON parsing fails after one retry.
# NEVER return an empty/success-looking result on failure.
FALLBACK_RESULT: Dict[str, Any] = {
    "reply_text": "",
    "grounding_tweet_ids": [],
    "notes": "generation_failed",
}


def build_reply_prompt(
    customer_text: str,
    intent: str,
    exemplars: List[Dict[str, Any]],
) -> str:
    """Construct a few-shot grounded reply prompt from retrieved historical exemplars.

    Args:
        customer_text: The incoming customer message to reply to.
        intent: Predicted intent category (e.g. "Billing & Subscription").
        exemplars: Top-k output from retrieve_top_k. Each dict contains
            customer_text, reply_text, reply_tweet_id, and similarity_score.

    Returns:
        Fully-formatted prompt string including the anti-fabrication guardrail
        and requirement to cite grounding_tweet_ids.
    """
    n_show = min(MAX_EXEMPLARS, len(exemplars))
    exemplar_lines: List[str] = []
    for i, ex in enumerate(exemplars[:n_show], 1):
        r_id = ex["reply_tweet_id"]
        score = ex["similarity_score"]
        exemplar_lines.append(
            f"--- Past Example {i} (reply_tweet_id={r_id}, similarity={score:.4f}) ---"
        )
        exemplar_lines.append(f"Customer: {ex['customer_text']}")
        exemplar_lines.append(f"Brand Reply: {ex['reply_text']}")
        exemplar_lines.append("")

    exemplar_block = "\n".join(exemplar_lines).strip()
    available_ids = ", ".join(ex["reply_tweet_id"] for ex in exemplars[:n_show])

    prompt = (
        "You are a customer support assistant for a SaaS brand. "
        "Draft a reply to the following customer message, grounded exclusively in how "
        "this brand has actually resolved similar issues in the past.\n\n"
        f"Customer intent category: {intent}\n\n"
        "Retrieved historical examples of how this brand has responded to similar issues:\n\n"
        f"{exemplar_block}\n\n"
        "Now draft a reply to this NEW customer message:\n"
        f"Customer: {customer_text}\n\n"
        "STRICT RULES you must follow:\n"
        "1. Mirror the brand tone, length, and resolution approach shown in the examples.\n"
        "2. DO NOT invent specific refund amounts, dates, policy terms, or account details "
        "that are not explicitly stated in the examples above.\n"
        "3. If none of the examples provide a clear resolution path, politely offer to look "
        "into it further (e.g. ask the customer to send a DM).\n"
        "4. In grounding_tweet_ids, cite which example reply_tweet_id(s) most directly "
        "informed your draft. If you drew from none, leave the list empty.\n"
        f"   Available IDs to cite: [{available_ids}]\n\n"
        "Return ONLY a valid JSON object with exactly these three fields, no extra text:\n"
        '{"reply_text": "<drafted reply>", "grounding_tweet_ids": ["<id>"], "notes": "<caveats>"}'
    )
    return prompt


def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse a JSON dict from raw model output. Returns None on parse failure."""
    text = raw.strip()
    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def draft_reply(
    customer_text: str,
    intent: str,
    exemplars: List[Dict[str, Any]],
    llm_client: OllamaClient,
) -> Dict[str, Any]:
    """Draft a grounded brand reply using the LLM, with a single JSON-strict retry.

    Args:
        customer_text: Incoming customer message text.
        intent: Predicted intent category string.
        exemplars: Retrieved exemplar dicts from retrieve_top_k.
        llm_client: Configured OllamaClient instance (reused from intents module).

    Returns:
        Dict with keys reply_text (str), grounding_tweet_ids (list[str]), notes (str).
        On malformed JSON after one retry, returns FALLBACK_RESULT with
        notes="generation_failed". Never silently returns empty/success-looking result.
    """
    prompt = build_reply_prompt(customer_text, intent, exemplars)
    raw = llm_client.generate(prompt, temperature=0.2)
    parsed = _parse_json_response(raw)

    if parsed is None:
        logger.warning("Malformed JSON on first attempt; retrying with strict JSON instruction")
        strict_prompt = (
            prompt
            + "\n\nIMPORTANT: Your previous response was not valid JSON. "
            "Return ONLY valid JSON in the exact format specified — no markdown fences, "
            "no explanation, no extra text before or after the JSON object."
        )
        raw_retry = llm_client.generate(strict_prompt, temperature=0.0)
        parsed = _parse_json_response(raw_retry)

    if parsed is None:
        logger.error("JSON parsing failed after one retry; returning fallback result")
        return FALLBACK_RESULT.copy()

    return {
        "reply_text": str(parsed.get("reply_text", "")),
        "grounding_tweet_ids": list(parsed.get("grounding_tweet_ids", [])),
        "notes": str(parsed.get("notes", "")),
    }


def main(config_path: Optional[str] = None) -> None:
    """Draft replies for 5 sample customer messages; print customer text, cited IDs, and draft."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    cfg_file = Path(config_path) if config_path else root_dir / "configs" / "config.yaml"

    brand = "SpotifyCares"
    ollama_url = "http://localhost:11434"
    gen_model = "qwen2.5:3b-instruct"
    cache_dir_str = str(root_dir / "cache" / "llm_responses")

    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        brand = cfg.get("data", {}).get("brand", cfg.get("brand", brand))
        llm_cfg = cfg.get("llm", {})
        ollama_url = llm_cfg.get("ollama_base_url", ollama_url)
        _gen = llm_cfg.get("generation_model", "").strip()
        gen_model = _gen if _gen else llm_cfg.get("classification_model", gen_model)
        _cd = llm_cfg.get("cache_dir", "")
        if _cd:
            cache_dir_str = _cd if Path(_cd).is_absolute() else str(root_dir / _cd)

    llm_client = OllamaClient(base_url=ollama_url, model=gen_model, cache_dir=cache_dir_str)

    processed_dir = root_dir / "data" / "processed"
    pairs_csv = processed_dir / f"{brand}_retrieval_pairs.csv"
    embeddings_npy = processed_dir / f"{brand}_retrieval_embeddings.npy"

    if not pairs_csv.exists() or not embeddings_npy.exists():
        logger.error(
            "Retrieval artifacts not found at %s / %s. Run retrieval/embed.py first.",
            pairs_csv,
            embeddings_npy,
        )
        return

    pairs_df = pd.read_csv(pairs_csv, dtype=str)
    embeddings = np.load(embeddings_npy)
    index = build_index(embeddings)
    emb_model = load_embedding_model(DEFAULT_MODEL_NAME)

    smoke_path = root_dir / "reports" / "classifier_smoke_sample.csv"
    sample_queries: List[Dict[str, str]] = []
    if smoke_path.exists():
        smoke_df = pd.read_csv(smoke_path, dtype=str)
        text_col = "text_clean" if "text_clean" in smoke_df.columns else "text"
        for idx in [0, 4, 9, 14, 19]:
            if idx < len(smoke_df):
                row = smoke_df.iloc[idx]
                sample_queries.append(
                    {
                        "tweet_id": str(row["tweet_id"]).strip(),
                        "text": str(row[text_col]),
                        "intent": str(row.get("predicted_intent", "Unknown")),
                    }
                )

    if not sample_queries:
        sample_queries = [
            {"tweet_id": "d1", "text": "Can not log in, password reset link broken", "intent": "Account Access/Login"},
            {"tweet_id": "d2", "text": "Charged twice this month for student plan", "intent": "Billing & Subscription"},
            {"tweet_id": "d3", "text": "Songs stop after 30 seconds on iPhone", "intent": "Playback & App Bugs"},
            {"tweet_id": "d4", "text": "Why are there ads on my premium account", "intent": "Advertisement Complaints"},
            {"tweet_id": "d5", "text": "Please add more albums from this artist", "intent": "Content & Feature Requests"},
        ]

    print("\n" + "=" * 80)
    print("              Generation Sample: 5 Grounded Reply Drafts")
    print("=" * 80)

    for q_idx, query_info in enumerate(sample_queries, 1):
        tweet_id = query_info["tweet_id"]
        customer_text = query_info["text"]
        intent = query_info["intent"]

        safe_text = customer_text.encode("ascii", errors="replace").decode("ascii")
        print(f"\n[{q_idx}] tweet_id={tweet_id} | intent={intent}")
        print(f"  Customer: \"{safe_text}\"")

        query_emb = embed_texts([customer_text], emb_model)
        exemplars = retrieve_top_k(query_emb, index, pairs_df, k=3, exclude_tweet_id=tweet_id)
        retrieved_ids = [e["reply_tweet_id"] for e in exemplars]
        print(f"  Retrieved exemplar IDs: {retrieved_ids}")

        result = draft_reply(customer_text, intent, exemplars, llm_client)
        reply_text = result["reply_text"].encode("ascii", errors="replace").decode("ascii")
        notes = result["notes"].encode("ascii", errors="replace").decode("ascii")

        print(f"  Cited grounding IDs: {result['grounding_tweet_ids']}")
        print(f"  Drafted reply: \"{reply_text}\"")
        if notes and notes != "generation_failed":
            print(f"  Notes: {notes}")
        if result["notes"] == "generation_failed":
            print("  [WARNING] generation_failed — JSON parsing failed after retry")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

