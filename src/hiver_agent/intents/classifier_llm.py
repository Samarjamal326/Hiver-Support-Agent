"""LLM-based zero/few-shot customer support intent classifier using Ollama."""

import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import yaml

# Ensure src root is in sys.path when executed directly as a script
_src_root = Path(__file__).resolve().parent.parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from hiver_agent.intents.cluster import load_thread_openers
from hiver_agent.llm.client import OllamaClient

logger = logging.getLogger(__name__)

# Single source of truth for intent labels
INTENT_TAXONOMY: List[str] = [
    "Account Access/Login",
    "Billing & Subscription",
    "Advertisement Complaints",
    "Playback & App Bugs",
    "Platform-Specific Technical",
    "Content & Feature Requests",
    "Family Plan Management",
    "Regional/Country Restriction",
    "General Complaint/Frustration",
    "Praise/Off-topic",
]

# Canonical intent descriptions
INTENT_DESCRIPTIONS: Dict[str, str] = {
    "Account Access/Login": "Issues logging in, password reset failures, account compromise, or third-party/Facebook login errors.",
    "Billing & Subscription": "Inquiries or disputes regarding subscription charges, renewals, student/promotional discounts, or payment method issues.",
    "Advertisement Complaints": "Complaints about ads playing on premium accounts, excessive or disturbing ads, or inability to skip advertisements.",
    "Playback & App Bugs": "Audio playback stopping/skipping, queue errors, disappearing songs/playlists, or offline sync bugs.",
    "Platform-Specific Technical": "Platform-specific issues including desktop app lag, mobile OS compatibility, search loading errors, or device-specific bugs.",
    "Content & Feature Requests": "Requests to add specific songs, albums, or artists, restore missing features like lyrics, or product improvements.",
    "Family Plan Management": "Adding/removing members from a family plan, plan switching, invitations, or partner billing transitions.",
    "Regional/Country Restriction": "Country mismatch errors when traveling abroad, regional licensing restrictions, or availability in specific countries.",
    "General Complaint/Frustration": "Expressions of overall dissatisfaction, venting about customer support quality, or general grievances without a bug report.",
    "Praise/Off-topic": "Expressions of gratitude, casual acknowledgments, confirmations (such as 'DM sent'), or non-support banter.",
}

# Few-shot exemplar pairs (2 real examples per intent category)
FEW_SHOT_EXAMPLES: List[Tuple[str, str]] = [
    (
        "Account Access/Login",
        "reset the password & tells me I can't do that be because I must use my FB! #HELP",
    ),
    (
        "Account Access/Login",
        "someone has attempted to compromise my account. Can still log in with facebook but email/password have been changed",
    ),
    (
        "Billing & Subscription",
        "my student discount expired... will it cancel my other membership? Or charge twice?",
    ),
    (
        "Billing & Subscription",
        "Trying to update payment info but asked to give College info for Student account (not a student).",
    ),
    (
        "Advertisement Complaints",
        "If I'm paying for premium, why do I still see advertisements?",
    ),
    (
        "Advertisement Complaints",
        "HELLO????? SKIP ADS? ANYONE THERE?",
    ),
    (
        "Playback & App Bugs",
        "since a recent update I can't add multiple songs to the queue on android, and the app is also very slow.",
    ),
    (
        "Playback & App Bugs",
        "Some songs disappeared from my offline playlist",
    ),
    (
        "Platform-Specific Technical",
        "why does your app lag so much on Desktop.",
    ),
    (
        "Platform-Specific Technical",
        "is the app going to be ready for the iPhone X when it releases?",
    ),
    (
        "Content & Feature Requests",
        "PUT ZHANG YIXING AKA LAY'S SECOND ALBUM SHEEP ON SPOTIFY RIGHT NOW",
    ),
    (
        "Content & Feature Requests",
        "any chance of the in application lyrics coming back",
    ),
    (
        "Family Plan Management",
        "how do I add a family member to our family plan?",
    ),
    (
        "Family Plan Management",
        "Is it possible to move individual accounts billed via Sprint to a family plan?",
    ),
    (
        "Regional/Country Restriction",
        "uh how do i change the country on my spotify acc",
    ),
    (
        "Regional/Country Restriction",
        "can you give an access to people in russia to use it",
    ),
    (
        "General Complaint/Frustration",
        "seems you do not care about your PR customers. CSR was like a robot and had to beg",
    ),
    (
        "General Complaint/Frustration",
        "Dear @115888, can you please take your money and let me listen to my music please",
    ),
    (
        "Praise/Off-topic",
        "LOOOOOOL thanks x",
    ),
    (
        "Praise/Off-topic",
        "DM sent :)",
    ),
]

JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def build_classification_prompt(message: str) -> str:
    """Construct an LLM classification prompt containing intent definitions, few-shot examples, and strict JSON output rules."""
    lines: List[str] = [
        "You are an expert customer support inquiry classification agent for a streaming music service.",
        "Your task is to classify incoming customer messages into exactly one intent category from the taxonomy below.",
        "",
        "=== INTENT TAXONOMY ===",
    ]

    for intent in INTENT_TAXONOMY:
        description = INTENT_DESCRIPTIONS.get(intent, "")
        lines.append(f"- {intent}: {description}")

    lines.extend(["", "=== FEW-SHOT EXAMPLES ==="])
    for intent, example_text in FEW_SHOT_EXAMPLES:
        example_json = json.dumps(
            {
                "intent": intent,
                "confidence": 1.0,
                "reasoning": f"Inquiry directly addresses {intent.lower()} matters.",
            }
        )
        lines.append(f'Input: "{example_text}"')
        lines.append(f"Output: {example_json}")
        lines.append("")

    lines.extend(
        [
            "=== TARGET MESSAGE ===",
            f'Input: "{message}"',
            "",
            "=== INSTRUCTIONS ===",
            "Classify the target message above into one of the exact intent categories.",
            'Respond ONLY with a JSON object matching this schema:',
            '{"intent": "<one of the exact taxonomy strings>", "confidence": <float between 0.0 and 1.0>, "reasoning": "<one sentence explanation>"}',
            "Do NOT include markdown formatting (no ```json code blocks), no preamble, and no follow-up text.",
        ]
    )
    return "\n".join(lines)


def parse_classification_response(raw_text: str) -> Dict[str, Any]:
    """Parse raw LLM response text into a validated intent classification dictionary with error fallback."""
    stripped = raw_text.strip()

    # Attempt to extract JSON block if surrounded by markdown or extra text
    match = JSON_BLOCK_PATTERN.search(stripped)
    json_candidate = match.group(0) if match else stripped

    try:
        parsed = json.loads(json_candidate)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed JSON root is not an object.")

        intent = str(parsed.get("intent", "")).strip()
        if intent not in INTENT_TAXONOMY:
            raise ValueError(f"Unknown intent: '{intent}' not in taxonomy.")

        confidence = float(parsed.get("confidence", 0.0))
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence {confidence} out of range [0.0, 1.0].")

        reasoning = str(parsed.get("reasoning", "")).strip()
        return {
            "intent": intent,
            "confidence": confidence,
            "reasoning": reasoning,
        }
    except Exception:
        preview = stripped.replace("\n", " ")[:200]
        return {
            "intent": "Unclassified",
            "confidence": 0.0,
            "reasoning": f"parse_error: {preview}",
        }


def classify_message(message: str, client: OllamaClient) -> Dict[str, Any]:
    """Classify a single message by generating an LLM response and parsing the JSON result."""
    prompt = build_classification_prompt(message)
    raw_response = client.generate(prompt)
    return parse_classification_response(raw_response)


def classify_dataframe(
    df: pd.DataFrame, text_column: str, client: OllamaClient
) -> pd.DataFrame:
    """Classify all messages in a DataFrame row by row and append prediction columns."""
    result_df = df.copy()
    intents: List[str] = []
    confidences: List[float] = []
    reasonings: List[str] = []

    total = len(result_df)
    for idx, (_, row) in enumerate(result_df.iterrows(), start=1):
        text = str(row.get(text_column, "")).strip()
        prediction = classify_message(text, client)
        intents.append(prediction["intent"])
        confidences.append(prediction["confidence"])
        reasonings.append(prediction["reasoning"])

        if idx % 25 == 0 or idx == total:
            print(f"Classified {idx}/{total} messages ({(idx / total) * 100:.1f}%)")

    result_df["predicted_intent"] = intents
    result_df["confidence"] = confidences
    result_df["reasoning"] = reasonings
    return result_df


def main(
    csv_path: Optional[str] = None,
    output_path: Optional[str] = None,
    sample_size: int = 20,
    random_state: int = 99,
) -> None:
    """Run smoke classification on 20 thread openers and write results to reports/classifier_smoke_sample.csv."""
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    config_file = root_dir / "configs" / "config.yaml"

    # Default fallback values
    base_url = "http://localhost:11434"
    model_name = "qwen2.5:3b-instruct"
    cache_dir = root_dir / "cache" / "llm_responses"
    brand = "SpotifyCares"

    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            brand = cfg.get("data", {}).get("brand", cfg.get("brand", brand))
            llm_cfg = cfg.get("llm", {})
            base_url = llm_cfg.get("ollama_base_url", base_url)
            model_name = llm_cfg.get("classification_model", model_name)
            cache_path_str = llm_cfg.get("cache_dir", "cache/llm_responses")
            cache_dir = root_dir / cache_path_str

    if csv_path is None:
        csv_path = str(root_dir / "data" / "processed" / f"{brand}_clean.csv")

    if output_path is None:
        output_path = str(root_dir / "reports" / "classifier_smoke_sample.csv")

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(
        csv_path,
        dtype={
            "tweet_id": str,
            "in_response_to_tweet_id": str,
            "response_tweet_id": str,
            "thread_id": str,
        },
    )

    openers = load_thread_openers(df)
    print(f"Found {len(openers)} thread openers. Sampling {sample_size} with random_state={random_state}...")
    sample_df = openers.sample(n=min(sample_size, len(openers)), random_state=random_state)

    text_col = "text_clean" if "text_clean" in sample_df.columns else "text"
    client = OllamaClient(base_url=base_url, model=model_name, cache_dir=cache_dir)

    print(f"Running LLM classification with model '{model_name}'...")
    classified_df = classify_dataframe(sample_df, text_column=text_col, client=client)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    classified_df.to_csv(out_file, index=False)
    print(f"Saved smoke sample results to {out_file}\n")

    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # Print clean readable preview table
    print("=" * 105)
    print(f"{'Message':<55} | {'Predicted Intent':<32} | {'Confidence':<10}")
    print("-" * 105)
    for _, row in classified_df.iterrows():
        raw_msg = str(row[text_col]).replace("\n", " ")[:52]
        msg_preview = raw_msg.encode("ascii", errors="replace").decode("ascii")
        intent = str(row["predicted_intent"])[:32]
        conf = f"{float(row['confidence']):.2f}"
        print(f"{msg_preview:<55} | {intent:<32} | {conf:<10}")
    print("=" * 105)


if __name__ == "__main__":
    main()
