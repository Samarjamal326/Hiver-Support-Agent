"""Unit tests for LLM intent classifier and prompt construction."""

from typing import Any, Dict
import pandas as pd
import pytest

from hiver_agent.intents.classifier_llm import (
    FEW_SHOT_EXAMPLES,
    INTENT_TAXONOMY,
    build_classification_prompt,
    classify_dataframe,
    classify_message,
    parse_classification_response,
)


class MockOllamaClient:
    """Test double providing deterministic responses without network calls."""

    def __init__(self, response_text: str = "") -> None:
        self.response_text = response_text
        self.last_prompt = ""
        self.last_temperature = 0.0
        self.call_count = 0

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self.last_prompt = prompt
        self.last_temperature = temperature
        self.call_count += 1
        return self.response_text


def test_build_classification_prompt_includes_message_and_all_taxonomy_names() -> None:
    """Verify build_classification_prompt contains the target message and all 10 taxonomy categories."""
    target_msg = "Can you please help me recover my account?"
    prompt = build_classification_prompt(target_msg)

    assert target_msg in prompt
    for intent in INTENT_TAXONOMY:
        assert intent in prompt

    # Verify few-shot examples are included in the prompt
    for intent, example in FEW_SHOT_EXAMPLES:
        assert intent in prompt
        assert example in prompt


def test_parse_classification_response_valid_json() -> None:
    """Verify parse_classification_response correctly parses well-formed JSON."""
    raw = '{"intent": "Billing & Subscription", "confidence": 0.95, "reasoning": "User asks about subscription charges."}'
    result = parse_classification_response(raw)

    assert result["intent"] == "Billing & Subscription"
    assert result["confidence"] == 0.95
    assert result["reasoning"] == "User asks about subscription charges."


def test_parse_classification_response_extracts_json_from_markdown_fences() -> None:
    """Verify parse_classification_response correctly extracts JSON even if enclosed in markdown code blocks."""
    raw = """Here is your classification:
```json
{
  "intent": "Account Access/Login",
  "confidence": 0.88,
  "reasoning": "User forgotten password."
}
```"""
    result = parse_classification_response(raw)

    assert result["intent"] == "Account Access/Login"
    assert result["confidence"] == 0.88
    assert result["reasoning"] == "User forgotten password."


def test_parse_classification_response_fallback_on_malformed_json() -> None:
    """Verify parse_classification_response never crashes on malformed inputs and returns Unclassified."""
    # Test case 1: Non-JSON plain text
    res1 = parse_classification_response("Sorry, I cannot classify this message.")
    assert res1["intent"] == "Unclassified"
    assert res1["confidence"] == 0.0
    assert res1["reasoning"].startswith("parse_error:")

    # Test case 2: Missing required intent key
    res2 = parse_classification_response('{"confidence": 0.9, "reasoning": "No intent"}')
    assert res2["intent"] == "Unclassified"
    assert res2["confidence"] == 0.0

    # Test case 3: Unknown intent label not in taxonomy
    res3 = parse_classification_response('{"intent": "Unknown Intent", "confidence": 0.9, "reasoning": "Wrong"}')
    assert res3["intent"] == "Unclassified"
    assert res3["confidence"] == 0.0

    # Test case 4: Confidence out of range
    res4 = parse_classification_response('{"intent": "Praise/Off-topic", "confidence": 1.5, "reasoning": "Over 1.0"}')
    assert res4["intent"] == "Unclassified"
    assert res4["confidence"] == 0.0

    # Test case 5: Empty input
    res5 = parse_classification_response("")
    assert res5["intent"] == "Unclassified"
    assert res5["confidence"] == 0.0


def test_classify_message_with_mock_client() -> None:
    """Verify classify_message coordinates prompt building, client generation, and response parsing."""
    expected_response = '{"intent": "Playback & App Bugs", "confidence": 0.92, "reasoning": "Song skipping issue."}'
    client = MockOllamaClient(response_text=expected_response)

    result = classify_message("Songs keep pausing unexpectedly", client)

    assert client.call_count == 1
    assert "Songs keep pausing unexpectedly" in client.last_prompt
    assert result["intent"] == "Playback & App Bugs"
    assert result["confidence"] == 0.92
    assert result["reasoning"] == "Song skipping issue."


def test_classify_dataframe_populates_columns() -> None:
    """Verify classify_dataframe processes rows and adds predicted_intent, confidence, reasoning columns."""
    df = pd.DataFrame(
        {
            "tweet_id": ["101", "102", "103"],
            "text_clean": [
                "Charged twice this month",
                "App crashes on launch",
                "Thanks for the quick help!",
            ],
        }
    )

    client = MockOllamaClient(
        response_text='{"intent": "Praise/Off-topic", "confidence": 0.85, "reasoning": "Thank you message."}'
    )

    result_df = classify_dataframe(df, text_column="text_clean", client=client)

    assert len(result_df) == 3
    assert "predicted_intent" in result_df.columns
    assert "confidence" in result_df.columns
    assert "reasoning" in result_df.columns
    assert client.call_count == 3
    assert all(result_df["predicted_intent"] == "Praise/Off-topic")
    assert all(result_df["confidence"] == 0.85)
