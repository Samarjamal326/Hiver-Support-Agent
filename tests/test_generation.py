"""Unit tests for grounded reply generation prompt construction and execution."""

import json
from typing import Any, Dict, List, Optional
import pytest

from hiver_agent.generation.draft import (
    FALLBACK_RESULT,
    MAX_EXEMPLARS,
    _parse_json_response,
    build_reply_prompt,
    draft_reply,
)


class MockOllamaClient:
    """Test double providing scripted responses for draft_reply tests."""

    def __init__(self, responses: Optional[List[str]] = None, single_response: str = "") -> None:
        if responses is not None:
            self.responses = list(responses)
        else:
            self.responses = [single_response]
        self.call_count = 0
        self.prompts_received: List[str] = []
        self.temperatures_received: List[float] = []

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self.prompts_received.append(prompt)
        self.temperatures_received.append(temperature)
        idx = min(self.call_count, len(self.responses) - 1)
        resp = self.responses[idx]
        self.call_count += 1
        return resp


def test_build_reply_prompt_includes_exemplars_and_antifabrication_guardrail() -> None:
    """Verify build_reply_prompt contains customer text, all exemplar texts, and anti-fabrication instructions."""
    customer_text = "I was charged $15.99 instead of $9.99 for my family plan."
    intent = "Billing & Subscription"
    exemplars = [
        {
            "customer_text": "Why did my monthly bill increase?",
            "reply_text": "Hey there! Please DM us your account email so we can investigate your billing details.",
            "reply_tweet_id": "rep_101",
            "similarity_score": 0.88,
        },
        {
            "customer_text": "Overcharged for premium subscription",
            "reply_text": "We'd love to help sort this out! Send us a direct message with your registered address.",
            "reply_tweet_id": "rep_102",
            "similarity_score": 0.85,
        },
    ]

    prompt = build_reply_prompt(customer_text, intent, exemplars)

    # Verify incoming text and intent are included
    assert customer_text in prompt
    assert intent in prompt

    # Verify all exemplar customer and reply texts are included
    for ex in exemplars:
        assert ex["customer_text"] in prompt
        assert ex["reply_text"] in prompt
        assert ex["reply_tweet_id"] in prompt

    # Verify anti-fabrication guardrail instructions
    assert "DO NOT invent specific refund amounts, dates, policy terms" in prompt
    assert "grounding_tweet_ids" in prompt


def test_draft_reply_succeeds_on_well_formed_synthetic_response() -> None:
    """Verify draft_reply parses valid JSON on the first attempt without retrying."""
    customer_text = "I cannot log in on desktop."
    intent = "Account Access/Login"
    exemplars = [
        {
            "customer_text": "Login error 403 on Mac",
            "reply_text": "Hey! Clear your cache or reinstall the desktop app. DM us if it persists.",
            "reply_tweet_id": "rep_201",
            "similarity_score": 0.91,
        }
    ]

    expected_output = {
        "reply_text": "Hey there! Try reinstalling the desktop app and clearing cache. DM us if you still need a hand!",
        "grounding_tweet_ids": ["rep_201"],
        "notes": "Follows standard desktop troubleshooting exemplar",
    }
    mock_client = MockOllamaClient(single_response=json.dumps(expected_output))

    result = draft_reply(customer_text, intent, exemplars, mock_client)

    assert mock_client.call_count == 1
    assert result["reply_text"] == expected_output["reply_text"]
    assert result["grounding_tweet_ids"] == ["rep_201"]
    assert result["notes"] == expected_output["notes"]


def test_draft_reply_handles_markdown_wrapped_json() -> None:
    """Verify draft_reply parses JSON wrapped in markdown code fences."""
    raw_payload = """```json
{
  "reply_text": "Please send us a DM with your username.",
  "grounding_tweet_ids": ["rep_301"],
  "notes": "Direct to DM"
}
```"""
    mock_client = MockOllamaClient(single_response=raw_payload)

    result = draft_reply(
        "Where do I send payment info?",
        "Billing & Subscription",
        [{"customer_text": "Payment?", "reply_text": "DM us", "reply_tweet_id": "rep_301", "similarity_score": 0.8}],
        mock_client,
    )

    assert mock_client.call_count == 1
    assert result["reply_text"] == "Please send us a DM with your username."
    assert result["grounding_tweet_ids"] == ["rep_301"]


def test_draft_reply_recovers_on_retry_after_first_malformed_json() -> None:
    """Verify draft_reply retries once when first attempt fails, and succeeds if retry returns valid JSON."""
    valid_payload = {
        "reply_text": "Hey! DM us your email address.",
        "grounding_tweet_ids": ["rep_401"],
        "notes": "Recovered on retry",
    }
    # First response is garbage text; second is valid JSON
    mock_client = MockOllamaClient(responses=["Here is your reply: Please DM us!", json.dumps(valid_payload)])

    result = draft_reply(
        "Need help with student discount",
        "Billing & Subscription",
        [{"customer_text": "Student discount issue", "reply_text": "DM email", "reply_tweet_id": "rep_401", "similarity_score": 0.85}],
        mock_client,
    )

    assert mock_client.call_count == 2
    assert result["reply_text"] == valid_payload["reply_text"]
    assert result["grounding_tweet_ids"] == ["rep_401"]
    assert result["notes"] == valid_payload["notes"]
    # Check that retry prompt contained stricter instruction
    assert "Your previous response was not valid JSON" in mock_client.prompts_received[1]


def test_draft_reply_malformed_json_triggers_documented_fallback_after_retry() -> None:
    """Verify malformed JSON on both initial call and retry returns FALLBACK_RESULT with notes='generation_failed'."""
    # Mock client returns garbage twice
    mock_client = MockOllamaClient(responses=["Sorry, I cannot produce JSON right now.", "Not JSON either!"])

    result = draft_reply(
        "App keeps crashing on launch",
        "Playback & App Bugs",
        [{"customer_text": "Crash on launch", "reply_text": "Reinstall app", "reply_tweet_id": "rep_501", "similarity_score": 0.89}],
        mock_client,
    )

    assert mock_client.call_count == 2
    assert result["reply_text"] == ""
    assert result["grounding_tweet_ids"] == []
    assert result["notes"] == "generation_failed"
    assert result == FALLBACK_RESULT
