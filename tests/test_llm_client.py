"""Unit tests for OllamaClient caching and error handling."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import requests

from hiver_agent.llm.client import (
    OllamaAPIError,
    OllamaClient,
    OllamaConnectionError,
)


def test_ollama_client_cache_read_and_write(tmp_path: Path) -> None:
    """Verify that generate caches responses and skips network calls on subsequent requests."""
    client = OllamaClient(
        base_url="http://localhost:11434",
        model="qwen2.5:3b-instruct",
        cache_dir=tmp_path / "cache",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": '{"intent": "Account Access/Login", "confidence": 0.95}'
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        # First call: cache miss, triggers HTTP call
        res1 = client.generate("Help resetting my password")
        assert res1 == '{"intent": "Account Access/Login", "confidence": 0.95}'
        assert mock_post.call_count == 1

        # Verify cache file was written to disk
        cache_files = list((tmp_path / "cache").glob("*.json"))
        assert len(cache_files) == 1
        cached_content = json.loads(cache_files[0].read_text(encoding="utf-8"))
        assert cached_content["prompt"] == "Help resetting my password"
        assert cached_content["response"] == res1
        assert cached_content["model"] == "qwen2.5:3b-instruct"

        # Second call with same prompt and temperature: cache hit, no network call
        res2 = client.generate("Help resetting my password")
        assert res2 == res1
        assert mock_post.call_count == 1  # call_count unchanged!


def test_ollama_client_different_temperature_uses_distinct_cache_key(
    tmp_path: Path,
) -> None:
    """Verify that different temperatures produce distinct cache entries."""
    client = OllamaClient(
        base_url="http://localhost:11434",
        model="qwen2.5:3b-instruct",
        cache_dir=tmp_path / "cache",
    )

    mock_resp1 = MagicMock(status_code=200)
    mock_resp1.json.return_value = {"response": "response_temp_0"}
    mock_resp2 = MagicMock(status_code=200)
    mock_resp2.json.return_value = {"response": "response_temp_0.7"}

    with patch("requests.post", side_effect=[mock_resp1, mock_resp2]) as mock_post:
        res1 = client.generate("Hello", temperature=0.0)
        res2 = client.generate("Hello", temperature=0.7)

        assert res1 == "response_temp_0"
        assert res2 == "response_temp_0.7"
        assert mock_post.call_count == 2
        assert len(list((tmp_path / "cache").glob("*.json"))) == 2


def test_ollama_client_raises_connection_error_on_refused_connection(
    tmp_path: Path,
) -> None:
    """Verify that network connection failures raise OllamaConnectionError."""
    client = OllamaClient(
        base_url="http://non-existent-host:11434",
        model="qwen2.5:3b-instruct",
        cache_dir=tmp_path / "cache",
    )

    with patch(
        "requests.post",
        side_effect=requests.exceptions.ConnectionError("Connection refused"),
    ):
        with pytest.raises(OllamaConnectionError) as exc_info:
            client.generate("test prompt")
        assert "Failed to connect to Ollama server" in str(exc_info.value)


def test_ollama_client_raises_api_error_on_non_200_response(
    tmp_path: Path,
) -> None:
    """Verify that non-200 HTTP responses raise OllamaAPIError."""
    client = OllamaClient(
        base_url="http://localhost:11434",
        model="non-existent-model",
        cache_dir=tmp_path / "cache",
    )

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = '{"error": "model not found"}'

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(OllamaAPIError) as exc_info:
            client.generate("test prompt")
        assert "HTTP status 404" in str(exc_info.value)
