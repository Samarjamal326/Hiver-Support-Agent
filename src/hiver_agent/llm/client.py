"""Backend-agnostic LLM client interface with disk caching for Ollama."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import requests

logger = logging.getLogger(__name__)


class OllamaClientError(Exception):
    """Base exception for Ollama client failures."""


class OllamaConnectionError(OllamaClientError):
    """Raised when connecting to Ollama fails (e.g. connection refused, timeout)."""


class OllamaAPIError(OllamaClientError):
    """Raised when Ollama returns an error status code or invalid response body."""


class OllamaClient:
    """Ollama HTTP client with deterministic local disk caching."""

    def __init__(
        self,
        base_url: str,
        model: str,
        cache_dir: Union[str, Path] = "cache/llm_responses",
        timeout: float = 60.0,
    ) -> None:
        """Initialize OllamaClient with endpoint URL, target model, and response cache directory."""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_cache_key(self, prompt: str, temperature: float) -> str:
        """Compute SHA256 hex digest key based on model, prompt, and temperature."""
        raw_key = f"{self.model}:{prompt}:{temperature}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate response text from Ollama or return cached result if previously computed."""
        cache_key = self._compute_cache_key(prompt, temperature)
        cache_file = self.cache_dir / f"{cache_key}.json"

        # Check disk cache first
        if cache_file.exists():
            try:
                cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
                logger.debug("Cache hit for key %s", cache_key)
                return str(cached_data.get("response", ""))
            except Exception as exc:
                logger.warning("Failed to read cache file %s: %s; refetching", cache_file, exc)

        # Call Ollama HTTP API
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            raise OllamaConnectionError(
                f"Failed to connect to Ollama server at '{self.base_url}'. "
                "Ensure Ollama is running (`ollama serve`) and accessible."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaClientError(
                f"Unexpected network error while querying Ollama: {exc}"
            ) from exc

        if response.status_code != 200:
            raise OllamaAPIError(
                f"Ollama returned HTTP status {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
            response_text = str(data.get("response", ""))
        except Exception as exc:
            raise OllamaAPIError(
                f"Failed to parse JSON response from Ollama: {exc}"
            ) from exc

        # Persist to disk cache
        cache_payload: Dict[str, Any] = {
            "prompt": prompt,
            "response": response_text,
            "model": self.model,
        }
        try:
            cache_file.write_text(
                json.dumps(cache_payload, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("Failed to write response cache to %s: %s", cache_file, exc)

        return response_text
