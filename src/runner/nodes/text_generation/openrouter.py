from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# json_schema forcing a flat list of sentence objects. ``strict`` keeps providers that
# support structured outputs from drifting off-shape.
SENTENCES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }
    },
    "required": ["sentences"],
}


def request_sentences(
    *,
    token: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    timeout: float = 180.0,
) -> list[str]:
    """Call OpenRouter with a forced json_schema and return the list of sentence texts."""
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "tts_sentences", "strict": True, "schema": SENTENCES_SCHEMA},
        },
    }
    request = urllib.request.Request(
        OPENROUTER_CHAT_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://runflow.local",
            "X-Title": "runflow TTS text generation",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"openrouter_http_{error.code}:{detail}") from error
    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return [item["text"].strip() for item in parsed["sentences"] if item["text"].strip()]
