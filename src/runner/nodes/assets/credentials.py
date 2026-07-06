from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit

from shared.db import database_session
from shared.db.settings import crud as settings_crud


_LOGGER = logging.getLogger(__name__)

# Env overrides win over the stored setting so CI / one-off runs can inject a token.
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
_HF_HOSTS = ("huggingface.co", "hf.co")


def huggingface_token() -> str | None:
    """Resolve the Hugging Face token from the environment, then the settings row."""
    for name in _HF_TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    try:
        with database_session() as session:
            token = settings_crud.get_integration_settings(session).hf_token
    except Exception:
        _LOGGER.warning("could not read Hugging Face token from settings", exc_info=True)
        return None
    token = token.strip()
    return token or None


def huggingface_auth_headers(url: str) -> dict[str, str]:
    """Bearer auth header for Hugging Face URLs; empty for any other host.

    The token only needs to reach ``huggingface.co``: it authorizes the request and
    returns a redirect to a pre-signed CDN URL that carries its own credentials.
    """
    if not _is_huggingface_url(url):
        return {}
    token = huggingface_token()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _is_huggingface_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == base or host.endswith(f".{base}") for base in _HF_HOSTS)
