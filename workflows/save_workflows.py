#!/usr/bin/env python3
"""Save every workflow definition in this folder into the backend as a saved workflow.

Idempotent: a workflow whose ``name`` already exists in the backend is left
untouched, so this can be re-run at any time to seed a fresh backend without
creating duplicates.

Usage::

    python workflows/save_workflows.py

Set ``RUNFLOW_BACKEND_URL`` to point at a non-default backend (default matches
``BACKEND_HOST``/``BACKEND_PORT`` from ``nix/runflow-dev.sh``).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_BACKEND_URL = "http://127.0.0.1:8001"
WORKFLOWS_DIR = Path(__file__).parent


def _load_definitions() -> list[tuple[Path, dict]]:
    definitions = []
    for path in sorted(WORKFLOWS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("hidden", False)
        definitions.append((path, payload))
    return definitions


def _existing_names(backend_url: str) -> set[str]:
    request = urllib.request.Request(f"{backend_url}/workflows", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise SystemExit(f"Could not reach backend at {backend_url}: {error}") from error
    return {item["name"] for item in body}


def _save(backend_url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{backend_url}/workflows",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    backend_url = os.environ.get("RUNFLOW_BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")
    existing = _existing_names(backend_url)

    saved = 0
    skipped = 0
    for path, payload in _load_definitions():
        name = payload.get("name", path.stem)
        if name in existing:
            print(f"skip   {path.name}: already saved ({name!r})")
            skipped += 1
            continue
        try:
            body = _save(backend_url, payload)
        except urllib.error.URLError as error:
            raise SystemExit(f"Could not save {path.name} to {backend_url}: {error}") from error
        print(f"saved  {path.name}: {body['id']} ({body['name']!r})")
        existing.add(name)
        saved += 1

    print(f"\nDone: {saved} saved, {skipped} already present.")


if __name__ == "__main__":
    main()
