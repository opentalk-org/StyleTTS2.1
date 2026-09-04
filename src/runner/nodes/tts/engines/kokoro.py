from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from runner.nodes.tts.engines.base import (
    EngineRuntime,
    require_checkpoint_dir,
    resolve_device,
)
from runner.nodes.tts.voices import Voice

KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE = 24000

# Kokoro derives its G2P pipeline from a single-letter language code that matches the
# first character of each voice id (a=American EN, b=British EN, e=Spanish, f=French,
# h=Hindi, i=Italian, j=Japanese, p=BR Portuguese, z=Mandarin).
_VALID_LANG_CODES = frozenset("abefhijpz")


class KokoroRuntime(EngineRuntime):
    SAMPLE_RATE = KOKORO_SAMPLE_RATE

    def __init__(self, model: Any):
        self._model = model
        self._pipelines: dict[str, Any] = {}

    def synthesize(
        self, text: str, voice: Voice, language: str
    ) -> tuple[np.ndarray, int]:
        voice_id = voice.require_preset()
        pipeline = self._pipeline_for(voice_id[0])
        chunks = [
            result.audio
            for result in pipeline(text, voice=voice_id, speed=1)
            if result.audio is not None
        ]
        if not chunks:
            raise RuntimeError(f"kokoro_empty_audio:{voice_id}")
        waveform = np.concatenate(
            [chunk.detach().cpu().numpy().reshape(-1) for chunk in chunks]
        ).astype(np.float32)
        return waveform, KOKORO_SAMPLE_RATE

    def _pipeline_for(self, lang_code: str):
        if lang_code not in _VALID_LANG_CODES:
            raise ValueError(f"kokoro_unknown_lang_code:{lang_code}")
        pipeline = self._pipelines.get(lang_code)
        if pipeline is None:
            from kokoro import KPipeline

            pipeline = KPipeline(
                lang_code=lang_code, repo_id=KOKORO_REPO_ID, model=self._model
            )
            self._pipelines[lang_code] = pipeline
        return pipeline


def load(checkpoint_dir: Path, device: str | None = None) -> KokoroRuntime:
    try:
        from kokoro import KModel
    except ImportError as exc:
        raise RuntimeError("kokoro_not_installed") from exc
    require_checkpoint_dir(checkpoint_dir)
    config = _first_existing(checkpoint_dir, ("config.json", "config.yml"))
    weights = _first_existing(
        checkpoint_dir, ("kokoro-v1_0.pth", "kokoro-v1.0.pth", "model.pth")
    )
    model = KModel(repo_id=KOKORO_REPO_ID, config=str(config), model=str(weights))
    model = model.to(device or resolve_device()).eval()
    return KokoroRuntime(model)


def _first_existing(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    matches = sorted(root.rglob(names[0]))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"kokoro_checkpoint_file_missing:{names}:{root}")
