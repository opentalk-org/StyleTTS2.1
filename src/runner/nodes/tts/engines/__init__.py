from __future__ import annotations

from pathlib import Path
from typing import Callable

from runner.nodes.tts.engines import chatterbox, dia, f5_tts, fish_speech, kokoro, orpheus, raon_opentts
from runner.nodes.tts.engines.base import EngineRuntime
from runner.nodes.tts.voices import TtsEngine

# Each engine module exposes ``load(checkpoint_dir, device) -> EngineRuntime``.
_LOADERS: dict[TtsEngine, Callable[[Path], EngineRuntime]] = {
    TtsEngine.KOKORO: kokoro.load,
    TtsEngine.CHATTERBOX: chatterbox.load,
    TtsEngine.F5_TTS: f5_tts.load,
    TtsEngine.ORPHEUS: orpheus.load,
    TtsEngine.DIA: dia.load,
    TtsEngine.FISH_SPEECH: fish_speech.load,
    TtsEngine.RAON_OPENTTS: raon_opentts.load,
}


def load_engine(engine: TtsEngine, checkpoint_dir: Path) -> EngineRuntime:
    return _LOADERS[engine](checkpoint_dir)
