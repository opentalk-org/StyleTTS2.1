from __future__ import annotations

from runner.nodes.tts.synthesis import (
    ChatterboxSynthesisNode,
    DiaSynthesisNode,
    F5TtsSynthesisNode,
    FishSpeechSynthesisNode,
    KokoroSynthesisNode,
    OrpheusSynthesisNode,
    RaonOpenTtsSynthesisNode,
)
from runner.nodes.tts.voice_nodes import (
    ChatterboxCloneVoiceNode,
    DiaCloneVoiceNode,
    F5TtsCloneVoiceNode,
    FishSpeechCloneVoiceNode,
    OrpheusCloneVoiceNode,
    RaonOpenTtsCloneVoiceNode,
    TtsRandomVoicesNode,
    TtsSelectVoiceNode,
)

TTS_SYNTHESIS_NODES = [
    KokoroSynthesisNode,
    ChatterboxSynthesisNode,
    F5TtsSynthesisNode,
    OrpheusSynthesisNode,
    DiaSynthesisNode,
    FishSpeechSynthesisNode,
    RaonOpenTtsSynthesisNode,
]

TTS_VOICE_NODES = [
    TtsSelectVoiceNode,
    TtsRandomVoicesNode,
    ChatterboxCloneVoiceNode,
    F5TtsCloneVoiceNode,
    OrpheusCloneVoiceNode,
    DiaCloneVoiceNode,
    FishSpeechCloneVoiceNode,
    RaonOpenTtsCloneVoiceNode,
]

TTS_NODES = TTS_SYNTHESIS_NODES + TTS_VOICE_NODES

__all__ = [cls.__name__ for cls in TTS_NODES] + ["TTS_NODES", "TTS_SYNTHESIS_NODES", "TTS_VOICE_NODES"]
