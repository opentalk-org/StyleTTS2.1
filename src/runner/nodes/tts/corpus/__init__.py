from runner.nodes.tts.corpus.models import (
    CorpusJob,
    CorpusPlan,
    PiperModelPlan,
)
from runner.nodes.tts.corpus.kokoro import KokoroCorpusSynthesisNode
from runner.nodes.tts.corpus.other import OtherTtsCorpusSynthesisNode
from runner.nodes.tts.corpus.piper import PiperCorpusSynthesisNode
from runner.nodes.tts.corpus.plan import build_corpus_plan, without_completed

TTS_CORPUS_NODES = [
    PiperCorpusSynthesisNode,
    KokoroCorpusSynthesisNode,
    OtherTtsCorpusSynthesisNode,
]

__all__ = [
    "CorpusJob",
    "CorpusPlan",
    "PiperModelPlan",
    "PiperCorpusSynthesisNode",
    "KokoroCorpusSynthesisNode",
    "OtherTtsCorpusSynthesisNode",
    "TTS_CORPUS_NODES",
    "build_corpus_plan",
    "without_completed",
]
