from __future__ import annotations

from runflow.tmp_nodes.asr.canary import CanaryNode
from runflow.tmp_nodes.asr.parakeet import ParakeetNode
from runflow.tmp_nodes.asr.whisper import WhisperNode
from runflow.tmp_nodes.diarization.audio_cut_by_speakers import AudioCutBySpeakersNode
from runflow.tmp_nodes.diarization.sortformer import SortformerDiarizationNode
from runflow.tmp_nodes.enhancement.deepfilternet import DeepFilterNetNode
from runflow.tmp_nodes.io.directory_input import DirectoryInputNode
from runflow.tmp_nodes.io.load_audio import LoadAudioNode
from runflow.tmp_nodes.io.save_audio import SaveAudioNode, SaveTranscriptNode
from runflow.tmp_nodes.vad.audio_cut_by_segments import AudioCutBySegmentsNode
from runflow.tmp_nodes.vad.vad_detect import VADDetectNode
from runflow.registry.node_registry import NodeRegistry


def register_builtin_nodes(registry: NodeRegistry) -> NodeRegistry:
    for node_cls in [
        DirectoryInputNode,
        LoadAudioNode,
        VADDetectNode,
        AudioCutBySegmentsNode,
        SortformerDiarizationNode,
        AudioCutBySpeakersNode,
        DeepFilterNetNode,
        WhisperNode,
        ParakeetNode,
        CanaryNode,
        SaveAudioNode,
        SaveTranscriptNode,
    ]:
        registry.register(node_cls)
    return registry
