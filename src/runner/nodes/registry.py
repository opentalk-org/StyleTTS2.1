from __future__ import annotations

from runflow.registry.node_registry import NodeRegistry
from runflow.registry.type_registry import TypeRegistry
from runner.nodes.asr import CanaryTranscribeNode, ParakeetTranscribeNode, WhisperTranscribeNode
from runner.nodes.audio_io import LoadBucketAudioNode, SaveAudioArtifactNode, SaveTranscriptNode
from runner.nodes.audio_processing import (
    CalculateAudioStatsNode,
    CutAudioBySegmentsNode,
    CutAudioBySpeakersNode,
    DeepFilterNetDenoiseNode,
    NormalizeLoudnessNode,
    SortformerDiarizationNode,
    VadDetectNode,
)
from runner.nodes.audio_sources import AllAudioSourceNode, DatasetAudioSourceNode, SelectedAudioSourceNode
from runner.nodes.dataset_writeback import AddAudioToDatasetNode, AssignVoiceNode, DeleteAudioRecordsNode, RemoveAudioFromDatasetNode
from runner.nodes.datatypes import register_runner_types
from runner.nodes.text_processing import PhonemizeTranscriptNode


def register_runner_nodes(registry: NodeRegistry) -> NodeRegistry:
    for node_cls in [
        SelectedAudioSourceNode,
        DatasetAudioSourceNode,
        AllAudioSourceNode,
        LoadBucketAudioNode,
        SaveTranscriptNode,
        SaveAudioArtifactNode,
        WhisperTranscribeNode,
        CanaryTranscribeNode,
        ParakeetTranscribeNode,
        PhonemizeTranscriptNode,
        VadDetectNode,
        CutAudioBySegmentsNode,
        SortformerDiarizationNode,
        CutAudioBySpeakersNode,
        DeepFilterNetDenoiseNode,
        NormalizeLoudnessNode,
        CalculateAudioStatsNode,
        AddAudioToDatasetNode,
        RemoveAudioFromDatasetNode,
        AssignVoiceNode,
        DeleteAudioRecordsNode,
    ]:
        registry.register(node_cls)
    return registry


def register_runner_types_for_ui(registry: TypeRegistry) -> TypeRegistry:
    return register_runner_types(registry)
