from __future__ import annotations

from runflow.registry.node_registry import NodeRegistry
from runflow.registry.type_registry import TypeRegistry
from runner.nodes.asr import CanaryTranscribeNode, ParakeetTranscribeNode, WhisperTranscribeNode
from runner.nodes.audio_io import LoadAudioNode, SaveAudioArtifactNode, SaveTranscriptNode
from runner.nodes.audio_processing import (
    CalculateAudioStatsNode,
    CutAudioBySegmentsNode,
    CutAudioBySpeakersNode,
    SortformerDiarizationNode,
    VadDetectNode,
)
from runner.nodes.audio_enhancement.denoise import DeepFilterNetDenoiseNode
from runner.nodes.audio_enhancement.normalize import NormalizeLoudnessNode
from runner.nodes.audio_segments.extract import ExtractSegmentGroupAudioNode, PersistSplitAudioRecordsNode
from runner.nodes.audio_segments.grouping import PlanSegmentGroupsNode
from runner.nodes.audio_segments.writeback import (
    LoadAudioSegmentsNode,
    SaveAudioRecordNode,
    SaveAudioSegmentsNode,
    UpdateAudioRecordBytesNode,
    UpdateSegmentPhonemesNode,
    UpdateSegmentTextNode,
)
from runner.nodes.audio_segments.transcripts import ApplyTranscriptToSegmentsNode, TranscriptToSegmentsNode
from runner.nodes.audio_sources import AllAudioSourceNode, DatasetAudioSourceNode, SelectedAudioSourceNode
from runner.nodes.dataset_writeback import AddAudioToDatasetNode, AssignVoiceNode, DeleteAudioRecordsNode, RemoveAudioFromDatasetNode
from runner.nodes.datatypes import register_runner_types
from runner.nodes.text_processing import PhonemizeSegmentsNode, PhonemizeTranscriptNode
from runner.nodes.testing import (
    SelectStyleReferenceNode,
    StyleReferenceSweepNode,
    StyleTtsSweepSynthesisNode,
    StyleTtsSynthesisNode,
    TestingPromptPhonemizerNode,
    TestingRunInputNode,
    TestingTextPromptNode,
)
from runner.nodes.training import (
    AsrModelTrainingNode,
    F0ModelTrainingNode,
    ListDatasetAudioIdsNode,
    PhonemeAlphabetNode,
    PrefetchCheckpointNode,
    PrefetchOodTextSetsNode,
    PrefetchTrainingAssetsNode,
    SelectCheckpointNode,
    SelectOodTextSetsNode,
    SelectTrainingAssetsNode,
    SelectTrainingDatasetNode,
    StyleTtsFinetuneNode,
    TrainingRunInputNode,
)


def register_runner_nodes(registry: NodeRegistry) -> NodeRegistry:
    for node_cls in [
        SelectedAudioSourceNode,
        DatasetAudioSourceNode,
        AllAudioSourceNode,
        LoadAudioNode,
        SaveAudioRecordNode,
        UpdateAudioRecordBytesNode,
        SaveTranscriptNode,
        SaveAudioArtifactNode,
        LoadAudioSegmentsNode,
        SaveAudioSegmentsNode,
        UpdateSegmentTextNode,
        UpdateSegmentPhonemesNode,
        WhisperTranscribeNode,
        CanaryTranscribeNode,
        ParakeetTranscribeNode,
        TranscriptToSegmentsNode,
        ApplyTranscriptToSegmentsNode,
        PhonemizeTranscriptNode,
        PhonemizeSegmentsNode,
        VadDetectNode,
        CutAudioBySegmentsNode,
        SortformerDiarizationNode,
        CutAudioBySpeakersNode,
        DeepFilterNetDenoiseNode,
        NormalizeLoudnessNode,
        CalculateAudioStatsNode,
        PlanSegmentGroupsNode,
        ExtractSegmentGroupAudioNode,
        PersistSplitAudioRecordsNode,
        AddAudioToDatasetNode,
        RemoveAudioFromDatasetNode,
        AssignVoiceNode,
        DeleteAudioRecordsNode,
        TrainingRunInputNode,
        SelectTrainingDatasetNode,
        SelectCheckpointNode,
        SelectTrainingAssetsNode,
        PhonemeAlphabetNode,
        SelectOodTextSetsNode,
        ListDatasetAudioIdsNode,
        PrefetchCheckpointNode,
        PrefetchTrainingAssetsNode,
        PrefetchOodTextSetsNode,
        StyleTtsFinetuneNode,
        F0ModelTrainingNode,
        AsrModelTrainingNode,
        TestingRunInputNode,
        TestingTextPromptNode,
        SelectStyleReferenceNode,
        StyleReferenceSweepNode,
        TestingPromptPhonemizerNode,
        StyleTtsSynthesisNode,
        StyleTtsSweepSynthesisNode,
    ]:
        registry.register(node_cls)
    return registry


def create_node_registry() -> NodeRegistry:
    return register_runner_nodes(NodeRegistry())


def register_runner_types_for_ui(registry: TypeRegistry) -> TypeRegistry:
    return register_runner_types(registry)
