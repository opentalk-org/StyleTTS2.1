from __future__ import annotations

from runflow.registry.node_registry import NodeRegistry
from runflow.registry.type_registry import TypeRegistry
from runner.nodes.asr import ASR_SOURCE, ParakeetTranscribeNode, WhisperTranscribeNode
from runner.nodes.audio_io import LoadAudioNode, SaveAudioArtifactNode, SaveTranscriptNode
from runner.nodes.audio_processing import (
    CalculateAudioStatsNode,
    CutAudioBySegmentsNode,
    VadDetectNode,
)
from runner.nodes.statistics.aggregate import AggregateDatasetStatisticsNode
from runner.nodes.statistics.audio_features import AnalyzeAudioFeaturesNode
from runner.nodes.statistics.writeback import SaveStatisticsEntryNode
from runner.nodes.synthesis.style_reference import ResolveStyleReferenceNode
from runner.nodes.synthesis.styletts import StyleTtsSweepSynthesisNode, StyleTtsSynthesisNode
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
from runner.nodes.assets.catalog import CatalogDownloadNode
from runner.nodes.assets.checkpoints import ResolveCheckpointNode
from runner.nodes.assets.training_assets import ResolveTrainingAssetsNode
from runner.nodes.dataset_writeback import AddAudioToDatasetNode, AssignVoiceNode, DeleteAudioRecordsNode, RemoveAudioFromDatasetNode
from runner.nodes.datatypes import register_runner_types
from runner.nodes.text_processing import PhonemizeSegmentsNode, PhonemizeTranscriptNode
from runner.nodes.testing import (
    SelectStyleReferenceNode,
    StyleReferenceSweepNode,
    TestingPromptPhonemizerNode,
    TestingRunInputNode,
    TestingTextPromptNode,
)
from runner.nodes.training.common.inputs import (
    ListDatasetAudioIdsNode,
    PhonemeAlphabetNode,
    PrefetchCheckpointNode,
    PrefetchOodTextSetsNode,
    PrefetchTrainingAssetsNode,
    SelectCheckpointNode,
    SelectOodTextSetsNode,
    SelectTrainingAssetsNode,
    SelectTrainingDatasetNode,
    TrainingRunInputNode,
)
from runner.nodes.training.asr import AsrModelTrainingNode
from runner.nodes.training.common.manifest import TRAINING_SEGMENT_INPUT, BuildTrainingManifestNode
from runner.nodes.training.f0 import F0ModelTrainingNode
from runner.nodes.training.styletts import BuildStyleTtsFinetuneConfigNode, StyleTtsFinetuneNode


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
        ParakeetTranscribeNode,
        TranscriptToSegmentsNode,
        ApplyTranscriptToSegmentsNode,
        PhonemizeTranscriptNode,
        PhonemizeSegmentsNode,
        VadDetectNode,
        CutAudioBySegmentsNode,
        DeepFilterNetDenoiseNode,
        NormalizeLoudnessNode,
        AnalyzeAudioFeaturesNode,
        CalculateAudioStatsNode,
        AggregateDatasetStatisticsNode,
        SaveStatisticsEntryNode,
        PlanSegmentGroupsNode,
        ExtractSegmentGroupAudioNode,
        PersistSplitAudioRecordsNode,
        AddAudioToDatasetNode,
        RemoveAudioFromDatasetNode,
        AssignVoiceNode,
        DeleteAudioRecordsNode,
        CatalogDownloadNode,
        ResolveCheckpointNode,
        ResolveTrainingAssetsNode,
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
        BuildTrainingManifestNode,
        BuildStyleTtsFinetuneConfigNode,
        StyleTtsFinetuneNode,
        F0ModelTrainingNode,
        AsrModelTrainingNode,
        TestingRunInputNode,
        TestingTextPromptNode,
        SelectStyleReferenceNode,
        StyleReferenceSweepNode,
        ResolveStyleReferenceNode,
        TestingPromptPhonemizerNode,
        StyleTtsSynthesisNode,
        StyleTtsSweepSynthesisNode,
    ]:
        registry.register(node_cls)
    return registry


def create_node_registry() -> NodeRegistry:
    return register_runner_nodes(NodeRegistry())


def register_runner_types_for_ui(registry: TypeRegistry) -> TypeRegistry:
    register_runner_types(registry)
    registry.register(TRAINING_SEGMENT_INPUT)
    registry.register(ASR_SOURCE)
    return registry
