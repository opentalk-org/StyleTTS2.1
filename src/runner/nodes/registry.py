from __future__ import annotations

from runflow.registry.node_registry import NodeRegistry
from runflow.registry.type_registry import TypeRegistry
from runner.nodes.asr import (
    CanaryTranscribeNode,
    ParakeetTranscribeNode,
    WhisperTranscribeNode,
    WhisperXAlignNode,
)
from runner.nodes.audio_io import LoadAudioNode, SaveAudioArtifactNode
from runner.nodes.audio_processing import (
    CutAudioBySegmentsNode,
    SortformerDiarizationNode,
    VadDetectNode,
)
from runner.nodes.statistics.aggregate import AggregateDatasetStatisticsNode
from runner.nodes.statistics.audio_features import AnalyzeAudioFeaturesNode
from runner.nodes.statistics.voice_embedding_plot import EmbedVoicesPcaPlotNode
from runner.nodes.statistics.writeback import SaveStatisticsEntryNode
from runner.nodes.synthesis.style_reference import ResolveStyleReferenceNode
from runner.nodes.synthesis.styletts import StyleTtsSynthesisNode
from runner.nodes.audio_enhancement.denoise import DeepFilterNetDenoiseNode
from runner.nodes.audio_enhancement.normalize import NormalizeLoudnessNode
from runner.nodes.audio_segments.extract import ExtractSegmentGroupAudioNode, PersistSplitAudioRecordsNode
from runner.nodes.audio_segments.dedup_overlap import DeduplicateOverlappingSegmentsNode
from runner.nodes.audio_segments.grouping import PlanSegmentGroupsNode
from runner.nodes.audio_segments.merge_alignment import MergeAlignmentNode
from runner.nodes.audio_segments.speaker_split import DiarizeSplitSpeakersNode
from runner.nodes.audio_segments.writeback import (
    LoadAudioSegmentsNode,
    SaveAudioRecordNode,
    SaveAudioSegmentsNode,
    UpdateAudioRecordBytesNode,
)
from runner.nodes.audio_sources import AudioSourceNode
from runner.nodes.assets.catalog import CatalogDownloadNode
from runner.nodes.assets.checkpoints import ResolveCheckpointNode
from runner.nodes.assets.training_assets import ResolveTrainingAssetsNode
from runner.nodes.dataset_writeback import AddAudioToDatasetNode, AssignVoiceNode, DeleteAudioRecordsNode, RemoveAudioFromDatasetNode
from runner.nodes.datatypes import register_runner_types
from runner.nodes.hetzner import HetznerDsV1ParquetAudioSourceNode, HetznerDsV2ParquetAudioSourceNode
from runner.nodes.youtube_source import YouTubeAudioSourceNode
from runner.nodes.text_generation import OpenRouterGenerateTextsNode
from runner.nodes.text_processing import PhonemizeSegmentsNode
from runner.nodes.tts import TTS_NODES
from runner.nodes.testing import (
    SelectStyleReferenceNode,
    StyleReferenceSweepNode,
    TestingRunInputNode,
    TestingTextPromptNode,
)
from runner.nodes.training.common.inputs import (
    ListDatasetAudioIdsNode,
    PhonemeAlphabetNode,
    SelectCheckpointNode,
    SelectTrainingAssetsNode,
    SelectTrainingDatasetNode,
    TrainingRunInputNode,
)
from runner.nodes.training.asr import AsrModelTrainingNode
from runner.nodes.training.common.manifest import BuildTrainingManifestNode
from runner.nodes.training.f0 import F0ModelTrainingNode
from runner.nodes.training.styletts import StyleTtsFinetuneNode


def register_runner_nodes(registry: NodeRegistry) -> NodeRegistry:
    for node_cls in [
        AudioSourceNode,
        HetznerDsV1ParquetAudioSourceNode,
        HetznerDsV2ParquetAudioSourceNode,
        YouTubeAudioSourceNode,
        LoadAudioNode,
        SaveAudioRecordNode,
        UpdateAudioRecordBytesNode,
        SaveAudioArtifactNode,
        LoadAudioSegmentsNode,
        SaveAudioSegmentsNode,
        WhisperTranscribeNode,
        ParakeetTranscribeNode,
        CanaryTranscribeNode,
        WhisperXAlignNode,
        PhonemizeSegmentsNode,
        VadDetectNode,
        CutAudioBySegmentsNode,
        SortformerDiarizationNode,
        DeepFilterNetDenoiseNode,
        NormalizeLoudnessNode,
        AnalyzeAudioFeaturesNode,
        EmbedVoicesPcaPlotNode,
        AggregateDatasetStatisticsNode,
        SaveStatisticsEntryNode,
        PlanSegmentGroupsNode,
        DeduplicateOverlappingSegmentsNode,
        MergeAlignmentNode,
        DiarizeSplitSpeakersNode,
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
        ListDatasetAudioIdsNode,
        BuildTrainingManifestNode,
        StyleTtsFinetuneNode,
        F0ModelTrainingNode,
        AsrModelTrainingNode,
        TestingRunInputNode,
        TestingTextPromptNode,
        SelectStyleReferenceNode,
        StyleReferenceSweepNode,
        ResolveStyleReferenceNode,
        StyleTtsSynthesisNode,
        OpenRouterGenerateTextsNode,
        *TTS_NODES,
    ]:
        registry.register(node_cls)
    return registry


def create_node_registry() -> NodeRegistry:
    return register_runner_nodes(NodeRegistry())


def register_runner_types_for_ui(registry: TypeRegistry) -> TypeRegistry:
    register_runner_types(registry)
    return registry
