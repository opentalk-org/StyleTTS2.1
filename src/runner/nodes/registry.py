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
from runner.nodes.audio_filters import FilterAudioLanguageNode, FilterProcessedAudioNode
from runner.nodes.audio_processing import (
    CutAudioBySegmentsNode,
    SortformerDiarizationNode,
    VadDetectNode,
)
from runner.nodes.statistics.aggregate import AggregateDatasetStatisticsNode
from runner.nodes.statistics.audio_features import AnalyzeAudioFeaturesNode
from runner.nodes.statistics.database_features import DatabaseStatisticsFeaturesNode
from runner.nodes.statistics.voice_embedding_plot import EmbedVoicesPcaPlotNode
from runner.nodes.statistics.writeback import SaveStatisticsEntryNode
from runner.nodes.synthesis.style_reference import ResolveStyleReferenceNode
from runner.nodes.synthesis.styletts import StyleTtsSynthesisNode
from runner.nodes.audio_enhancement.denoise import DeepFilterNetDenoiseNode
from runner.nodes.audio_enhancement.normalize import NormalizeLoudnessNode
from runner.nodes.audio_enhancement.pad_silence import PadSilenceNode
from runner.nodes.audio_segments.extract import (
    ExtractSegmentGroupAudioNode,
    PersistSplitAudioRecordsNode,
)
from runner.nodes.audio_segments.dedup_overlap import DeduplicateOverlappingSegmentsNode
from runner.nodes.audio_segments.grouping import PlanSegmentGroupsNode
from runner.nodes.audio_segments.merge_alignment import MergeAlignmentNode
from runner.nodes.audio_segments.silence_breaks import InsertSilenceBreaksNode
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
from runner.nodes.dataset_writeback import (
    AddAudioToDatasetNode,
    AssignSpeakerNode,
    DeleteAudioRecordsNode,
    RemoveAudioFromDatasetNode,
)
from runner.nodes.datatypes import register_runner_types
from runner.nodes.hetzner import (
    HetznerDsV1ParquetAudioSourceNode,
    HetznerDsV2SourceNode,
)
from runner.nodes.libritts import LibriTtsSourceNode
from runner.nodes.mos import MosModelTrainingNode, PredictMosScoreNode
from runner.nodes.smart_turn import SmartTurnPredictNode
from runner.nodes.speaker_clustering import (
    SpeakerSegmentSource,
)
from runner.nodes.youtube_source import YouTubeAudioSourceNode
from runner.nodes.text_generation import OpenRouterGenerateTextsNode
from runner.nodes.text_processing import PhonemizeSegmentsNode
from runner.nodes.tts.registration import TTS_NODES
from runner.nodes.testing import (
    SelectStyleReferenceNode,
    StyleReferenceSweepNode,
    TestingRunInputNode,
    TestingTextPromptNode,
    TestingTextSourceNode,
)
from runner.nodes.training.common.inputs import (
    PhonemeAlphabetNode,
    SelectCheckpointNode,
    SelectTrainingAssetsNode,
    SelectTrainingDatasetNode,
    TrainingRunInputNode,
)
from runner.nodes.training.asr import AsrModelTrainingNode
from runner.nodes.training.f0 import F0ModelTrainingNode
from runner.nodes.training.styletts import StyleTtsFinetuneNode
from runner.nodes.training.styletts3 import StyleTts3TrainingNode


def register_runner_nodes(registry: NodeRegistry) -> NodeRegistry:
    for node_cls in [
        AudioSourceNode,
        FilterAudioLanguageNode,
        FilterProcessedAudioNode,
        HetznerDsV1ParquetAudioSourceNode,
        HetznerDsV2SourceNode,
        LibriTtsSourceNode,
        YouTubeAudioSourceNode,
        LoadAudioNode,
        SaveAudioRecordNode,
        UpdateAudioRecordBytesNode,
        SaveAudioArtifactNode,
        LoadAudioSegmentsNode,
        SaveAudioSegmentsNode,
        InsertSilenceBreaksNode,
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
        PadSilenceNode,
        PredictMosScoreNode,
        SmartTurnPredictNode,
        SpeakerSegmentSource,
        AnalyzeAudioFeaturesNode,
        DatabaseStatisticsFeaturesNode,
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
        AssignSpeakerNode,
        DeleteAudioRecordsNode,
        CatalogDownloadNode,
        ResolveCheckpointNode,
        ResolveTrainingAssetsNode,
        TrainingRunInputNode,
        SelectTrainingDatasetNode,
        SelectCheckpointNode,
        SelectTrainingAssetsNode,
        PhonemeAlphabetNode,
        StyleTtsFinetuneNode,
        StyleTts3TrainingNode,
        F0ModelTrainingNode,
        AsrModelTrainingNode,
        MosModelTrainingNode,
        TestingRunInputNode,
        TestingTextPromptNode,
        TestingTextSourceNode,
        SelectStyleReferenceNode,
        StyleReferenceSweepNode,
        ResolveStyleReferenceNode,
        StyleTtsSynthesisNode,
        OpenRouterGenerateTextsNode,
        *TTS_NODES,
    ]:
        registry.register(node_cls)
    return registry


def register_runner_types_for_ui(registry: TypeRegistry) -> TypeRegistry:
    register_runner_types(registry)
    return registry
