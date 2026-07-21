from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from transformers import BertModel, BertTokenizerFast

from shared.db.assets import crud as asset_crud
from shared.db.connection import database_session

from ..config import BeetleConfig, config_fingerprint, load_config
from ..data import (
    DatabaseSegmentIndex,
    ValidationLoader,
    ValidationSource,
    select_validation_audio_ids,
)
from ..models.modules.aligner import PhonemeAligner
from ..models.modules.alignment_backbone import StyleTTSAlignerBackbone
from ..models.modules.audio import F0Extractor
from .checkpoint import (
    CheckpointManager,
    CheckpointPayload,
    validate_resume_fingerprints,
)
from .state import StageKind


class PreparationCallbacks(Protocol):
    def check_cancel(self) -> None: ...

    def report_index_progress(self, scanned: int, total: int) -> None: ...


@dataclass(frozen=True)
class PhonemeResources:
    model: BertModel
    tokenizer: BertTokenizerFast


@dataclass(frozen=True)
class TextResources:
    model: BertModel
    tokenizer: BertTokenizerFast


def load_phoneme_resources(model_path: Path) -> PhonemeResources:
    model = BertModel.from_pretrained(model_path, local_files_only=True)
    tokenizer = BertTokenizerFast.from_pretrained(
        model_path,
        local_files_only=True,
    )
    return PhonemeResources(model, tokenizer)


def load_text_resources(model_name: str) -> TextResources:
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    return TextResources(model, tokenizer)


def load_f0_extractor() -> F0Extractor:
    checkpoint = (
        Path(__file__).parents[1]
        / "external"
        / "StyleTTS2"
        / "Utils"
        / "JDC"
        / "bst.t7"
    )
    return F0Extractor.from_checkpoint(checkpoint)


def load_aligner(config: BeetleConfig) -> PhonemeAligner:
    settings = config.architecture.aligner
    with database_session() as session:
        folder = asset_crud.get_checkpoint_path(session, settings.checkpoint_asset_id)
    relative = Path(settings.checkpoint_filename)
    if relative.name != settings.checkpoint_filename:
        raise ValueError("aligner checkpoint filename must be one folder entry")
    checkpoint = folder / relative
    backbone = StyleTTSAlignerBackbone(
        input_channels=config.audio.mel_channels,
        hidden_channels=settings.hidden_channels,
        token_count=config.architecture.phoneme_token_count,
        layer_count=settings.layer_count,
        token_embedding_channels=settings.token_embedding_channels,
    )
    aligner = PhonemeAligner(
        backbone,
        settings,
        config.architecture.phoneme_token_count,
        settings.frame_reduction,
    )
    aligner.load_checkpoint(checkpoint)
    return aligner


@dataclass(frozen=True)
class RunPreparation:
    stage: StageKind
    config: BeetleConfig
    config_fingerprint: str
    index: DatabaseSegmentIndex
    validation: ValidationSource
    checkpoint_manager: CheckpointManager
    resume: CheckpointPayload | None


def load_validation_source(
    config: BeetleConfig,
    index: DatabaseSegmentIndex,
    stage_number: int,
    loader: ValidationLoader,
) -> ValidationSource:
    audio_file_ids = select_validation_audio_ids(
        index,
        stage_number,
        config.validation.sample_count,
        config.runtime.seed,
    )
    return loader.load_source(stage_number, audio_file_ids)


def prepare_run(
    stage: StageKind,
    config_path: Path,
    output_path: Path,
    resume_path: Path | None,
    callbacks: PreparationCallbacks,
) -> RunPreparation:
    config = load_config(config_path)
    fingerprint = config_fingerprint(config)
    callbacks.check_cancel()
    index = DatabaseSegmentIndex.build(
        config.data.selection,
        config.architecture.language.values,
        config.data.maximum_seconds,
        config.data.prefetch.page_size,
        callbacks,
    )
    stage_number = {
        StageKind.STAGE1: 1,
        StageKind.STAGE2: 2,
        StageKind.STAGE3: 3,
    }[stage]
    index.report.require(stage_number, config.data.sentence_probability)
    validation = load_validation_source(
        config,
        index,
        stage_number,
        ValidationLoader.from_database(config),
    )
    callbacks.check_cancel()
    manager = CheckpointManager(output_path / "checkpoints")
    resume = manager.load(resume_path) if resume_path is not None else None
    if resume is not None:
        validate_resume_fingerprints(
            resume,
            stage,
            fingerprint,
            index.fingerprint,
        )
    return RunPreparation(
        stage,
        config,
        fingerprint,
        index,
        validation,
        manager,
        resume,
    )
