from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import AlbertConfig, AlbertModel, BertModel, BertTokenizerFast

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
from ..external.StyleTTS2.text_utils import symbols
from .checkpoint import (
    CheckpointManager,
    CheckpointPayload,
    validate_resume_fingerprints,
)


class PreparationCallbacks(Protocol):
    def check_cancel(self) -> None: ...

    def report_index_progress(self, scanned: int, total: int) -> None: ...


@dataclass(frozen=True)
class PhonemeResources:
    model: AlbertModel
    tokenizer: "PhonemeTokenizer"


@dataclass(frozen=True)
class TextResources:
    model: BertModel
    tokenizer: BertTokenizerFast


class PhonemeTokenizer:
    def __init__(self) -> None:
        self._token_ids = {symbol: index for index, symbol in enumerate(symbols)}

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text).difference(self._token_ids))
        if unknown:
            raise ValueError(f"phonemes are outside the StyleTTS2 vocabulary: {unknown}")
        return [self._token_ids[symbol] for symbol in text]


def load_phoneme_resources(model_path: Path) -> PhonemeResources:
    settings = yaml.safe_load((model_path / "config.yml").read_text())
    model = AlbertModel(AlbertConfig(**settings["model_params"]))
    checkpoints = sorted(
        model_path.glob("step_*.t7"),
        key=lambda path: int(path.stem.removeprefix("step_")),
    )
    if not checkpoints:
        raise ValueError(f"PL-BERT checkpoint is missing from {model_path}")
    payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
    prefix = "module.encoder."
    state = {
        key.removeprefix(prefix): value
        for key, value in payload["net"].items()
        if key.startswith(prefix) and not key.endswith("position_ids")
    }
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"PL-BERT checkpoint keys do not match: {incompatible}")
    return PhonemeResources(model, PhonemeTokenizer())


def load_text_resources(model_name: str) -> TextResources:
    snapshot = Path(snapshot_download(model_name, local_files_only=True))
    model = BertModel.from_pretrained(snapshot)
    tokenizer = BertTokenizerFast.from_pretrained(snapshot)
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
    config: BeetleConfig
    config_fingerprint: str
    index: DatabaseSegmentIndex
    validation: ValidationSource
    checkpoint_manager: CheckpointManager
    resume: CheckpointPayload | None


def load_validation_source(
    config: BeetleConfig,
    index: DatabaseSegmentIndex,
    loader: ValidationLoader,
) -> ValidationSource:
    audio_file_ids = select_validation_audio_ids(
        index,
        config.validation.sample_count,
        config.runtime.seed,
        config.validation.audio_file_ids,
    )
    return loader.load_source(audio_file_ids)


def prepare_run(
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
    index.report.require()
    validation = load_validation_source(
        config,
        index,
        ValidationLoader(config),
    )
    callbacks.check_cancel()
    manager = CheckpointManager(
        output_path / "checkpoints",
        config.checkpoint.keep_last,
    )
    resume = manager.load(resume_path) if resume_path is not None else None
    if resume is not None:
        validate_resume_fingerprints(
            resume,
            fingerprint,
            index.fingerprint,
        )
    return RunPreparation(
        config,
        fingerprint,
        index,
        validation,
        manager,
        resume,
    )
