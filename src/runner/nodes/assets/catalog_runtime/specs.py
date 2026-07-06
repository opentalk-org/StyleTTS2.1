from __future__ import annotations

from pathlib import Path
from typing import Any

from runner.nodes.assets.catalog_runtime.types import CatalogFile, CheckpointSpec, CheckpointType, ExtraFileSpec, ExtraFileType
from runner.nodes.assets.catalog_runtime.validation import (
    asr_bundle_valid,
    default_symbols_metadata,
    f0_bundle_valid,
    official_styletts_bundle_valid,
    plbert_bundle_valid,
    styletts_checkpoint_metadata,
)


STYLETTS2_UTILS_RAW_BASE = "https://raw.githubusercontent.com/yl4579/StyleTTS2/main/Utils"
STYLETTS2_REPO_RAW_MAIN = "https://raw.githubusercontent.com/yl4579/StyleTTS2/main"
STYLETTS2_OOD_TEXTS_URL = f"{STYLETTS2_REPO_RAW_MAIN}/Data/OOD_texts.txt"

LJSPEECH_PTH_URL = "https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/Models/LJSpeech/epoch_2nd_00100.pth"
LJSPEECH_CONFIG_URL = "https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/Models/LJSpeech/config.yml"
LIBRITTS_PTH_URL = "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth"
LIBRITTS_CONFIG_URL = "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/config.yml"
VOKAN_PTH_URL = "https://huggingface.co/ShoukanLabs/Vokan/resolve/main/Model/epoch_2nd_00012.pth"
VOKAN_CONFIG_URL = "https://huggingface.co/ShoukanLabs/Vokan/resolve/main/Model/config.yml"

PAPERCUP_CONFIG_URL = "https://huggingface.co/papercup-ai/multilingual-pl-bert/resolve/main/config.yml"
PAPERCUP_WEIGHT_URL = "https://huggingface.co/papercup-ai/multilingual-pl-bert/resolve/main/step_1100000.t7"


def styletts2_utils_specs() -> tuple[CheckpointSpec, CheckpointSpec, CheckpointSpec, ExtraFileSpec]:
    return (
        CheckpointSpec(
            key="styletts2_utils_asr",
            name="StyleTTS2 Utils ASR",
            type_=CheckpointType.ASR_BUNDLE,
            files=_github_files(("ASR/config.yml", "config.yml"), ("ASR/epoch_00080.pth", "epoch_00080.pth")),
            metadata=_metadata("styletts2_github", {"bundle": "asr"}, default_symbols_metadata()),
            is_valid=asr_bundle_valid,
            metadata_from_path=_empty_path_metadata,
        ),
        CheckpointSpec(
            key="styletts2_utils_f0",
            name="StyleTTS2 Utils F0 (JDC)",
            type_=CheckpointType.F0_MODEL,
            files=_github_files(("JDC/bst.t7", "bst.t7")),
            metadata=_metadata("styletts2_github", {"bundle": "f0"}, {}),
            is_valid=f0_bundle_valid,
            metadata_from_path=_empty_path_metadata,
        ),
        CheckpointSpec(
            key="styletts2_utils_plbert",
            name="StyleTTS2 Utils PL-BERT",
            type_=CheckpointType.PLBERT,
            files=_github_files(("PLBERT/config.yml", "config.yml"), ("PLBERT/step_1000000.t7", "step_1000000.t7")),
            metadata=_metadata("styletts2_github", {"bundle": "plbert"}, default_symbols_metadata()),
            is_valid=plbert_bundle_valid,
            metadata_from_path=_empty_path_metadata,
        ),
        ExtraFileSpec(
            key="styletts2_libritts_ood",
            name="LibriTTS OOD",
            type_=ExtraFileType.OOD_TEXT_SET,
            url=STYLETTS2_OOD_TEXTS_URL,
            metadata=_metadata("styletts2_github", {"bundle": "libritts_ood"}, {}),
        ),
    )


def official_styletts_specs() -> tuple[CheckpointSpec, CheckpointSpec]:
    return (
        _official_spec(
            key="official_styletts2_ljspeech",
            name="StyleTTS2 LJSpeech 24 kHz (Hugging Face)",
            repo="yl4579/StyleTTS2-LJSpeech",
            checkpoint_url=LJSPEECH_PTH_URL,
            config_url=LJSPEECH_CONFIG_URL,
            filename="epoch_2nd_00100.pth",
        ),
        _official_spec(
            key="official_styletts2_libritts",
            name="StyleTTS2 LibriTTS (Hugging Face)",
            repo="yl4579/StyleTTS2-LibriTTS",
            checkpoint_url=LIBRITTS_PTH_URL,
            config_url=LIBRITTS_CONFIG_URL,
            filename="epochs_2nd_00020.pth",
        ),
    )


def papercup_plbert_spec() -> CheckpointSpec:
    return CheckpointSpec(
        key="papercup_multilingual_pl_bert",
        name="Papercup multilingual PL-BERT (Hugging Face)",
        type_=CheckpointType.PLBERT,
        files=(CatalogFile(PAPERCUP_CONFIG_URL, "config.yml"), CatalogFile(PAPERCUP_WEIGHT_URL, "step_1100000.t7")),
        metadata=_metadata("papercup_hf_multilingual", {}, default_symbols_metadata()),
        is_valid=plbert_bundle_valid,
        metadata_from_path=_empty_path_metadata,
    )


def vokan_styletts_spec() -> CheckpointSpec:
    return _official_spec(
        key="vokan_styletts2",
        name="Vokan (ShoukanLabs, Hugging Face)",
        repo="ShoukanLabs/Vokan",
        checkpoint_url=VOKAN_PTH_URL,
        config_url=VOKAN_CONFIG_URL,
        filename="epoch_2nd_00012.pth",
    )


def _github_files(*pairs: tuple[str, str]) -> tuple[CatalogFile, ...]:
    return tuple(CatalogFile(f"{STYLETTS2_UTILS_RAW_BASE}/{suffix}", name) for suffix, name in pairs)


def _official_spec(
    *,
    key: str,
    name: str,
    repo: str,
    checkpoint_url: str,
    config_url: str,
    filename: str,
) -> CheckpointSpec:
    return CheckpointSpec(
        key=key,
        name=name,
        type_=CheckpointType.STYLETTS2,
        files=(CatalogFile(checkpoint_url, filename), CatalogFile(config_url, "config.yml")),
        metadata=_metadata("huggingface", {"repo": repo, "url": checkpoint_url, "config_url": config_url}, {}),
        is_valid=official_styletts_bundle_valid,
        metadata_from_path=_styletts_metadata_from_path,
    )


def _metadata(source: str, state: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    return {"source": source, "state": state, **extra}


def _empty_path_metadata(_path: Path) -> dict[str, Any]:
    return {}


def _styletts_metadata_from_path(path: Path) -> dict[str, Any]:
    return styletts_checkpoint_metadata(path / "config.yml")
