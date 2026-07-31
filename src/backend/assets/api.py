from pydantic import BaseModel, Field
from fastapi import APIRouter, status

from runner.nodes.text.runtime.symbols import (
    DEFAULT_STYLETTS_SYMBOLS,
    MODEL_BERT_STYLETTS_SYMBOLS,
)
from shared.db import database_session
from shared.db.assets import crud as asset_crud
from shared.db.assets.schemas import (
    ConfigCreate,
    ConfigRead,
    ConfigUpdate,
    ExtraFileCreate,
    FileAssetRead,
)


router = APIRouter(prefix="/assets")


class TextFileAssetCreate(BaseModel):
    name: str
    type_: str
    content: str
    metadata: dict = Field(default_factory=dict)


@router.get("/files", response_model=list[FileAssetRead])
async def list_file_assets(type_: str | None = None) -> list[FileAssetRead]:
    with database_session() as session:
        return [FileAssetRead.model_validate(item) for item in asset_crud.list_extra_files(session, _asset_type(type_))]


@router.post("/files/text", response_model=FileAssetRead, status_code=status.HTTP_201_CREATED)
async def create_text_file_asset(request: TextFileAssetCreate) -> FileAssetRead:
    metadata = dict(request.metadata)
    if _asset_type(request.type_) == "ood_text_set" and "line_count" not in metadata:
        metadata["line_count"] = len([line for line in request.content.splitlines() if line.strip()])
    with database_session() as session:
        item = asset_crud.create_extra_file(
            session,
            ExtraFileCreate(
                name=request.name,
                data=request.content.encode("utf-8"),
                type_=_asset_type(request.type_) or request.type_,
                metadata=metadata,
            ),
        )
        return FileAssetRead.model_validate(item)


@router.get("/configs", response_model=list[ConfigRead])
async def list_asset_configs(type_: str | None = None) -> list[ConfigRead]:
    with database_session() as session:
        canonical = _config_type(type_)
        if canonical == "phoneme_alphabet":
            _ensure_default_phoneme_alphabets(session)
        return [ConfigRead.model_validate(item) for item in asset_crud.list_configs(session, canonical)]


@router.post("/configs", response_model=ConfigRead, status_code=status.HTTP_201_CREATED)
async def create_asset_config(request: ConfigCreate) -> ConfigRead:
    with database_session() as session:
        item = asset_crud.create_config(session, request)
        return ConfigRead.model_validate(item)


def _asset_type(type_: str | None) -> str | None:
    if type_ is None:
        return None
    aliases = {
        "ood_text": "ood_text_set",
        "ood_text_set": "ood_text_set",
        "f0": "f0_model",
        "f0_model": "f0_model",
        "asr": "asr_bundle",
        "asr_bundle": "asr_bundle",
        "plbert": "plbert",
        "styletts2": "styletts2",
    }
    return aliases.get(type_.strip().lower(), type_)


def _config_type(type_: str | None) -> str | None:
    if type_ is None:
        return None
    aliases = {
        "alphabet": "phoneme_alphabet",
        "phoneme_alphabet": "phoneme_alphabet",
    }
    return aliases.get(type_.strip().lower(), type_)


def _ensure_default_phoneme_alphabets(session) -> None:
    existing = {
        str(item.metadata_["preset"]): item
        for item in asset_crud.list_configs(session, "phoneme_alphabet")
        if item.metadata_.get("builtin") and "preset" in item.metadata_
    }
    presets = (
        ("ipa", "StyleTTS2 IPA", DEFAULT_STYLETTS_SYMBOLS),
        (
            "model-bert-styletts2",
            "Model BERT + StyleTTS2 extras",
            MODEL_BERT_STYLETTS_SYMBOLS,
        ),
    )
    for preset, name, symbols in presets:
        metadata = {
            "builtin": True,
            "preset": preset,
            "symbols": [str(symbol) for symbol in symbols],
        }
        if preset in existing:
            item = existing[preset]
            if item.name != name or item.metadata_ != metadata:
                asset_crud.update_config(
                    session,
                    item.id,
                    ConfigUpdate(
                        name=name,
                        type_="phoneme_alphabet",
                        metadata=metadata,
                    ),
                )
        else:
            asset_crud.create_config(
                session,
                ConfigCreate(
                    name=name,
                    type_="phoneme_alphabet",
                    metadata=metadata,
                ),
            )
