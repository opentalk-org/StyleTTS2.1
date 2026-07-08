from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import ASSET_BUNDLE
from runner.nodes.models import AssetBundleRef, stable_id
from shared.db import database_session
from shared.db.assets import crud as asset_crud


class ResolveTrainingAssetsSettings(StrictSettings):
    asr_bundle_file_ids: list[str] = Field(default_factory=list, title="ASR bundle files")
    f0_model_file_id: str = Field(default="", title="F0 model")
    plbert_file_id: str = Field(default="", title="PL-BERT")
    ood_text_set_file_ids: list[str] = Field(default_factory=list, title="OOD text sets")


@dataclass(frozen=True)
class AssetFileRef:
    role: str
    file_id: UUID | None
    checkpoint_id: UUID | None
    name: str
    type_: str
    path: Path
    size: int
    content_hash: str
    metadata: dict[str, Any]


def resolve_training_asset_bundle(
    asr_bundle_file_ids: list[str],
    f0_model_file_id: str,
    plbert_file_id: str,
    ood_text_set_file_ids: list[str],
) -> AssetBundleRef:
    requested = _requested_assets(asr_bundle_file_ids, f0_model_file_id, plbert_file_id, ood_text_set_file_ids)
    refs = [_resolve_asset(role, file_id) for role, file_id in requested]
    extra_file_ids = [ref.file_id for ref in refs if ref.file_id is not None]
    checkpoint_ids = [ref.checkpoint_id for ref in refs if ref.checkpoint_id is not None]
    digest_parts = [str(ref.file_id or ref.checkpoint_id) for ref in refs]
    ref_id = stable_id("assets", *digest_parts) if digest_parts else stable_id("assets", "empty")
    return AssetBundleRef(
        bundle_key="training_assets",
        name="Training assets",
        paths=[ref.path for ref in refs],
        id=ref_id,
        lineage_id=ref_id,
        extra_file_ids=extra_file_ids,
        metadata={
            "assets": [_asset_metadata(ref) for ref in refs],
            "roles": {
                "asr_bundle": [str(file_id) for file_id in asr_bundle_file_ids],
                "f0_model": f0_model_file_id,
                "plbert": plbert_file_id,
                "ood_text_sets": [str(file_id) for file_id in ood_text_set_file_ids],
            },
            "checkpoint_ids": [str(checkpoint_id) for checkpoint_id in checkpoint_ids],
        },
    )


def _requested_assets(
    asr_bundle_file_ids: list[str],
    f0_model_file_id: str,
    plbert_file_id: str,
    ood_text_set_file_ids: list[str],
) -> list[tuple[str, UUID]]:
    requested = [("asr_bundle", UUID(file_id)) for file_id in asr_bundle_file_ids]
    if f0_model_file_id:
        requested.append(("f0_model", UUID(f0_model_file_id)))
    if plbert_file_id:
        requested.append(("plbert", UUID(plbert_file_id)))
    requested.extend(("ood_text_set", UUID(file_id)) for file_id in ood_text_set_file_ids)
    return requested


def _resolve_asset(role: str, file_id: UUID) -> AssetFileRef:
    with database_session() as session:
        try:
            extra_file = asset_crud.get_extra_file(session, file_id)
            path = asset_crud.get_extra_file_path(session, file_id)
            return AssetFileRef(
                role=role,
                file_id=file_id,
                checkpoint_id=None,
                name=extra_file.name,
                type_=extra_file.type_,
                path=path,
                size=extra_file.size,
                content_hash=extra_file.content_hash,
                metadata=extra_file.metadata_,
            )
        except KeyError:
            checkpoint = asset_crud.get_checkpoint(session, file_id)
            path = _checkpoint_asset_path(asset_crud.get_checkpoint_path(session, file_id), role)
    return AssetFileRef(
        role=role,
        file_id=None,
        checkpoint_id=file_id,
        name=checkpoint.name,
        type_=checkpoint.type_,
        path=path,
        size=checkpoint.size,
        content_hash=checkpoint.content_hash,
        metadata=checkpoint.metadata_,
    )


def _asset_metadata(ref: AssetFileRef) -> dict[str, Any]:
    data = {
        "role": ref.role,
        "name": ref.name,
        "type": ref.type_,
        "path": str(ref.path),
        "size": ref.size,
        "content_hash": ref.content_hash,
        "metadata": ref.metadata,
    }
    if ref.file_id is not None:
        data["id"] = str(ref.file_id)
        data["storage"] = "extra_file"
    if ref.checkpoint_id is not None:
        data["id"] = str(ref.checkpoint_id)
        data["checkpoint_id"] = str(ref.checkpoint_id)
        data["storage"] = "checkpoint"
    return data


def _checkpoint_asset_path(folder: Path, role: str) -> Path:
    if role == "f0_model":
        return _single_file(folder, (".t7", ".pth", ".pt"))
    if role in {"asr_bundle", "plbert"}:
        return _single_file(folder, (".pth", ".t7", ".pt"))
    return folder


def _single_file(folder: Path, suffixes: tuple[str, ...]) -> Path:
    matches = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
    if not matches:
        raise ValueError(f"checkpoint asset file missing in {folder}")
    return matches[0]


class ResolveTrainingAssetsNode(Node):
    NODE_TYPE = "ResolveTrainingAssets"
    CATEGORY = "Training"
    SETTINGS = ResolveTrainingAssetsSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"assets": Port("assets", ASSET_BUNDLE)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params):
        super().__init__(node_id=node_id, **params)
        self._emitted = False

    def remaining_items(self, context):
        return 0 if self._emitted else 1

    async def execute(self, batch, context):
        assert not self._emitted, f"training assets node already emitted: {self.id}"
        self._emitted = True
        return [
            {
                "assets": resolve_training_asset_bundle(
                    self.settings.asr_bundle_file_ids,
                    self.settings.f0_model_file_id,
                    self.settings.plbert_file_id,
                    self.settings.ood_text_set_file_ids,
                )
            }
        ]
