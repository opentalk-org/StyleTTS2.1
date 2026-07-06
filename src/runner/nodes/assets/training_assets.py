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
    file_id: UUID
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
    refs = [_resolve_extra_file(role, file_id) for role, file_id in requested]
    extra_file_ids = [ref.file_id for ref in refs]
    digest_parts = [str(file_id) for file_id in extra_file_ids]
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
        },
    )


def prefetch_training_assets(value: AssetBundleRef | dict[str, Any]) -> AssetBundleRef:
    if isinstance(value, AssetBundleRef):
        return value
    raise TypeError("PrefetchTrainingAssets requires a resolved AssetBundleRef")


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


def _resolve_extra_file(role: str, file_id: UUID) -> AssetFileRef:
    with database_session() as session:
        extra_file = asset_crud.get_extra_file(session, file_id)
        path = asset_crud.get_extra_file_path(session, file_id)
    return AssetFileRef(
        role=role,
        file_id=file_id,
        name=extra_file.name,
        type_=extra_file.type_,
        path=path,
        size=extra_file.size,
        content_hash=extra_file.content_hash,
        metadata=extra_file.metadata_,
    )


def _asset_metadata(ref: AssetFileRef) -> dict[str, Any]:
    return {
        "role": ref.role,
        "id": str(ref.file_id),
        "name": ref.name,
        "type": ref.type_,
        "path": str(ref.path),
        "size": ref.size,
        "content_hash": ref.content_hash,
        "metadata": ref.metadata,
    }


class ResolveTrainingAssetsNode(Node):
    NODE_TYPE = "ResolveTrainingAssets"
    CATEGORY = "Training / Assets"
    SETTINGS = ResolveTrainingAssetsSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"asset_refs": Port("asset_refs", ASSET_BUNDLE)}
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
                "asset_refs": resolve_training_asset_bundle(
                    self.settings.asr_bundle_file_ids,
                    self.settings.f0_model_file_id,
                    self.settings.plbert_file_id,
                    self.settings.ood_text_set_file_ids,
                )
            }
        ]
