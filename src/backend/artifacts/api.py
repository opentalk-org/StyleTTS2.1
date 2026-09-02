from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Response, status

from shared.db import database_session
from shared.db.assets import clickhouse as assets
from shared.db.assets import crud as asset_crud
from shared.db.assets.clickhouse import AssetKind
from shared.db.assets.schemas import FileAssetRead

ARTIFACT_TYPE = "artifact"

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


@router.get("", response_model=list[FileAssetRead])
async def list_artifacts() -> list[FileAssetRead]:
    return [
        FileAssetRead.model_validate(item)
        for item in assets.list_assets(AssetKind.FILE, ARTIFACT_TYPE)
    ]


@router.get("/{artifact_id}/content")
async def artifact_content(artifact_id: UUID) -> Response:
    try:
        item = assets.get_asset(artifact_id)
        if item.kind != AssetKind.FILE:
            raise KeyError(f"File asset not found: {artifact_id}")
        with database_session() as session:
            data = asset_crud.read_extra_file(session, artifact_id)
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
        ) from error
    return Response(
        data,
        media_type=_content_type(item.metadata_),
        headers={"Content-Length": str(len(data))},
    )


@router.delete("/{artifact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_artifact(artifact_id: UUID) -> None:
    try:
        with database_session() as session:
            asset_crud.delete_extra_file(session, artifact_id)
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
        ) from error


def _content_type(metadata: dict[str, Any]) -> str:
    if "content_type" in metadata:
        return str(metadata["content_type"])
    return "application/octet-stream"
