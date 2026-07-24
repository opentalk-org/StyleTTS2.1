from __future__ import annotations

from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, Field

from runner.nodes.models import CheckpointRef, TrainingManifest, stable_id
from shared.db import database_session
from shared.db.mos import crud as mos_crud
from shared.db.mos.models import MosComparison


class MosManifestRow(BaseModel):
    comparison_id: UUID
    audio_a_id: UUID
    audio_b_id: UUID
    score_a: float = Field(allow_inf_nan=False)
    score_b: float = Field(allow_inf_nan=False)
    preferred_audio_id: UUID

    @property
    def preferred_sign(self) -> float:
        if self.preferred_audio_id == self.audio_a_id:
            return 1.0
        if self.preferred_audio_id == self.audio_b_id:
            return -1.0
        raise ValueError(f"MOS manifest preference is outside pair: {self.comparison_id}")


class MosManifestPaths(BaseModel):
    train_path: Path
    validation_path: Path
    train_count: int
    validation_count: int


def build_mos_training_manifest(
    dataset_id: UUID,
    checkpoint: CheckpointRef,
    validation_comparisons: int,
    output_dir: Path,
) -> TrainingManifest:
    if checkpoint.metadata["type"] != "mos_base":
        raise ValueError(f"MOS training requires mos_base checkpoint: {checkpoint.checkpoint_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _write_manifest_files(dataset_id, validation_comparisons, output_dir)
    manifest_id = stable_id(
        "mos_training_manifest",
        dataset_id,
        checkpoint.id,
        paths.train_count,
        paths.validation_count,
    )
    return TrainingManifest(
        dataset_id=dataset_id,
        audio_file_ids=[],
        base_checkpoint=checkpoint,
        phoneme_alphabet=[],
        id=manifest_id,
        lineage_id=manifest_id,
        metadata={
            "kind": "mos_pairs",
            "train_manifest_path": str(paths.train_path),
            "validation_manifest_path": str(paths.validation_path),
            "train_count": paths.train_count,
            "validation_count": paths.validation_count,
            "comparison_count": paths.train_count + paths.validation_count,
        },
    )


def _write_manifest_files(
    dataset_id: UUID,
    validation_comparisons: int,
    output_dir: Path,
) -> MosManifestPaths:
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    with database_session() as session:
        comparison_count = mos_crud.count_comparisons(session, dataset_id)
        if comparison_count < 2:
            raise ValueError(f"MOS training requires at least two comparisons: {dataset_id}")
        validation_count = min(validation_comparisons, comparison_count - 1)
        train_count = comparison_count - validation_count
        written = 0
        with train_path.open("w", encoding="utf-8") as train_file, validation_path.open("w", encoding="utf-8") as validation_file:
            for index, comparison in enumerate(mos_crud.iter_comparisons(session, dataset_id)):
                target = train_file if index < train_count else validation_file
                target.write(_manifest_row(comparison).model_dump_json() + "\n")
                written += 1
        assert written == comparison_count, f"MOS comparison count changed while building manifest: {written}/{comparison_count}"
    return MosManifestPaths(
        train_path=train_path,
        validation_path=validation_path,
        train_count=train_count,
        validation_count=validation_count,
    )


def _manifest_row(comparison: MosComparison) -> MosManifestRow:
    return MosManifestRow(
        comparison_id=comparison.id,
        audio_a_id=comparison.audio_a_id,
        audio_b_id=comparison.audio_b_id,
        score_a=comparison.score_a,
        score_b=comparison.score_b,
        preferred_audio_id=comparison.preferred_audio_id,
    )
