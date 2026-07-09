from __future__ import annotations

from pathlib import Path
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import AssetBundlePort, CheckpointRefPort, JsonPort, TrainingManifestPort
from runner.nodes.models import AssetBundleRef, AudioSegment, CheckpointRef, TrainingManifest, stable_id, typed_assets, typed_checkpoint
from runner.nodes.training.common.manifest.build import (
    audio_file_selection,
    manifest_audio_dir,
    manifest_lines,
    materialize_audio_files,
    phoneme_alphabet_symbols,
    segments_from_audio_file_ids,
    speaker_id_map,
    speaker_key,
    usable_groups,
    write_manifest,
    write_text_sidecar,
)
from runner.nodes.training.common.manifest.cleanup import sweep_orphan_run_dirs
from runner.nodes.training.common.manifest.stream_plan import STREAM_PLAN_FILENAME, build_stream_plan, write_stream_plan

DEFAULT_MANIFEST_OUTPUT_DIR = Path("data/training/manifests")


class BuildTrainingManifestSettings(StrictSettings):
    validation_samples: int = Field(default=32, title="Validation samples", ge=1, le=512)
    output_dir: Path = Field(default=DEFAULT_MANIFEST_OUTPUT_DIR, title="Output directory")
    root_path: str = Field(default="", title="Manifest root path")
    stream_from_buckets: bool = Field(
        default=False,
        title="Stream audio from buckets",
        description="Skip copying training audio up front; the trainer fetches bucket files on demand into a bounded cache. Only consumers that support bucket streaming (StyleTTS finetune) may use this.",
    )


class BuildTrainingManifestNode(Node):
    NODE_TYPE = "BuildTrainingManifest"
    CATEGORY = "Training"
    SETTINGS = BuildTrainingManifestSettings
    INPUTS = {
        "audio_file_ids": JsonPort(),
        "checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST),
        "assets": AssetBundlePort(join_mode=JoinMode.BROADCAST, optional=True, default=None),
        "phoneme_alphabet": JsonPort(join_mode=JoinMode.BROADCAST, optional=True, default={}),
    }
    OUTPUTS = {"training_manifest": TrainingManifestPort()}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            selection = audio_file_selection(inputs["audio_file_ids"])
            settings = manifest_settings_for_run(self.settings, context.run_id)
            outputs.append({
                "training_manifest": build_training_manifest(
                    dataset_id=selection.dataset_id,
                    segments=segments_from_audio_file_ids(selection.audio_file_ids),
                    base_checkpoint=typed_checkpoint(inputs["checkpoint"]),
                    assets=typed_assets(inputs["assets"]),
                    phoneme_alphabet=phoneme_alphabet_symbols(inputs["phoneme_alphabet"]),
                    settings=settings,
                )
            })
        return outputs


def build_training_manifest(
    dataset_id: UUID,
    segments: list[AudioSegment],
    base_checkpoint: CheckpointRef,
    assets: AssetBundleRef | None,
    phoneme_alphabet: list[str],
    settings: BuildTrainingManifestSettings,
) -> TrainingManifest:
    output_dir = Path(settings.output_dir)
    sweep_orphan_run_dirs(output_dir.parent, output_dir.name)
    audio_dir = manifest_audio_dir(output_dir, settings.root_path)
    groups = usable_groups(segments)
    if len(groups) < 2:
        raise ValueError("training manifest requires at least 2 usable source audio groups")

    natural_ids = list(groups)
    validation_count = min(settings.validation_samples, len(natural_ids) - 1)
    if validation_count < 1:
        raise ValueError("training manifest requires at least 1 validation item")
    train_ids = natural_ids[:-validation_count]
    validation_ids = natural_ids[-validation_count:]
    speaker_ids = speaker_id_map(groups)

    lists_dir = output_dir / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    train_ids, stream_metadata = _prepare_audio(train_ids, validation_ids, groups, audio_dir, lists_dir, settings)

    train_lines = manifest_lines(groups, train_ids, settings.root_path, audio_dir, speaker_ids)
    validation_lines = manifest_lines(groups, validation_ids, settings.root_path, audio_dir, speaker_ids)
    if not train_lines:
        raise ValueError("training manifest requires at least 1 training item")
    if not validation_lines:
        raise ValueError("training manifest requires at least 1 validation item")

    train_path = lists_dir / "train.txt"
    validation_path = lists_dir / "validation.txt"
    write_manifest(train_path, train_lines)
    write_manifest(validation_path, validation_lines)

    ordered_audio_ids = train_ids + validation_ids
    sidecar_path = lists_dir / "segments.jsonl"
    write_text_sidecar(sidecar_path, groups, ordered_audio_ids)

    all_lines = train_lines + validation_lines
    manifest_id = stable_id("training_manifest", dataset_id, base_checkpoint.id, len(all_lines))
    return TrainingManifest(
        dataset_id=dataset_id,
        audio_file_ids=ordered_audio_ids,
        base_checkpoint=base_checkpoint,
        phoneme_alphabet=phoneme_alphabet,
        id=manifest_id,
        lineage_id=manifest_id,
        assets=assets,
        metadata={
            "train_manifest_path": str(train_path),
            "validation_manifest_path": str(validation_path),
            "segment_text_path": str(sidecar_path),
            "train_count": len(train_lines),
            "validation_count": len(validation_lines),
            "segment_count": sum(len(items) for items in groups.values()),
            "audio_count": len(all_lines),
            "root_path": settings.root_path,
            "audio_dir": str(audio_dir),
            "ordered_audio_ids": [str(audio_id) for audio_id in ordered_audio_ids],
            "validation_audio_ids": [str(audio_id) for audio_id in validation_ids],
            **stream_metadata,
        },
    )


def _prepare_audio(
    train_ids: list[UUID],
    validation_ids: list[UUID],
    groups: dict[UUID, list[AudioSegment]],
    audio_dir: Path,
    lists_dir: Path,
    settings: BuildTrainingManifestSettings,
) -> tuple[list[UUID], dict[str, object]]:
    """Materialize audio for the run and return the training order + metadata.

    Streaming skips copying the train split (the trainer fetches buckets on
    demand) and only keeps the small validation split resident; the eager path
    copies every wav as before so asr/f0 trainers keep reading loose files."""
    if not settings.stream_from_buckets:
        materialize_audio_files(train_ids + validation_ids, audio_dir)
        return train_ids, {}
    speaker_of = {audio_id: speaker_key(groups[audio_id][0]) for audio_id in train_ids}
    plan = build_stream_plan(train_ids, speaker_of)
    plan_path = lists_dir / STREAM_PLAN_FILENAME
    write_stream_plan(plan_path, plan)
    materialize_audio_files(validation_ids, audio_dir)
    metadata = {
        "stream_from_buckets": True,
        "stream_plan_path": str(plan_path),
        "cache_dir": str(audio_dir),
    }
    return plan.ordered_audio_ids(), metadata


def manifest_settings_for_run(settings: BuildTrainingManifestSettings, run_id: object) -> BuildTrainingManifestSettings:
    if settings.output_dir != DEFAULT_MANIFEST_OUTPUT_DIR:
        return settings
    return settings.model_copy(update={"output_dir": DEFAULT_MANIFEST_OUTPUT_DIR / str(run_id)})
