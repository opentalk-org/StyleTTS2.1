from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port, PortMode
from runflow.core.settings import StrictSettings
from runflow.core.types import UnionDataType
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import ASSET_BUNDLE, AUDIO_SEGMENT, CHECKPOINT_REF, JSON, SEGMENT_GROUP, TRAINING_MANIFEST
from runner.nodes.models import AssetBundleRef, AudioSegment, CheckpointRef, SegmentGroup, TrainingManifest, stable_id
from shared.db import database_session
from shared.db.audio import crud as audio_crud


TRAINING_SEGMENT_INPUT = UnionDataType("TRAINING_SEGMENT_INPUT", (AUDIO_SEGMENT, SEGMENT_GROUP), "Segment or segment group")


class BuildTrainingManifestSettings(StrictSettings):
    dataset_id: UUID = Field(title="Dataset")
    validation_samples: int = Field(default=32, title="Validation samples", ge=1, le=512)
    output_dir: Path = Field(title="Output directory")
    root_path: str = Field(default="", title="Manifest root path")


@dataclass(frozen=True)
class ManifestLine:
    audio_id: UUID
    value: str
    phon: str
    speaker: str


class BuildTrainingManifestNode(Node):
    NODE_TYPE = "BuildTrainingManifest"
    CATEGORY = "Training / Preparation"
    SETTINGS = BuildTrainingManifestSettings
    INPUTS = {
        "segments": Port("segments", TRAINING_SEGMENT_INPUT, mode=PortMode.LIST),
        "base_checkpoint": Port("base_checkpoint", CHECKPOINT_REF),
        "assets": Port("assets", ASSET_BUNDLE, optional=True, default=None),
        "phoneme_alphabet": Port("phoneme_alphabet", JSON, optional=True, default={}),
    }
    OUTPUTS = {"training_manifest": Port("training_manifest", TRAINING_MANIFEST)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        return [
            {
                "training_manifest": build_training_manifest(
                    segments=flatten_segment_inputs(inputs["segments"]),
                    base_checkpoint=_typed_checkpoint(inputs["base_checkpoint"]),
                    assets=_typed_assets(inputs["assets"]),
                    phoneme_alphabet=phoneme_alphabet_symbols(inputs["phoneme_alphabet"]),
                    settings=self.settings,
                )
            }
            for inputs in batch
        ]


def build_training_manifest(
    segments: list[AudioSegment],
    base_checkpoint: CheckpointRef,
    assets: AssetBundleRef | None,
    phoneme_alphabet: list[str],
    settings: BuildTrainingManifestSettings,
) -> TrainingManifest:
    audio_dir = _manifest_audio_dir(settings)
    groups = _usable_groups(segments, audio_dir)
    if len(groups) < 2:
        raise ValueError("training manifest requires at least 2 usable source audio groups")

    ordered_audio_ids = list(groups)
    validation_count = min(settings.validation_samples, len(ordered_audio_ids) - 1)
    if validation_count < 1:
        raise ValueError("training manifest requires at least 1 validation item")

    validation_ids = set(ordered_audio_ids[-validation_count:])
    train_lines = _manifest_lines(groups, ordered_audio_ids[:-validation_count], settings.root_path, audio_dir)
    validation_lines = _manifest_lines(groups, ordered_audio_ids[-validation_count:], settings.root_path, audio_dir)
    if not train_lines:
        raise ValueError("training manifest requires at least 1 training item")
    if not validation_lines:
        raise ValueError("training manifest requires at least 1 validation item")

    lists_dir = Path(settings.output_dir) / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    train_path = lists_dir / "train.txt"
    validation_path = lists_dir / "validation.txt"
    _write_manifest(train_path, train_lines)
    _write_manifest(validation_path, validation_lines)

    sidecar_path = lists_dir / "segments.jsonl"
    all_lines = train_lines + validation_lines
    _write_text_sidecar(sidecar_path, groups, ordered_audio_ids)

    manifest_id = stable_id("training_manifest", settings.dataset_id, base_checkpoint.id, len(all_lines))
    return TrainingManifest(
        dataset_id=settings.dataset_id,
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
            "validation_audio_ids": [str(audio_id) for audio_id in ordered_audio_ids if audio_id in validation_ids],
        },
    )


def phoneme_alphabet_symbols(value: dict) -> list[str]:
    symbols = value["symbols"] if "symbols" in value else ""
    if isinstance(symbols, str):
        return [part for part in symbols.split(" ") if part]
    if isinstance(symbols, list):
        return [str(part) for part in symbols]
    raise TypeError("phoneme alphabet symbols must be a string or list")


def flatten_segment_inputs(values: list[AudioSegment | SegmentGroup]) -> list[AudioSegment]:
    segments: list[AudioSegment] = []
    for value in values:
        if isinstance(value, AudioSegment):
            segments.append(value)
        elif isinstance(value, SegmentGroup):
            segments.extend(value.segments)
        else:
            raise TypeError(f"unsupported training segment input: {type(value).__name__}")
    return segments


def _usable_groups(segments: list[AudioSegment], audio_dir: Path) -> dict[UUID, list[AudioSegment]]:
    groups: dict[UUID, list[AudioSegment]] = {}
    for segment in sorted(segments, key=_segment_sort_key):
        if not segment.phon.strip() or segment.start >= segment.end:
            continue
        groups.setdefault(segment.source_audio_id, []).append(segment)
    usable = {audio_id: items for audio_id, items in groups.items() if items}
    _materialize_audio_files(usable, audio_dir)
    return usable


def _segment_sort_key(segment: AudioSegment) -> tuple[str, float, float, str]:
    segment_key = segment.segment_id if segment.segment_id is not None else segment.id
    return (str(segment.source_audio_id), segment.start, segment.end, segment_key)


def _manifest_lines(
    groups: dict[UUID, list[AudioSegment]],
    audio_ids: list[UUID],
    root_path: str,
    audio_dir: Path,
) -> list[ManifestLine]:
    lines: list[ManifestLine] = []
    for audio_id in audio_ids:
        segments = groups[audio_id]
        lines.append(
            ManifestLine(
                audio_id=audio_id,
                value=_manifest_audio_value(audio_id, root_path, audio_dir),
                phon=" ".join(segment.phon.strip() for segment in segments),
                speaker=_speaker_key(segments[0]),
            )
        )
    return lines


def _manifest_audio_value(audio_id: UUID, root_path: str, audio_dir: Path) -> str:
    filename = f"{audio_id}.wav"
    if root_path:
        return str(Path(root_path) / filename)
    return str((audio_dir / filename).resolve())


def _manifest_audio_dir(settings: BuildTrainingManifestSettings) -> Path:
    if settings.root_path:
        return Path(settings.root_path)
    return Path(settings.output_dir) / "audio"


def _speaker_key(segment: AudioSegment) -> str:
    if segment.voice_id is not None:
        return str(segment.voice_id)
    if segment.speaker is not None and segment.speaker.strip():
        return segment.speaker.strip()
    return "0"


def _materialize_audio_files(groups: dict[UUID, list[AudioSegment]], audio_dir: Path) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)
    with database_session() as session:
        for audio_id in groups:
            target = audio_dir / f"{audio_id}.wav"
            target.write_bytes(audio_crud.read_audio_file(session, audio_id))


def _write_manifest(path: Path, lines: list[ManifestLine]) -> None:
    content = "".join(f"{line.value}|{line.phon}|{line.speaker}\n" for line in lines)
    path.write_text(content, encoding="utf-8")


def _write_text_sidecar(path: Path, groups: dict[UUID, list[AudioSegment]], audio_ids: list[UUID]) -> None:
    rows = [_segment_sidecar_row(segment) for audio_id in audio_ids for segment in groups[audio_id]]
    content = "".join(json.dumps(row, default=str, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def _segment_sidecar_row(segment: AudioSegment) -> dict[str, object]:
    return {
        "audio_id": str(segment.source_audio_id),
        "segment_id": segment.segment_id,
        "runtime_segment_id": segment.id,
        "lineage_id": segment.lineage_id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "phon": segment.phon,
        "speaker": segment.speaker,
        "voice_id": str(segment.voice_id) if segment.voice_id is not None else None,
        "metadata": segment.metadata,
    }


def _typed_checkpoint(value: CheckpointRef | dict) -> CheckpointRef:
    if isinstance(value, CheckpointRef):
        return value
    raise TypeError("BuildTrainingManifest requires a resolved CheckpointRef for base_checkpoint")


def _typed_assets(value: AssetBundleRef | dict | None) -> AssetBundleRef | None:
    if value is None or isinstance(value, AssetBundleRef):
        return value
    raise TypeError("BuildTrainingManifest requires resolved AssetBundleRef values for assets")
