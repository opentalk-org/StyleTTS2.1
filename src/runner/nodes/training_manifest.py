from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from uuid import UUID

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port, PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runflow.core.types import UnionDataType
from runner.nodes.datatypes import AUDIO_SEGMENT, JSON, SEGMENT_GROUP, TRAINING_MANIFEST
from runner.nodes.models import AssetBundleRef, AudioSegment, CheckpointRef, SegmentGroup, TrainingManifest, stable_id
from runner.nodes.training_config import ASSET_BUNDLE_OR_JSON, CHECKPOINT_REF_OR_JSON


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
    text: str
    phon: str
    speaker: str


class BuildTrainingManifestNode(Node):
    NODE_TYPE = "BuildTrainingManifest"
    CATEGORY = "Training / Preparation"
    SETTINGS = BuildTrainingManifestSettings
    INPUTS = {
        "segments": Port("segments", TRAINING_SEGMENT_INPUT, mode=PortMode.LIST),
        "base_checkpoint": Port("base_checkpoint", CHECKPOINT_REF_OR_JSON),
        "assets": Port("assets", ASSET_BUNDLE_OR_JSON, optional=True, default=None),
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
    groups = _usable_groups(segments)
    if len(groups) < 2:
        raise ValueError("training manifest requires at least 2 usable source audio groups")

    ordered_audio_ids = list(groups)
    validation_count = min(settings.validation_samples, len(ordered_audio_ids) - 1)
    if validation_count < 1:
        raise ValueError("training manifest requires at least 1 validation item")

    validation_ids = set(ordered_audio_ids[:validation_count])
    train_lines = _manifest_lines(groups, ordered_audio_ids[validation_count:], settings.root_path)
    validation_lines = _manifest_lines(groups, ordered_audio_ids[:validation_count], settings.root_path)
    if not validation_lines:
        raise ValueError("training manifest requires at least 1 validation segment")
    if not train_lines:
        raise ValueError("training manifest requires at least 1 training segment")

    lists_dir = Path(settings.output_dir) / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    train_path = lists_dir / "train.txt"
    validation_path = lists_dir / "validation.txt"
    _write_manifest(train_path, train_lines)
    _write_manifest(validation_path, validation_lines)

    text_sidecar_path = lists_dir / "segments.jsonl"
    all_lines = train_lines + validation_lines
    _write_text_sidecar(text_sidecar_path, all_lines)

    segment_count = len(all_lines)
    manifest_id = stable_id("training_manifest", settings.dataset_id, base_checkpoint.id, segment_count)
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
            "segment_text_path": str(text_sidecar_path),
            "train_count": len(train_lines),
            "validation_count": len(validation_lines),
            "segment_count": segment_count,
            "root_path": settings.root_path,
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


def _usable_groups(segments: list[AudioSegment]) -> dict[UUID, list[AudioSegment]]:
    groups: dict[UUID, list[AudioSegment]] = {}
    for segment in sorted(segments, key=_segment_sort_key):
        if not segment.phon.strip() or segment.start >= segment.end:
            continue
        groups.setdefault(segment.source_audio_id, []).append(segment)
    return {audio_id: items for audio_id, items in groups.items() if items}


def _segment_sort_key(segment: AudioSegment) -> tuple[str, float, float, str]:
    segment_key = segment.segment_id if segment.segment_id is not None else segment.id
    return (str(segment.source_audio_id), segment.start, segment.end, segment_key)


def _manifest_lines(groups: dict[UUID, list[AudioSegment]], audio_ids: list[UUID], root_path: str) -> list[ManifestLine]:
    lines: list[ManifestLine] = []
    for audio_id in audio_ids:
        for segment in groups[audio_id]:
            lines.append(
                ManifestLine(
                    audio_id=audio_id,
                    value=_manifest_audio_value(segment, root_path),
                    text=segment.text,
                    phon=segment.phon,
                    speaker=segment.speaker or "speaker_0",
                )
            )
    return lines


def _manifest_audio_value(segment: AudioSegment, root_path: str) -> str:
    if root_path:
        return str(Path(root_path) / segment.name)
    return str(segment.source_audio_id)


def _write_manifest(path: Path, lines: list[ManifestLine]) -> None:
    content = "".join(f"{line.value}|{line.phon}|{line.speaker}\n" for line in lines)
    path.write_text(content, encoding="utf-8")


def _write_text_sidecar(path: Path, lines: list[ManifestLine]) -> None:
    content = "".join(json.dumps(line.__dict__, default=str, ensure_ascii=False) + "\n" for line in lines)
    path.write_text(content, encoding="utf-8")


def _typed_checkpoint(value: CheckpointRef | dict) -> CheckpointRef:
    if isinstance(value, CheckpointRef):
        return value
    raise TypeError("BuildTrainingManifest requires a resolved CheckpointRef for base_checkpoint")


def _typed_assets(value: AssetBundleRef | dict | None) -> AssetBundleRef | None:
    if value is None or isinstance(value, AssetBundleRef):
        return value
    raise TypeError("BuildTrainingManifest requires resolved AssetBundleRef values for assets")
