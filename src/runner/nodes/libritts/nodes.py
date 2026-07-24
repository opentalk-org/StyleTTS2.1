from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5
import wave

from pydantic import Field
from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio, AudioSegment, stable_id
from shared.audio_annotations import AudioAnnotations


class LibriTtsSourceSettings(StrictSettings):
    split_root: Path = Field(title="Extracted LibriTTS split directory")
    row_offset: int = Field(default=0, ge=0)
    row_limit: int | None = Field(default=None, ge=1)


def audio_from_path(split_root: Path, audio_path: Path) -> Audio:
    relative_path = audio_path.relative_to(split_root)
    speaker_id = relative_path.parts[0]
    transcript_path = audio_path.with_suffix(".normalized.txt")
    text = transcript_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty LibriTTS transcript: {transcript_path}")
    with wave.open(str(audio_path), "rb") as audio_file:
        sample_rate = audio_file.getframerate()
        channels = audio_file.getnchannels()
        duration = audio_file.getnframes() / float(sample_rate)
    data = audio_path.read_bytes()
    source_key = f"libritts:{relative_path.as_posix()}"
    audio_file_id = uuid5(NAMESPACE_URL, source_key)
    annotations = AudioAnnotations(
        speaker_id=speaker_id,
        metadata={
            "source": "libritts",
            "source_path": relative_path.as_posix(),
            "split": split_root.name,
            "language": "en",
            "sample_rate": sample_rate,
            "channels": channels,
            "duration": duration,
        },
    )
    segment = AudioSegment(
        source_audio_id=audio_file_id,
        name=audio_path.name,
        start=0.0,
        end=duration,
        sample_rate=sample_rate,
        channels=channels,
        text=text,
        phon="",
        id=stable_id("libritts_segment", source_key),
        lineage_id=stable_id("libritts_segment_lineage", source_key),
        segment_id=stable_id("libritts_segment_entry", source_key),
        annotations=AudioAnnotations(
            speaker_id=speaker_id,
            metadata={"type_": "libritts_normalized", "model": "libritts"},
        ),
    )
    return Audio(
        audio_file_id=audio_file_id,
        name=audio_path.name,
        data=data,
        sample_rate=sample_rate,
        channels=channels,
        start=0.0,
        end=duration,
        annotations=annotations,
        id=stable_id("libritts_audio", source_key),
        lineage_id=stable_id("libritts_audio_lineage", source_key),
        byte_length=len(data),
        virtual=False,
        segments=[segment],
    )


class LibriTtsSourceNode(Node):
    NODE_TYPE = "LibriTtsSource"
    DESCRIPTION = "Stream one extracted LibriTTS split into a graph with its normalized transcripts and speaker IDs."
    CATEGORY = "Inputs"
    SETTINGS = LibriTtsSourceSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"audio": AudioPort(mode=PortMode.STREAM)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)
    QUEUE_MAX_SIZE = 64

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._paths: list[Path] | None = None
        self._cursor = 0

    def remaining_items(self, context: Any) -> int:
        self._discover_paths()
        assert self._paths is not None
        return len(self._paths) - self._cursor

    async def execute(self, batch: list[dict[str, Any]], context: Any) -> list[dict[str, Audio]]:
        self._discover_paths()
        assert self._paths is not None
        end = self._cursor + self.runtime.queue_max_size
        paths = self._paths[self._cursor:end]
        items = []
        for path in context.cancellable(paths):
            items.append({"audio": audio_from_path(self.settings.split_root, path)})
        self._cursor += len(items)
        return items

    def _discover_paths(self) -> None:
        if self._paths is not None:
            return
        root = self.settings.split_root
        if not root.is_dir():
            raise ValueError(f"LibriTTS split directory does not exist: {root}")
        paths = sorted(root.glob("*/*/*.wav"))
        if not paths:
            raise ValueError(f"LibriTTS split contains no WAV files: {root}")
        end = None
        if self.settings.row_limit is not None:
            end = self.settings.row_offset + self.settings.row_limit
        self._paths = paths[self.settings.row_offset:end]
