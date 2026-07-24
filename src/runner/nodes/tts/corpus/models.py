from collections.abc import Mapping
from dataclasses import dataclass
from uuid import UUID

from runner.nodes.tts.voices import TtsEngine


@dataclass(frozen=True, slots=True)
class PiperModelPlan:
    voice_id: str
    language: str
    num_speakers: int


@dataclass(frozen=True, slots=True)
class CorpusJob:
    engine: TtsEngine
    stream_id: str
    language: str
    sentence_index: int
    text: str
    voice_id: str
    speaker_id: int | None
    source_key: str

    @property
    def dataset_name(self) -> str:
        return f"tts_{self.engine.value}"


@dataclass(frozen=True, slots=True)
class CorpusPlan:
    piper_jobs: tuple[CorpusJob, ...]
    kokoro_jobs: tuple[CorpusJob, ...]
    piper_models: Mapping[str, PiperModelPlan]

    @property
    def jobs(self) -> tuple[CorpusJob, ...]:
        return self.piper_jobs + self.kokoro_jobs


@dataclass(frozen=True, slots=True)
class OtherCorpusJob:
    engine: TtsEngine
    stream_id: str
    language: str
    sentence_index: int
    text: str
    voice_id: str
    reference_audio_id: UUID | None
    source_key: str

    @property
    def dataset_name(self) -> str:
        return f"tts_{self.engine.value}"
