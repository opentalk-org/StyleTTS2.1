from collections import Counter
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict

from runner.nodes.tts.corpus.models import (
    CorpusJob,
    CorpusPlan,
    PiperModelPlan,
)
from runner.nodes.tts.piper_catalog import PiperCatalog, PiperVoiceEntry
from runner.nodes.tts.voices import PRESET_VOICES, TtsEngine


EXPECTED_LINES = 101_250
EXPECTED_STREAMS = 741
EXPECTED_PIPER_JOBS = 71_100
KOKORO_PREFIXES = {
    "en": ("a", "b"),
    "es": ("e",),
    "fr": ("f",),
    "hi": ("h",),
    "it": ("i",),
    "ja": ("j",),
    "pt": ("p",),
    "zh": ("z",),
}
QUALITY_ORDER = {"x_low": 0, "low": 1, "medium": 2, "high": 3}


class VoiceRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    identity: str
    kind: str
    language: str
    path: Path
    lines: int


class CorpusManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    voices: tuple[VoiceRecord, ...]


def build_corpus_plan(root: Path, catalog: PiperCatalog) -> CorpusPlan:
    manifest_path = root / "manifest.json"
    manifest = CorpusManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if len(manifest.voices) != EXPECTED_STREAMS:
        raise ValueError(
            f"{manifest_path}: expected {EXPECTED_STREAMS} voices, "
            f"found {len(manifest.voices)}"
        )
    routed_engines = {
        voice.identity: _engine_for(voice)
        for voice in manifest.voices
    }
    piper_languages = {
        voice.language
        for voice in manifest.voices
        if routed_engines[voice.identity] is TtsEngine.PIPER
    }
    selected_models = _select_piper_models(catalog, piper_languages)
    piper_models = {
        language: PiperModelPlan(
            voice.voice_id,
            language,
            voice.num_speakers,
        )
        for language, voice in selected_models.items()
    }
    stream_positions: Counter[tuple[TtsEngine, str]] = Counter()
    piper_jobs: list[CorpusJob] = []
    kokoro_jobs: list[CorpusJob] = []
    for voice in manifest.voices:
        engine = routed_engines[voice.identity]
        position_key = (engine, voice.language)
        stream_position = stream_positions[position_key]
        stream_positions[position_key] += 1
        voice_id, speaker_id = _resolved_voice(
            engine,
            voice.language,
            stream_position,
            selected_models,
        )
        lines = _voice_lines(root, voice)
        target = piper_jobs if engine is TtsEngine.PIPER else kokoro_jobs
        target.extend(
            _jobs_for_voice(voice, lines, engine, voice_id, speaker_id)
        )
    plan = CorpusPlan(
        tuple(piper_jobs),
        tuple(kokoro_jobs),
        MappingProxyType(piper_models),
    )
    _validate_plan(plan)
    return plan


def without_completed(
    jobs: tuple[CorpusJob, ...],
    completed_keys: set[str],
) -> tuple[CorpusJob, ...]:
    return tuple(job for job in jobs if job.source_key not in completed_keys)


def _engine_for(voice: VoiceRecord) -> TtsEngine:
    if voice.kind not in {"registered", "piper"}:
        raise ValueError(f"{voice.identity}: unknown stream kind {voice.kind}")
    if voice.kind == "piper" and voice.language != "ja":
        return TtsEngine.PIPER
    if voice.language in KOKORO_PREFIXES:
        return TtsEngine.KOKORO
    return TtsEngine.PIPER


def _select_piper_models(
    catalog: PiperCatalog,
    languages: set[str],
) -> dict[str, PiperVoiceEntry]:
    selected: dict[str, PiperVoiceEntry] = {}
    for language in sorted(languages):
        candidates = [
            voice
            for voice in catalog
            if voice.language.family == language
        ]
        if not candidates:
            raise ValueError(f"piper catalog has no {language} voice")
        selected[language] = min(
            candidates,
            key=lambda voice: (
                -voice.num_speakers,
                QUALITY_ORDER[voice.quality],
                voice.key,
            ),
        )
    return selected


def _resolved_voice(
    engine: TtsEngine,
    language: str,
    stream_position: int,
    piper_models: dict[str, PiperVoiceEntry],
) -> tuple[str, int | None]:
    if engine is TtsEngine.PIPER:
        model = piper_models[language]
        speaker_id = (
            stream_position % model.num_speakers
            if model.num_speakers > 1
            else None
        )
        return model.voice_id, speaker_id
    prefixes = KOKORO_PREFIXES[language]
    presets = [
        voice_id
        for voice_id in PRESET_VOICES[TtsEngine.KOKORO]
        if voice_id[0] in prefixes
    ]
    if not presets:
        raise ValueError(f"kokoro has no {language} preset")
    return presets[stream_position % len(presets)], None


def _voice_lines(root: Path, voice: VoiceRecord) -> tuple[str, ...]:
    path = root / voice.path
    lines = tuple(path.read_text(encoding="utf-8").splitlines())
    if len(lines) != voice.lines:
        raise ValueError(
            f"{path}: expected {voice.lines} lines, found {len(lines)}"
        )
    if any(not line.strip() or line != line.strip() for line in lines):
        raise ValueError(f"{path}: lines must be nonempty and trimmed")
    return lines


def _jobs_for_voice(
    voice: VoiceRecord,
    lines: tuple[str, ...],
    engine: TtsEngine,
    voice_id: str,
    speaker_id: int | None,
) -> list[CorpusJob]:
    return [
        CorpusJob(
            engine=engine,
            stream_id=voice.identity,
            language=voice.language,
            sentence_index=index,
            text=text,
            voice_id=voice_id,
            speaker_id=speaker_id,
            source_key=(
                f"{engine.value}:{voice.identity}:{index:04d}"
            ),
        )
        for index, text in enumerate(lines)
    ]


def _validate_plan(plan: CorpusPlan) -> None:
    jobs = plan.jobs
    keys = [job.source_key for job in jobs]
    if len(jobs) != EXPECTED_LINES:
        raise ValueError(
            f"expected {EXPECTED_LINES} corpus jobs, found {len(jobs)}"
        )
    if len(plan.piper_jobs) != EXPECTED_PIPER_JOBS:
        raise ValueError(
            f"expected {EXPECTED_PIPER_JOBS} Piper jobs, "
            f"found {len(plan.piper_jobs)}"
        )
    if len(set(keys)) != len(keys):
        raise ValueError("corpus source keys are not unique")
