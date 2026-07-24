import hashlib
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from uuid import UUID

from runner.nodes.tts.corpus.models import OtherCorpusJob
from runner.nodes.tts.corpus.plan import CorpusManifest, VoiceRecord
from runner.nodes.tts.corpus.references import RegisteredReference
from runner.nodes.tts.voices import PRESET_VOICES, TtsEngine


ALL_LANGUAGES = frozenset({
    "en", "de", "fr", "nl", "zh", "ja", "hi", "es",
    "pt", "it", "ru", "pl", "ar", "tr", "ko",
})
ENGINE_LANGUAGES = {
    TtsEngine.CHATTERBOX: ALL_LANGUAGES,
    TtsEngine.F5_TTS: frozenset({"en", "zh"}),
    TtsEngine.DIA: frozenset({"en"}),
    TtsEngine.FISH_SPEECH: ALL_LANGUAGES,
    TtsEngine.RAON_OPENTTS: frozenset({"en"}),
}
EXPECTED_JOBS = {
    TtsEngine.CHATTERBOX: 43_200,
    TtsEngine.F5_TTS: 17_100,
    TtsEngine.ORPHEUS: 3_600,
    TtsEngine.DIA: 14_850,
    TtsEngine.FISH_SPEECH: 43_200,
    TtsEngine.RAON_OPENTTS: 14_850,
}


def registered_stream_languages(
    root: Path,
    engine: TtsEngine,
) -> dict[str, str]:
    if engine is TtsEngine.ORPHEUS:
        return {}
    manifest = CorpusManifest.model_validate_json(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    languages = ENGINE_LANGUAGES[engine]
    return {
        voice.identity: voice.language
        for voice in manifest.voices
        if voice.kind == "registered" and voice.language in languages
    }


def build_other_corpus_plan(
    root: Path,
    engine: TtsEngine,
    references: Mapping[str, RegisteredReference],
) -> tuple[OtherCorpusJob, ...]:
    manifest = CorpusManifest.model_validate_json(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    registered = tuple(voice for voice in manifest.voices if voice.kind == "registered")
    if engine is TtsEngine.ORPHEUS:
        jobs = _orpheus_jobs(root, registered)
    else:
        languages = ENGINE_LANGUAGES[engine]
        jobs = tuple(
            job
            for voice in registered
            if voice.language in languages
            for job in _clone_jobs(root, voice, engine, references[voice.identity])
        )
    expected = EXPECTED_JOBS[engine]
    if len(jobs) != expected:
        raise ValueError(
            f"{engine.value}: expected {expected} corpus jobs, found {len(jobs)}"
        )
    keys = [job.source_key for job in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{engine.value}: duplicate corpus source keys")
    return jobs


def _orpheus_jobs(
    root: Path,
    voices: tuple[VoiceRecord, ...],
) -> tuple[OtherCorpusJob, ...]:
    english = tuple(voice for voice in voices if voice.language == "en")
    presets = PRESET_VOICES[TtsEngine.ORPHEUS]
    return tuple(
        OtherCorpusJob(
            engine=TtsEngine.ORPHEUS,
            stream_id=voice.identity,
            language="en",
            sentence_index=index,
            text=text,
            voice_id=preset,
            reference_audio_id=None,
            source_key=_source_key(
                TtsEngine.ORPHEUS,
                preset,
                None,
                index,
                text,
            ),
        )
        for preset, voice in zip(presets, english[:len(presets)], strict=True)
        for index, text in enumerate(_voice_lines(root, voice))
    )


def _clone_jobs(
    root: Path,
    voice: VoiceRecord,
    engine: TtsEngine,
    reference: RegisteredReference,
) -> tuple[OtherCorpusJob, ...]:
    if reference.language != voice.language:
        raise ValueError(
            f"{voice.identity}: reference language {reference.language} "
            f"differs from {voice.language}"
        )
    return tuple(
        OtherCorpusJob(
            engine=engine,
            stream_id=voice.identity,
            language=voice.language,
            sentence_index=index,
            text=text,
            voice_id=voice.identity,
            reference_audio_id=reference.audio_file_id,
            source_key=_source_key(
                engine,
                voice.identity,
                reference.audio_file_id,
                index,
                text,
            ),
        )
        for index, text in enumerate(_voice_lines(root, voice))
    )


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


def _source_key(
    engine: TtsEngine,
    voice_id: str,
    reference_audio_id: UUID | None,
    sentence_index: int,
    text: str,
) -> str:
    normalized = unicodedata.normalize("NFC", " ".join(text.split()))
    digest = hashlib.blake2b(
        normalized.encode("utf-8"),
        digest_size=12,
    ).hexdigest()
    reference = str(reference_audio_id) if reference_audio_id is not None else "preset"
    return (
        f"{engine.value}:{voice_id}:{reference}:"
        f"{sentence_index:04d}:{digest}"
    )
