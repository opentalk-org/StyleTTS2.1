import argparse
import csv
import hashlib
import json
import re
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from imports.stage1.common.audio import normalize_audio
from imports.stage1.common.schema import AudioRecord, DatasetManifest, DatasetRecord, SegmentRecord


@dataclass(frozen=True)
class Candidate:
    source: Path
    source_id: str
    language: str
    text: str
    speaker_id: str | None
    style_prompt: str | None
    metadata: dict[str, str]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source_url: str
    candidates: tuple[Candidate, ...]


def paired_by_stem(audio: list[Path], text: list[Path]) -> list[tuple[Path, Path]]:
    audio_by_stem = {path.stem: path for path in audio}
    text_by_stem = {path.stem: path for path in text}
    return [(audio_by_stem[stem], text_by_stem[stem]) for stem in sorted(audio_by_stem.keys() & text_by_stem.keys())]


def relative_candidates(root: Path, paths: list[Path], language: str) -> tuple[Candidate, ...]:
    return tuple(Candidate(
        source=path,
        source_id=path.relative_to(root).as_posix(),
        language=language,
        text="",
        speaker_id=None,
        style_prompt=path.parent.name,
        metadata={"publisher_path": path.relative_to(root).as_posix(), "transcript_status": "untranscribed"},
    ) for path in sorted(paths))


def kuet_spec(stage_root: Path) -> DatasetSpec:
    root = stage_root / "tmp/repository"
    paths = list(root.glob("*/*.wav"))
    if len(paths) != 900:
        raise ValueError(f"KUET/KBES expected 900 WAVs, found {len(paths)}")
    return DatasetSpec("KUET/KBES", "https://data.mendeley.com/datasets/vsn37ps3rx/4", relative_candidates(root, paths, "bn"))


def arabic_natural_spec(stage_root: Path) -> DatasetSpec:
    root = stage_root / "tmp/extracted/1sec_segmented"
    paths = []
    for path in root.rglob("*.wav"):
        try:
            sf.info(path)
        except sf.LibsndfileError:
            continue
        paths.append(path)
    if len(paths) != 1_420:
        raise ValueError(f"Arabic Natural Audio expected 1,420 clips, found {len(paths)}")
    return DatasetSpec("Arabic Natural Audio", "https://data.mendeley.com/datasets/xm232yxf7t/1", relative_candidates(root, paths, "ar"))


def msa_moroccan_spec(stage_root: Path) -> DatasetSpec:
    root = stage_root / "tmp/repository"
    csv_path = root / "Darija-MSA/darija_MSA.csv"
    rows = {row["audio_path"]: row for row in csv.DictReader(csv_path.open(encoding="utf-8"))}
    paths = sorted(root.glob("Darija-MSA/Audio/*.wav"))
    if len(paths) != 1_000:
        raise ValueError(f"MSA-Moroccan expected 1,000 public WAVs, found {len(paths)}")
    candidates = []
    for path in paths:
        source_id = path.relative_to(root).as_posix()
        row = rows.get(source_id)
        candidates.append(Candidate(
            source=path, source_id=source_id, language="ary",
            text=row["text"].strip() if row else "", speaker_id=None,
            style_prompt=None,
            metadata={
                "publisher_path": source_id,
                "publisher_label": row["label"] if row else "",
                "transcript_status": "publisher" if row else "untranscribed",
            },
        ))
    return DatasetSpec("MSA-Moroccan (public subset)", "https://data.mendeley.com/datasets/kfjztyzztb/1", tuple(candidates))


def mder_ma_spec(stage_root: Path) -> DatasetSpec:
    root = stage_root / "tmp/repository"
    pairs = paired_by_stem(list(root.glob("ERD-MA Audio/*/*.wav")), list(root.glob("ERD-MA Text/*/*.txt")))
    if len(pairs) != 1_245:
        raise ValueError(f"MDER-MA expected 1,245 pairs, found {len(pairs)}")
    candidates = []
    for audio, text in pairs:
        source_id = audio.relative_to(root).as_posix()
        fields = audio.stem.split("_")
        candidates.append(Candidate(
            source=audio, source_id=source_id, language="ary",
            text=text.read_text(encoding="utf-8-sig").strip(),
            speaker_id="_".join(fields[1:3]), style_prompt=audio.parent.name.lower(),
            metadata={"publisher_path": source_id, "transcript_path": text.relative_to(root).as_posix()},
        ))
    return DatasetSpec("MDER-MA (1,245 matched pairs)", "https://data.mendeley.com/datasets/yzsw3ff6rn/1", tuple(candidates))


def escorpus_pe_spec(stage_root: Path) -> DatasetSpec:
    root = stage_root / "tmp/extracted/Corpus_Globalv1"
    paths = list(root.glob("Audio*/*.wav"))
    if len(paths) != 3_764:
        raise ValueError(f"ESCorpus-PE expected 3,764 WAVs, found {len(paths)}")
    return DatasetSpec(
        "ESCorpus-PE",
        "https://zenodo.org/records/5793223",
        relative_candidates(root, paths, "es"),
    )


def banspemo_spec(stage_root: Path) -> DatasetSpec:
    root = stage_root / "tmp/extracted/BANSpEmo A Bangla Language Emotional Speech Recognition Dataset/BANSpEmo Dataset"
    paths = sorted(root.glob("*.wav"))
    if len(paths) != 792:
        raise ValueError(f"BANSpEmo expected 792 WAVs, found {len(paths)}")
    emotions = {
        "01": "disgust",
        "02": "happy",
        "03": "sad",
        "04": "surprised",
        "05": "anger",
        "06": "fear",
    }
    candidates = []
    for path in paths:
        match = re.fullmatch(r"ss([12])_s([1-6])_sp(\d+)_([mf])_(0[1-6])", path.stem)
        if match is None:
            raise ValueError(f"unexpected BANSpEmo filename: {path.name}")
        sentence_set, sentence, speaker, gender, emotion = match.groups()
        candidates.append(Candidate(
            source=path,
            source_id=path.name,
            language="bn",
            text="",
            speaker_id=f"banspemo_{speaker}",
            style_prompt=emotions[emotion],
            metadata={
                "publisher_path": path.name,
                "sentence_set": sentence_set,
                "sentence": sentence,
                "speaker": speaker,
                "gender": "male" if gender == "m" else "female",
                "emotion_code": emotion,
                "emotion": emotions[emotion],
                "transcript_status": "untranscribed",
            },
        ))
    return DatasetSpec(
        "BANSpEmo",
        "https://data.mendeley.com/datasets/rdwn4bs5ky/2",
        tuple(candidates),
    )


def banglaser_spec(stage_root: Path) -> DatasetSpec:
    root = stage_root / "tmp/extracted/t9h6p943xy-2/BEASC Dataset"
    paths = sorted(root.glob("Actor */*.wav"))
    if len(paths) != 1_224:
        raise ValueError(f"BanglaSER expected 1,224 publisher WAVs, found {len(paths)}")
    emotions = {"01": "happy", "02": "sad", "03": "angry", "04": "surprise"}
    candidates = []
    for path in paths:
        fields = path.stem.split("-")
        if len(fields) != 7:
            raise ValueError(f"unexpected BanglaSER filename: {path.name}")
        modality, channel, emotion, intensity, statement, repetition, actor = fields
        source_id = path.relative_to(root).as_posix()
        candidates.append(Candidate(
            source=path,
            source_id=source_id,
            language="bn",
            text="",
            speaker_id=f"banglaser_{actor}",
            style_prompt=emotions[emotion],
            metadata={
                "publisher_path": source_id,
                "modality_code": modality,
                "channel_code": channel,
                "emotion_code": emotion,
                "emotion": emotions[emotion],
                "intensity_code": intensity,
                "statement_code": statement,
                "repetition_code": repetition,
                "actor": actor,
                "gender": "male" if int(actor) % 2 else "female",
                "transcript_status": "untranscribed",
            },
        ))
    return DatasetSpec(
        "BanglaSER",
        "https://data.mendeley.com/datasets/t9h6p943xy/2",
        tuple(candidates),
    )


def soreva_spec(stage_root: Path) -> DatasetSpec:
    root = stage_root / "tmp/repository/data"
    language_codes = {"af_za": "af", "pcm_cm": "pcm", "swa_ke": "sw"}
    candidates = []
    for configuration, language in language_codes.items():
        archive = root / configuration / "audio/test.tar.gz"
        extract_root = root / configuration / "audio/extracted"
        with tarfile.open(archive) as handle:
            handle.extractall(extract_root, filter="data")
        rows = {}
        with (root / configuration / "test.tsv").open(encoding="utf-8") as handle:
            for filename, raw_text, text, gender in csv.reader(handle, delimiter="\t"):
                rows[filename] = (raw_text, text, gender)
        paths = sorted(extract_root.rglob("*.wav"))
        if len(paths) != 150 or set(path.name for path in paths) != set(rows):
            raise ValueError(f"SOREVA {configuration} publisher audio/metadata mismatch")
        for path in paths:
            raw_text, text, gender = rows[path.name]
            source_id = f"{configuration}/{path.name}"
            candidates.append(Candidate(
                source=path,
                source_id=source_id,
                language=language,
                text=text.strip(),
                speaker_id=None,
                style_prompt=None,
                metadata={
                    "publisher_path": source_id,
                    "configuration": configuration,
                    "raw_transcription": raw_text,
                    "gender": gender.lower(),
                    "transcript_status": "publisher",
                },
            ))
    return DatasetSpec(
        "SOREVA",
        "https://huggingface.co/datasets/OlameMend/soreva",
        tuple(candidates),
    )


def make_record(candidate: Candidate, stage_root: Path) -> AudioRecord:
    output_id = hashlib.sha1(candidate.source_id.encode()).hexdigest()[:16]
    destination = stage_root / "wavs" / f"{candidate.language}_{output_id}.wav"
    duration = normalize_audio(candidate.source, destination)
    return AudioRecord(
        path=destination.relative_to(stage_root).as_posix(), source_id=candidate.source_id,
        duration=duration, language=candidate.language, speaker_id=candidate.speaker_id,
        style_prompt=candidate.style_prompt, voice_prompt=None, score=None, accuracy=None,
        segments=[SegmentRecord(start=0.0, end=duration, text=candidate.text, source="dataset",
                                score=None, accuracy=None, alignment=[])],
        metadata=candidate.metadata,
    )


def prepare(slug: str, workers: int) -> None:
    stage_root = Path("imports/stage1") / slug
    factories = {
        "arabic_natural_audio": arabic_natural_spec,
        "kuet_kbes": kuet_spec,
        "msa_moroccan": msa_moroccan_spec,
        "mder_ma": mder_ma_spec,
        "escorpus_pe": escorpus_pe_spec,
        "banspemo": banspemo_spec,
        "banglaser": banglaser_spec,
        "soreva": soreva_spec,
    }
    spec = factories[slug](stage_root)
    shutil.rmtree(stage_root / "wavs", ignore_errors=True)
    (stage_root / "wavs").mkdir()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(tqdm(executor.map(partial(make_record, stage_root=stage_root), spec.candidates),
                            total=len(spec.candidates), desc=slug, unit="file"))
    hours: dict[str, float] = {}
    for record in records:
        hours[record.language] = hours.setdefault(record.language, 0.0) + record.duration / 3_600
    manifest = DatasetManifest(
        dataset=DatasetRecord(name=spec.name, language_limits_hours=hours, source_url=spec.source_url),
        audio_files=records,
    )
    temporary = stage_root / "data.json.tmp"
    temporary.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8")
    temporary.replace(stage_root / "data.json")
    print(f"PREPARED {slug} records={len(records)} hours={sum(hours.values()):.6f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "slug",
        choices=("arabic_natural_audio", "kuet_kbes", "msa_moroccan", "mder_ma", "escorpus_pe", "banspemo", "banglaser", "soreva"),
    )
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    prepare(arguments.slug, arguments.workers)


if __name__ == "__main__":
    main()
