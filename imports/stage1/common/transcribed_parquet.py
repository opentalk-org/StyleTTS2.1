import io
import json
import multiprocessing
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm


TARGET_SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    slug: str
    pattern: str
    expected_shards: int
    expected_records: int
    source_url: str
    speaker_prefix: str | None


@dataclass(frozen=True)
class ShardTask:
    path: Path
    wav_dir: Path
    config: DatasetConfig


def normalized_text(value: object) -> str | None:
    text = str(value)
    return None if text == "null" else text


def normalize_audio(audio_bytes: bytes, output: Path) -> float:
    with sf.SoundFile(io.BytesIO(audio_bytes)) as source:
        sample_rate = source.samplerate
        samples = source.read(dtype="float32", always_2d=True).mean(axis=1)
    divisor = np.gcd(sample_rate, TARGET_SAMPLE_RATE)
    normalized = resample_poly(samples, TARGET_SAMPLE_RATE // divisor, sample_rate // divisor).astype(np.float32)
    sf.write(output, normalized, TARGET_SAMPLE_RATE, subtype="PCM_24")
    return len(normalized) / TARGET_SAMPLE_RATE


def build_record(row: dict[str, object], task: ShardTask, index: int, duration: float) -> dict[str, object]:
    source_id = f"{task.path.stem}-{index:06d}"
    raw = dict(row)
    audio = dict(raw["audio"])
    raw["audio"] = {"path": audio["path"]}
    intensity = normalized_text(row["emotion_intensity"])
    emotion = str(row["emotion"])
    style_prompt = f"{intensity} intensity {emotion}" if intensity else emotion
    gender = normalized_text(row["gender"])
    source_speaker_id = str(row["speaker_id"])
    speaker_id = (
        f"{task.config.speaker_prefix}_{source_speaker_id}"
        if task.config.speaker_prefix is not None
        else source_speaker_id
    )
    return {
        "path": f"wavs/{source_id}.wav", "source_id": source_id,
        "duration": duration, "language": "en", "speaker_id": speaker_id,
        "style_prompt": style_prompt, "voice_prompt": gender,
        "score": None, "accuracy": None,
        "segments": [{
            "start": 0.0, "end": duration, "text": str(row["transcript"]),
            "source": "dataset", "score": None, "accuracy": None, "alignment": [],
        }],
        "metadata": {
            "source_dataset": task.config.name, "source_url": task.config.source_url,
            "source_shard": task.path.name, "source_row_index": index, "source_row": raw,
        },
    }


def prepare_shard(task: ShardTask) -> list[dict[str, object]]:
    records = []
    for index, row in enumerate(pq.read_table(task.path).to_pylist()):
        source_id = f"{task.path.stem}-{index:06d}"
        duration = normalize_audio(row["audio"]["bytes"], task.wav_dir / f"{source_id}.wav")
        records.append(build_record(row, task, index, duration))
    return records


def prepare_dataset(dataset_dir: Path, config: DatasetConfig) -> None:
    shards = sorted((dataset_dir / "tmp" / "repo" / "data").glob(config.pattern))
    assert len(shards) == config.expected_shards, f"expected {config.expected_shards} shards, found {len(shards)}"
    wav_dir = dataset_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    tasks = [ShardTask(path, wav_dir, config) for path in shards]
    records = []
    with multiprocessing.Pool(min(8, multiprocessing.cpu_count())) as pool:
        for result in tqdm(pool.imap(prepare_shard, tasks), total=len(tasks), desc=config.name):
            records.extend(result)
    assert len(records) == config.expected_records, f"expected {config.expected_records} clips, found {len(records)}"
    records.sort(key=lambda record: str(record["source_id"]))
    total_seconds = sum(float(record["duration"]) for record in records)
    payload = {
        "dataset": {"name": config.name, "language_limits_hours": {"en": total_seconds / 3600}, "source_url": config.source_url},
        "audio_files": records,
        "summary": {"audio_count": len(records), "duration_seconds": total_seconds, "duration_hours": total_seconds / 3600},
    }
    (dataset_dir / "data.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(dataset_dir / "tmp")
    (dataset_dir / "tmp").mkdir()
