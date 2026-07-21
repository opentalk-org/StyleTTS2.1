import csv
import io
import json
import multiprocessing
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm


TARGET_SAMPLE_RATE = 24_000
TARGET_SECONDS = 50 * 60 * 60
SELECTION_SEED = 1337


@dataclass(frozen=True)
class WorkItem:
    archive: Path
    member: str
    output: Path
    metadata: dict[str, str]


@dataclass(frozen=True)
class PreparedItem:
    filename: str
    duration: float
    source_audio: dict[str, object]
    metadata: dict[str, str]


def find_metadata_member(archive: zipfile.ZipFile) -> str:
    matches = [name for name in archive.namelist() if name.endswith("pstn_train.csv")]
    assert len(matches) == 1, f"expected one pstn_train.csv, found {matches}"
    return matches[0]


def load_work(archive_path: Path, wav_dir: Path) -> list[WorkItem]:
    with zipfile.ZipFile(archive_path) as archive:
        csv_member = find_metadata_member(archive)
        csv_bytes = archive.read(csv_member)
        rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"))))
        member_names = set(archive.namelist())
        prefix = str(Path(csv_member).parent)
    random.Random(SELECTION_SEED).shuffle(rows)
    work = []
    for row in rows:
        member = str(Path(prefix) / row["filename"])
        assert member in member_names, f"missing archive member: {member}"
        work.append(WorkItem(archive_path, member, wav_dir / Path(member).name, row))
    return work


def prepare_item(item: WorkItem) -> PreparedItem:
    with zipfile.ZipFile(item.archive) as archive:
        source_bytes = archive.read(item.member)
    with sf.SoundFile(io.BytesIO(source_bytes)) as source:
        source_audio = {
            "sample_rate": source.samplerate,
            "channels": source.channels,
            "format": source.format,
            "subtype": source.subtype,
            "frames": source.frames,
        }
        samples = source.read(dtype="float32", always_2d=True).mean(axis=1)
    source_rate = int(source_audio["sample_rate"])
    divisor = np.gcd(source_rate, TARGET_SAMPLE_RATE)
    normalized = resample_poly(
        samples,
        TARGET_SAMPLE_RATE // divisor,
        source_rate // divisor,
    ).astype(np.float32)
    sf.write(item.output, normalized, TARGET_SAMPLE_RATE, subtype="PCM_24")
    return PreparedItem(
        filename=item.output.name,
        duration=len(normalized) / TARGET_SAMPLE_RATE,
        source_audio=source_audio,
        metadata=item.metadata,
    )


def audio_record(item: PreparedItem) -> dict[str, object]:
    source_id = Path(item.filename).stem
    return {
        "path": f"wavs/{item.filename}",
        "source_id": source_id,
        "duration": item.duration,
        "language": "en-us",
        "speaker_id": None,
        "style_prompt": None,
        "voice_prompt": None,
        "score": float(item.metadata["MOS"]),
        "accuracy": None,
        "segments": [
            {
                "start": 0.0,
                "end": item.duration,
                "text": "",
                "source": "dataset",
                "score": float(item.metadata["MOS"]),
                "accuracy": None,
                "alignment": [],
            }
        ],
        "metadata": {
            "source_dataset": "PSTN Speech Quality Corpus",
            "source_url": "https://challenge.blob.core.windows.net/pstn/train.zip",
            "source_audio": item.source_audio,
            "original": item.metadata,
            "rating_protocol": "MOS",
        },
    }


def main() -> None:
    dataset_dir = Path(__file__).resolve().parent.parent
    archive_path = dataset_dir / "tmp" / "train.zip"
    wav_dir = dataset_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    work = load_work(archive_path, wav_dir)
    records = []
    total_seconds = 0.0
    workers = min(12, multiprocessing.cpu_count())
    with multiprocessing.Pool(workers) as pool:
        results = pool.imap(prepare_item, work, chunksize=8)
        for item in tqdm(results, total=len(work), desc="PSTN normalize", unit="clip"):
            if total_seconds + item.duration <= TARGET_SECONDS:
                records.append(audio_record(item))
                total_seconds += item.duration
                continue
            (wav_dir / item.filename).unlink()
            if TARGET_SECONDS - total_seconds < 0.05:
                pool.terminate()
                break
    selected_names = {Path(record["path"]).name for record in records}
    for wav_path in wav_dir.glob("*.wav"):
        if wav_path.name not in selected_names:
            wav_path.unlink()
    payload = {
        "dataset": {
            "name": "PSTN Speech Quality Corpus",
            "language_limits_hours": {"en-us": 50.0},
            "source_url": "https://challenge.blob.core.windows.net/pstn/train.zip",
        },
        "audio_files": records,
        "summary": {
            "audio_count": len(records),
            "duration_seconds": total_seconds,
            "duration_hours": total_seconds / 3600,
        },
    }
    (dataset_dir / "data.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(dataset_dir / "tmp")
    (dataset_dir / "tmp").mkdir()


if __name__ == "__main__":
    main()
