import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from imports.stage1.common.transcribed_parquet import normalize_audio


SOURCE_URL = "https://huggingface.co/datasets/renumics/emodb"
EMOTIONS = ("anger", "boredom", "disgust", "fear", "happiness", "neutral", "sadness")
GENDERS = ("female", "male")


def speaker_id_from_path(audio_path: str) -> str:
    return f"emodb_{Path(audio_path).stem[:2]}"


def main() -> None:
    dataset_dir = Path(__file__).resolve().parent.parent
    shard = next((dataset_dir / "tmp/repo/data").glob("*.parquet"))
    rows = pq.read_table(shard).to_pylist()
    assert len(rows) == 535
    wav_dir = dataset_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(rows):
        source_id = f"emodb-{index:04d}"
        duration = normalize_audio(row["audio"]["bytes"], wav_dir / f"{source_id}.wav")
        emotion = EMOTIONS[int(row["emotion"])]
        records.append({
            "path": f"wavs/{source_id}.wav", "source_id": source_id, "duration": duration,
            "language": "de", "speaker_id": speaker_id_from_path(row["audio"]["path"]),
            "style_prompt": emotion, "voice_prompt": f"{GENDERS[int(row['gender'])]}, age {row['age']}",
            "score": None, "accuracy": None,
            "segments": [{"start": 0.0, "end": duration, "text": "", "source": "dataset", "score": None, "accuracy": None, "alignment": []}],
            "metadata": {"source_dataset": "EmoDB", "source_url": SOURCE_URL, "original_filename": row["audio"]["path"], "age": row["age"], "gender_code": row["gender"], "emotion_code": row["emotion"]},
        })
    total = sum(float(record["duration"]) for record in records)
    payload = {"dataset": {"name": "EmoDB", "language_limits_hours": {"de": total / 3600}, "source_url": SOURCE_URL}, "audio_files": records, "summary": {"audio_count": len(records), "duration_seconds": total, "duration_hours": total / 3600}}
    (dataset_dir / "data.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(dataset_dir / "tmp")
    (dataset_dir / "tmp").mkdir()


if __name__ == "__main__":
    main()
