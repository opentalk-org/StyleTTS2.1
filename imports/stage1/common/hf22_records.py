from pathlib import Path

from imports.stage1.common.hf22_catalog import LocaleSpec
from imports.stage1.common.schema import AudioRecord, SegmentRecord
from imports.stage1_clean_cv26_quotes import remove_majority_quotes


def build_audio_record(
    spec: LocaleSpec,
    split: str,
    row: dict[str, str],
    destination: Path,
    duration: float,
) -> AudioRecord:
    raw_row = dict(row)
    cleaned_text = remove_majority_quotes(raw_row["sentence"])
    publisher_row = dict(raw_row)
    publisher_row["sentence"] = cleaned_text
    metadata: dict[str, object] = {
        "source": "Mozilla Common Voice Corpus 22.0",
        "release": "22.0",
        "source_url": (
            "https://huggingface.co/datasets/"
            "fsicoli/common_voice_22_0"
        ),
        "hf_locale": spec.hf_locale,
        "split": split,
        "publisher_row": publisher_row,
    }
    if cleaned_text != raw_row["sentence"]:
        metadata["publisher_row_original"] = raw_row
    return AudioRecord(
        path=f"wavs/{destination.name}",
        source_id=f"cv22:{spec.language}:{raw_row['path']}",
        duration=duration,
        language=spec.language,
        speaker_id=f"cv22:{raw_row['client_id']}",
        style_prompt=None,
        voice_prompt=_voice_prompt(raw_row),
        score=None,
        accuracy=None,
        segments=[
            SegmentRecord(
                start=0.0,
                end=duration,
                text=cleaned_text,
                source="dataset",
                score=None,
                accuracy=None,
                alignment=[],
            )
        ],
        metadata=metadata,
    )


def _voice_prompt(row: dict[str, str]) -> str | None:
    fields = ["age", "gender"]
    fields.append("accents" if "accents" in row else "accent")
    values = [row[field].strip() for field in fields if row[field].strip()]
    return ", ".join(values) if values else None
