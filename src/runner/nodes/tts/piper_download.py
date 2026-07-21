from __future__ import annotations

import json
from pathlib import Path

from runner.nodes.assets.catalog_runtime.persistence import ensure_checkpoint_bundle
from runner.nodes.assets.catalog_runtime.types import CatalogFile, CheckpointSpec, CheckpointType
from runner.nodes.assets.checkpoints import resolve_checkpoint_ref
from runner.nodes.models import CheckpointRef
from runner.nodes.tts.piper_catalog import PIPER_VOICES_BASE_URL, PiperVoiceEntry
from shared.db import database_session
from shared.db.assets import crud as asset_crud


def download_piper_voice(voice: PiperVoiceEntry) -> CheckpointRef:
    spec = CheckpointSpec(
        key=f"piper:{voice.voice_id}",
        name=f"Piper · {voice.language.name_english} · {voice.name} · {voice.quality}",
        type_=CheckpointType.PIPER,
        files=(
            CatalogFile(f"{PIPER_VOICES_BASE_URL}/{voice.model_path}", Path(voice.model_path).name),
            CatalogFile(f"{PIPER_VOICES_BASE_URL}/{voice.config_path}", Path(voice.config_path).name),
        ),
        metadata={
            "source": "rhasspy/piper-voices",
            "voice_id": voice.voice_id,
            "language": voice.language.model_dump(),
            "quality": voice.quality,
            "num_speakers": voice.num_speakers,
            "speaker_id_map": voice.speaker_id_map,
        },
        is_valid=_piper_bundle_valid,
        metadata_from_path=_sample_rate_metadata,
    )
    checkpoint, _skipped = ensure_checkpoint_bundle(spec)
    return resolve_checkpoint_ref(str(checkpoint.id), CheckpointType.PIPER.value)


def resolve_downloaded_piper_voice(voice_id: str) -> CheckpointRef:
    catalog_key = f"piper:{voice_id}"
    with database_session() as session:
        matches = [
            checkpoint for checkpoint in asset_crud.list_checkpoints(session)
            if checkpoint.type_ == CheckpointType.PIPER.value
            and checkpoint.metadata_["catalog_key"] == catalog_key
        ]
    if len(matches) != 1:
        raise ValueError(f"piper_voice_not_downloaded:{voice_id}")
    return resolve_checkpoint_ref(str(matches[0].id), CheckpointType.PIPER.value)


def _piper_bundle_valid(folder: Path) -> bool:
    return len(tuple(folder.glob("*.onnx"))) == 1 and len(tuple(folder.glob("*.onnx.json"))) == 1


def _sample_rate_metadata(folder: Path) -> dict[str, int]:
    config_path, = tuple(folder.glob("*.onnx.json"))
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return {"sample_rate": int(payload["audio"]["sample_rate"])}
