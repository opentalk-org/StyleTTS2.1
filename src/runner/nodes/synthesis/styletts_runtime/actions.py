from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from runner.nodes.models import CheckpointRef
from runner.nodes.synthesis.styletts_runtime.checkpoints import (
    resolve_asr_payload,
    resolve_f0_path,
    resolve_main_checkpoint,
    resolve_plbert_payload,
    resolve_symbols,
    resolve_weights_path,
)
from shared.db import database_session
from shared.db.audio import crud as audio_crud


class StyleTtsRequestSettings(BaseModel):
    weights_file: str = ""
    diffusion_steps: int
    embedding_scale: float
    phoneme_language: str
    phoneme_tie: bool
    alpha: float
    beta: float
    output_name: str
    asr_checkpoint_id: UUID | None
    f0_checkpoint_id: UUID | None
    f0_inner_filename: str
    plbert_checkpoint_id: UUID | None


def build_styletts_payload(
    *,
    checkpoint: CheckpointRef,
    prompt_text: dict[str, Any],
    style_reference: dict[str, Any],
    settings: StyleTtsRequestSettings,
    work_dir: Path,
    output_filename: str,
) -> dict[str, Any]:
    main = resolve_main_checkpoint(checkpoint)
    symbols = resolve_symbols(main.metadata)
    weights_path = resolve_weights_path(main.root, settings.weights_file)
    asr_config, asr_path = resolve_asr_payload(settings.asr_checkpoint_id, symbols)
    f0_path = resolve_f0_path(settings.f0_checkpoint_id, settings.f0_inner_filename)
    plbert_config, plbert_path = resolve_plbert_payload(settings.plbert_checkpoint_id, symbols)
    return {
        "bundle_root": str(main.root.resolve()),
        "weights_path": str(weights_path.resolve()),
        "symbols": symbols,
        "text": _prompt_text(prompt_text),
        "diffusion_steps": settings.diffusion_steps,
        "embedding_scale": settings.embedding_scale,
        "phoneme_language": settings.phoneme_language,
        "phoneme_tie": settings.phoneme_tie,
        "alpha": _mix_value(style_reference, "alpha", settings.alpha),
        "beta": _mix_value(style_reference, "beta", settings.beta),
        "asr_config": asr_config,
        "asr_path": asr_path,
        "f0_path": f0_path,
        "plbert_config": plbert_config,
        "plbert_path": plbert_path,
        "work_dir": str(work_dir.resolve()),
        "style_reference": materialize_style_reference(style_reference, work_dir),
        "output_filename": output_filename,
    }


def synthesize_to_wav_bytes(
    *,
    runtime: Any | None,
    payload: dict[str, Any],
) -> bytes:
    from runner.nodes.synthesis.styletts_runtime.runtime import load_synthesis_runtime, run_synthesis_with_runtime

    if runtime is None:
        runtime = load_synthesis_runtime(payload)
    out_path = run_synthesis_with_runtime(runtime, payload)
    if not out_path.is_file():
        raise ValueError("finetune_test_synthesize_failed")
    wav_bytes = out_path.read_bytes()
    if not wav_bytes:
        raise ValueError("finetune_test_synthesize_empty")
    return wav_bytes


def materialize_style_reference(reference: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    kind = reference["kind"]
    if kind == "audio_file":
        audio_file_id = UUID(str(reference["audio_file_id"]))
        with database_session() as session:
            wav_bytes = audio_crud.read_audio_file(session, audio_file_id)
        reference_path = work_dir / f"style_reference_{audio_file_id}.wav"
        reference_path.write_bytes(wav_bytes)
        return _without_large_fields({**reference, "kind": "path", "path": str(reference_path.resolve())})
    if kind == "wav_base64":
        return {"kind": "wav_base64", "data": str(reference["wav_base64"])}
    if kind == "path":
        return {"kind": "path", "path": str(reference["path"])}
    raise ValueError("finetune_test_style_reference_invalid")


def temporary_synthesis_dir():
    return tempfile.TemporaryDirectory(prefix="runflow-styletts-synthesis-")


def load_synthesis_runtime(payload: dict[str, Any]) -> Any:
    from runner.nodes.synthesis.styletts_runtime.runtime import load_synthesis_runtime as load_runtime

    return load_runtime(payload)


def _prompt_text(value: dict[str, Any]) -> str:
    if "text" in value:
        return str(value["text"])
    if "settings" in value:
        return str(value["settings"]["text"])
    return str(value)


def _mix_value(reference: dict[str, Any], name: str, default: float) -> float:
    if name in reference:
        return float(reference[name])
    return default


def _without_large_fields(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"wav_base64"}}
