from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from runner.nodes.models import CheckpointRef
from runner.nodes.synthesis.styletts_runtime.checkpoints import (
    latest_weight,
    resolve_asr_payload,
    resolve_f0_path,
    resolve_main_checkpoint,
    resolve_plbert_payload,
    resolve_symbols,
)
from runner.nodes.synthesis.styletts_runtime.runtime import (
    load_synthesis_runtime,
    run_synthesis_with_runtime,
)


class StyleTtsRequestSettings(BaseModel):
    diffusion_steps: int
    embedding_scale: float
    phoneme_language: str
    phoneme_tie: bool
    alpha: float
    beta: float
    asr_checkpoint_id: UUID | None
    f0_checkpoint_id: UUID | None
    f0_inner_filename: str
    plbert_checkpoint_id: UUID | None


@dataclass(frozen=True)
class StyleTtsPayloadRequest:
    prompt_text: dict[str, Any]
    style_reference: dict[str, Any]
    output_filename: str


def build_styletts_payload(
    *,
    checkpoint: CheckpointRef,
    prompt_text: dict[str, Any],
    style_reference: dict[str, Any],
    settings: StyleTtsRequestSettings,
    work_dir: Path,
    output_filename: str,
    audio_data: dict[UUID, bytes],
) -> dict[str, Any]:
    return build_styletts_payloads(
        checkpoint=checkpoint,
        requests=[StyleTtsPayloadRequest(prompt_text, style_reference, output_filename)],
        settings=settings,
        work_dir=work_dir,
        audio_data=audio_data,
    )[0]


def build_styletts_payloads(
    *,
    checkpoint: CheckpointRef,
    requests: list[StyleTtsPayloadRequest],
    settings: StyleTtsRequestSettings,
    work_dir: Path,
    audio_data: dict[UUID, bytes],
) -> list[dict[str, Any]]:
    main = resolve_main_checkpoint(checkpoint)
    symbols = resolve_symbols(main.metadata)
    weights_path = latest_weight(main.root)
    asr_config, asr_path = resolve_asr_payload(settings.asr_checkpoint_id, symbols)
    f0_path = resolve_f0_path(settings.f0_checkpoint_id, settings.f0_inner_filename)
    plbert_config, plbert_path = resolve_plbert_payload(settings.plbert_checkpoint_id, symbols)
    common = {
        "bundle_root": str(main.root.resolve()),
        "weights_path": str(weights_path.resolve()),
        "symbols": symbols,
        "diffusion_steps": settings.diffusion_steps,
        "embedding_scale": settings.embedding_scale,
        "phoneme_tie": settings.phoneme_tie,
        "asr_config": asr_config,
        "asr_path": asr_path,
        "f0_path": f0_path,
        "plbert_config": plbert_config,
        "plbert_path": plbert_path,
        "work_dir": str(work_dir.resolve()),
    }
    return [
        {
            **common,
            "text": _prompt_text(request.prompt_text),
            "phoneme_language": _phoneme_language(
                request.prompt_text,
                settings.phoneme_language,
            ),
            "alpha": _mix_value(request.style_reference, "alpha", settings.alpha),
            "beta": _mix_value(request.style_reference, "beta", settings.beta),
            "style_reference": materialize_style_reference(
                request.style_reference,
                work_dir,
                audio_data,
            ),
            "output_filename": request.output_filename,
        }
        for request in requests
    ]


def synthesize_to_wav_bytes(
    *,
    runtime: Any | None,
    payload: dict[str, Any],
) -> bytes:
    if runtime is None:
        runtime = load_synthesis_runtime(payload)
    out_path = run_synthesis_with_runtime(runtime, payload)
    if not out_path.is_file():
        raise ValueError("finetune_test_synthesize_failed")
    wav_bytes = out_path.read_bytes()
    if not wav_bytes:
        raise ValueError("finetune_test_synthesize_empty")
    return wav_bytes


def synthesize_to_wav_bytes_batch(
    payloads: list[dict[str, Any]],
    check_cancel: Callable[[], None],
    runtime: Any | None = None,
) -> tuple[Any | None, list[bytes]]:
    if not payloads:
        return runtime, []
    resolved_runtime = runtime if runtime is not None else load_synthesis_runtime(payloads[0])
    outputs = []
    for payload in payloads:
        check_cancel()
        outputs.append(synthesize_to_wav_bytes(runtime=resolved_runtime, payload=payload))
    return resolved_runtime, outputs


def materialize_style_reference(
    reference: dict[str, Any],
    work_dir: Path,
    audio_data: dict[UUID, bytes],
) -> dict[str, Any]:
    kind = reference["kind"]
    if kind == "audio_file":
        audio_file_id = UUID(str(reference["audio_file_id"]))
        reference_path = work_dir / f"style_reference_{audio_file_id}.wav"
        reference_path.write_bytes(audio_data[audio_file_id])
        return _without_large_fields({**reference, "kind": "path", "path": str(reference_path.resolve())})
    if kind == "wav_base64":
        return {"kind": "wav_base64", "data": str(reference["wav_base64"])}
    if kind == "path":
        return {"kind": "path", "path": str(reference["path"])}
    raise ValueError("finetune_test_style_reference_invalid")


def temporary_synthesis_dir():
    return tempfile.TemporaryDirectory(prefix="runflow-styletts-synthesis-")


def _prompt_text(value: dict[str, Any]) -> str:
    if "text" in value:
        return str(value["text"])
    if "settings" in value:
        return str(value["settings"]["text"])
    return str(value)


def _phoneme_language(prompt_text: dict[str, Any], default_language: str) -> str:
    if "settings" in prompt_text:
        settings = prompt_text["settings"]
        assert isinstance(settings, dict), "prompt settings must be a dict"
        if "language" in settings:
            language = str(settings["language"]).strip()
            if not language:
                raise ValueError("finetune_test_synth_language_empty")
            return language
    return default_language


def _mix_value(reference: dict[str, Any], name: str, default: float) -> float:
    if name in reference:
        return float(reference[name])
    return default


def _without_large_fields(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"wav_base64"}}
