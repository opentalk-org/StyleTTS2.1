from __future__ import annotations

import base64
from dataclasses import dataclass
import importlib
import io
import logging
from pathlib import Path
from typing import Any

import yaml

from runner.nodes.synthesis.styletts_runtime.helpers import (
    compute_style_from_wave,
    length_to_mask,
    phonemize_line,
    recursive_munch,
    tokenize_ipa,
    training_import_context,
)

logger = logging.getLogger(__name__)


@dataclass
class StyleTtsSynthesisRuntime:
    model: Any
    sampler: Any
    model_params: Any
    symbols_list: list[str]
    device: Any


def load_synthesis_runtime(payload: dict[str, Any]) -> StyleTtsSynthesisRuntime:
    bundle_root = Path(payload["bundle_root"])
    weights_path = Path(payload["weights_path"])
    symbols_str = str(payload["symbols"])
    arch_path = bundle_root / "config.yml"
    if not arch_path.is_file():
        raise ValueError("checkpoint_styletts_config_missing")
    arch = yaml.safe_load(arch_path.read_text(encoding="utf-8"))
    model_params = recursive_munch(arch["model_params"])
    model_params.n_token = len(symbols_str)
    torch = importlib.import_module("torch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with training_import_context():
        sampler_module = importlib.import_module("modules.diffusion.sampler")
        loading_module = importlib.import_module("loading")
        text_aligner = loading_module.load_ASR_models(payload["asr_path"], payload["asr_config"])
        pitch_extractor = loading_module.load_F0_models(payload["f0_path"])
        plbert = loading_module.load_plbert(payload["plbert_path"], payload["plbert_config"])
        model = loading_module.build_model(model_params, text_aligner, pitch_extractor, plbert)
        try:
            model, _ = loading_module.load_checkpoint(model, None, str(weights_path), load_only_params=True)
        except Exception as exc:
            raise ValueError("finetune_test_weights_incompatible") from exc
        _prepare_model(model, device)
        sampler = _build_sampler(sampler_module, model)

    return StyleTtsSynthesisRuntime(model, sampler, model_params, list(symbols_str), device)


def run_synthesis_with_runtime(runtime: StyleTtsSynthesisRuntime, payload: dict[str, Any]) -> Path:
    text = str(payload["text"])
    work_dir = Path(payload["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    out_wav = work_dir / str(payload["output_filename"])
    ipa = phonemize_line(
        text,
        phoneme_language=str(payload["phoneme_language"]),
        phoneme_tie=bool(payload["phoneme_tie"]),
    )
    tok = tokenize_ipa(ipa, runtime.symbols_list, runtime.device)
    ref_s = _decode_style_reference(payload["style_reference"], runtime.model, runtime.device)
    wav = _synthesize_wave(
        runtime=runtime,
        tok=tok,
        ref_s=ref_s,
        diffusion_steps=int(payload["diffusion_steps"]),
        embedding_scale=float(payload["embedding_scale"]),
        style_alpha=float(payload["style_mix_alpha"]),
        style_beta=float(payload["style_mix_beta"]),
    )
    soundfile = importlib.import_module("soundfile")
    soundfile.write(str(out_wav), wav, 24000)
    logger.info("styletts synthesis wrote %s samples", wav.size)
    return out_wav


def _decode_style_reference(style_ref: dict[str, Any], model: Any, device: Any) -> Any:
    if style_ref["kind"] == "path":
        librosa = importlib.import_module("librosa")
        wave, sr = librosa.load(str(style_ref["path"]), sr=None, mono=True)
        return compute_style_from_wave(model, wave, sr=int(sr), device=device)
    if style_ref["kind"] == "wav_base64":
        numpy = importlib.import_module("numpy")
        soundfile = importlib.import_module("soundfile")
        raw = base64.b64decode(str(style_ref["data"]))
        wave, sr = soundfile.read(io.BytesIO(raw))
        if wave.ndim > 1:
            wave = wave[:, 0]
        return compute_style_from_wave(model, wave.astype(numpy.float32), sr=int(sr), device=device)
    raise ValueError("finetune_test_style_reference_invalid")


def _synthesize_wave(
    *,
    runtime: StyleTtsSynthesisRuntime,
    tok: Any,
    ref_s: Any,
    diffusion_steps: int,
    embedding_scale: float,
    style_alpha: float,
    style_beta: float,
) -> Any:
    torch = importlib.import_module("torch")
    with torch.no_grad():
        input_lengths = torch.LongTensor([tok.shape[-1]]).to(runtime.device)
        text_mask = length_to_mask(input_lengths).to(runtime.device)
        t_en = runtime.model.text_encoder(tok, input_lengths, text_mask)
        bert_dur = runtime.model.bert(tok, attention_mask=(~text_mask).int())
        d_en = runtime.model.bert_encoder(bert_dur).transpose(-1, -2)
        s_pred = runtime.sampler(
            noise=torch.randn((1, 256), device=runtime.device).unsqueeze(1),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s,
            num_steps=diffusion_steps,
        ).squeeze(1)
        ref_blended, s_blended = _blend_style(s_pred, ref_s, style_alpha, style_beta)
        pred_aln_trg, d = _predict_alignment(runtime, d_en, s_blended, input_lengths, text_mask, tok.shape[-1])
        en = _shift_hifigan(runtime, d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(runtime.device))
        f0_pred, n_pred = runtime.model.predictor.F0Ntrain(en, s_blended)
        asr = _shift_hifigan(runtime, t_en @ pred_aln_trg.unsqueeze(0).to(runtime.device))
        out = runtime.model.decoder(asr, f0_pred, n_pred, ref_blended.squeeze().unsqueeze(0))
    return _normalize_wave(out.squeeze().detach().cpu().numpy())


def _predict_alignment(runtime, d_en, s_blended, input_lengths, text_mask, token_count: int):
    torch = importlib.import_module("torch")
    d = runtime.model.predictor.text_encoder(d_en, s_blended, input_lengths, text_mask)
    x, _ = runtime.model.predictor.lstm(d)
    duration = torch.sigmoid(runtime.model.predictor.duration_proj(x)).sum(axis=-1)
    pred_dur = torch.round(duration.squeeze()).clamp(min=1)
    frame_count = int(pred_dur.sum().item())
    pred_aln_trg = torch.zeros(token_count, frame_count, device=runtime.device)
    cursor = 0
    for index in range(token_count):
        width = int(pred_dur[index].item())
        pred_aln_trg[index, cursor : cursor + width] = 1
        cursor += width
    return pred_aln_trg, d


def _blend_style(s_pred: Any, ref_s: Any, style_alpha: float, style_beta: float):
    s_part = s_pred[:, 128:]
    ref_part = s_pred[:, :128]
    ref_blended = style_alpha * ref_part + (1.0 - style_alpha) * ref_s[:, :128]
    s_blended = style_beta * s_part + (1.0 - style_beta) * ref_s[:, 128:]
    return ref_blended, s_blended


def _shift_hifigan(runtime: StyleTtsSynthesisRuntime, value: Any) -> Any:
    torch = importlib.import_module("torch")
    if runtime.model_params.decoder.type != "hifigan":
        return value
    shifted = torch.zeros_like(value)
    shifted[:, :, 0] = value[:, :, 0]
    shifted[:, :, 1:] = value[:, :, 0:-1]
    return shifted


def _normalize_wave(wav: Any) -> Any:
    numpy = importlib.import_module("numpy")
    if wav.size > 50:
        wav = wav[..., :-50]
    peak = float(numpy.max(numpy.abs(wav))) if wav.size else 0.0
    if peak > 1e-6:
        wav = wav * (0.99 / peak)
    return wav.astype(numpy.float32)


def _prepare_model(model: Any, device: Any) -> None:
    for item in model:
        model[item].eval()
        model[item].to(device)


def _build_sampler(sampler_module: Any, model: Any) -> Any:
    sampler = sampler_module.DiffusionSampler(
        model.diffusion.diffusion,
        sampler=sampler_module.ADPM2Sampler(),
        sigma_schedule=sampler_module.KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )
    sampler.eval()
    return sampler
