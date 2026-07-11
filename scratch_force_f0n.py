"""Teacher-forcing diagnostic: synthesize with predicted vs. ground-truth F0/N.

Run the normal way (AGENTS.md): nix develop --command python scratch_force_f0n.py

Uses the already-extracted epoch-10 finetune bundle on disk and the reference
recording fetched from the backend; no DB/S3 access needed. The reference clip is
both the style reference and the source of ground-truth F0/N curves, so injecting
them isolates whether flat prosody comes from the F0/N predictor over-smoothing.
"""
from pathlib import Path
import sys

import librosa
import soundfile as sf
import torch
import torch.nn.functional as Fnn

REPO = Path("/workspace/styletts_studio_v2")
sys.path.insert(0, str(REPO / "src"))

from runner.nodes.synthesis.styletts_runtime.checkpoints import (
    latest_weight,
    resolve_asr_payload,
    resolve_f0_path,
    resolve_plbert_payload,
    resolve_symbols,
)
from runner.nodes.synthesis.styletts_runtime.helpers import (
    compute_style_from_wave,
    length_to_mask,
    phonemize_line,
    tokenize_ipa,
    _preprocess_wave,
)
from runner.nodes.synthesis.styletts_runtime.runtime import (
    load_synthesis_runtime,
    _blend_style,
    _normalize_wave,
    _predict_alignment,
    _shift_hifigan,
)

BUNDLE = REPO / ".cache/runflow/assets/checkpoints/bab1075c8b1249b4a67d2c6c0d403396af134e16438198bd17aef92044fc1fab"
OUT = REPO / "outputs/f0n_forcing"
REF_WAV = OUT / "reference_gt.wav"
TEXT = (
    "Jakby wizorunkowo, jeżeli on ryzykuje na tyle, żeby przyjechać do kraju, "
    "który prowadzi taką wojnę podczas wojny, no to więc on jakby już nie przewiduje, "
    "że Ukraina może przegrać w całości tę wojnę."
)

# Same defaults the StyleTtsSynthesis node uses; aux ids None -> weights restored
# from the main checkpoint. Symbols fall back to the 178-symbol default set, which
# matches this checkpoint's n_token=178.
symbols = resolve_symbols({})
asr_config, asr_path = resolve_asr_payload(None, symbols)
f0_path = resolve_f0_path(None, "")
plbert_config, plbert_path = resolve_plbert_payload(None, symbols)

payload = {
    "bundle_root": str(BUNDLE.resolve()),
    "weights_path": str(latest_weight(BUNDLE).resolve()),
    "symbols": symbols,
    "asr_config": asr_config,
    "asr_path": asr_path,
    "f0_path": f0_path,
    "plbert_config": plbert_config,
    "plbert_path": plbert_path,
}

runtime = load_synthesis_runtime(payload)
device = runtime.device
model = runtime.model

# Ground-truth prosody from the reference recording. log_norm energy is
# log(||exp(mel*std+mean)||_over_mel_bins) with the training mean=-4, std=4.
wave, _ = librosa.load(str(REF_WAV), sr=24000, mono=True)
mel = _preprocess_wave(wave).to(device)  # (1, 80, T_ref)
with torch.no_grad():
    f0_real, _, _ = model.pitch_extractor(mel.unsqueeze(1))
    F0_gt = f0_real.squeeze(-1)  # (1, T_ref)
    N_gt = torch.log(torch.exp(mel.unsqueeze(1) * 4.0 - 4.0).norm(dim=2)).squeeze(1)  # (1, T_ref)

ref_s = compute_style_from_wave(model, wave, sr=24000, device=device)

ipa = phonemize_line(TEXT, phoneme_language="pl", phoneme_tie=True)
tok = tokenize_ipa(ipa, runtime.symbols_list, device)

torch.manual_seed(0)  # fix the sampled style/duration so variants differ only in F0/N
with torch.no_grad():
    input_lengths = torch.LongTensor([tok.shape[-1]]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)
    t_en = model.text_encoder(tok, input_lengths, text_mask)
    bert_dur = model.bert(tok, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s_pred = runtime.sampler(
        noise=torch.randn((1, 256), device=device).unsqueeze(1),
        embedding=bert_dur,
        embedding_scale=1.0,
        features=ref_s,
        num_steps=5,
    ).squeeze(1)
    ref_blended, s_blended = _blend_style(s_pred, ref_s, 0.7, 0.3)
    pred_aln_trg, d = _predict_alignment(runtime, d_en, s_blended, input_lengths, text_mask, tok.shape[-1])
    en = _shift_hifigan(runtime, d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
    f0_pred, n_pred = model.predictor.F0Ntrain(en, s_blended)
    asr = _shift_hifigan(runtime, t_en @ pred_aln_trg.unsqueeze(0).to(device))

    # GT curves follow the reference's true duration; the synthesized utterance has
    # its own frame count from the duration predictor, so stretch GT to match.
    T = f0_pred.shape[-1]
    F0_gt_T = Fnn.interpolate(F0_gt.unsqueeze(1), size=T, mode="linear", align_corners=False).squeeze(1)
    N_gt_T = Fnn.interpolate(N_gt.unsqueeze(1), size=T, mode="linear", align_corners=False).squeeze(1)
    print(f"frames: reference={F0_gt.shape[-1]}  synthesized={T}")
    print(f"F0 std  pred={f0_pred.std().item():.4f}  gt(resampled)={F0_gt_T.std().item():.4f}")
    print(f"N  std  pred={n_pred.std().item():.4f}  gt(resampled)={N_gt_T.std().item():.4f}")

    # Voicing (F0<=10 -> unvoiced, matching SourceModuleHnNSF voiced_threshod=10)
    # vs voiced-region pitch dynamics. Use raw curves (no resample) so linear
    # interpolation doesn't smear the U/V boundary.
    thr = 10.0
    fp = f0_pred.squeeze()
    fg = F0_gt.squeeze()
    print("\nvoicing vs voiced-pitch (raw, alignment-free distributions):")
    print(f"unvoiced frac  pred={(fp <= thr).float().mean().item():.3f}  gt={(fg <= thr).float().mean().item():.3f}")
    print(f"voiced-F0 std  pred={fp[fp > thr].std().item():.2f}  gt={fg[fg > thr].std().item():.2f}")
    print(f"voiced-F0 max  pred={fp[fp > thr].max().item():.1f}  gt={fg[fg > thr].max().item():.1f}")

    tail = slice(int(T * 0.6), T)  # last 40% of the synthesized timeline
    print(f"\nlast 40% ({T - int(T * 0.6)} frames):")
    print(f"F0 std  pred={f0_pred[:, tail].std().item():.4f}  gt(resampled)={F0_gt_T[:, tail].std().item():.4f}")
    print(f"N  std  pred={n_pred[:, tail].std().item():.4f}  gt(resampled)={N_gt_T[:, tail].std().item():.4f}")
    print(f"F0 mean pred={f0_pred[:, tail].mean().item():.2f}  gt(resampled)={F0_gt_T[:, tail].mean().item():.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps = 24000 / 300  # hop_length=300 -> 80 frames/sec
    ts = torch.arange(T, device=device).cpu().numpy() / fps
    f0_p = f0_pred.squeeze().cpu().numpy()
    f0_g = F0_gt_T.squeeze().cpu().numpy()
    n_p = n_pred.squeeze().cpu().numpy()
    n_g = N_gt_T.squeeze().cpu().numpy()
    tail_t = int(T * 0.6) / fps

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    ax0.plot(ts, f0_g, color="tab:green", lw=1.0, label="ground truth")
    ax0.plot(ts, f0_p, color="tab:red", lw=1.0, label="predicted")
    ax0.axvspan(tail_t, ts[-1], color="grey", alpha=0.10, label="last 40%")
    ax0.set_ylabel("F0 (pitch)")
    ax0.legend(loc="upper right")
    ax0.set_title("StyleTTS2 epoch 10 — predicted vs ground-truth prosody")
    ax1.plot(ts, n_g, color="tab:green", lw=1.0)
    ax1.plot(ts, n_p, color="tab:red", lw=1.0)
    ax1.axvspan(tail_t, ts[-1], color="grey", alpha=0.10)
    ax1.set_ylabel("N (energy)")
    ax1.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(str(OUT / "prosody_curves.png"), dpi=110)
    print(f"wrote prosody_curves.png")

    style = ref_blended.squeeze().unsqueeze(0)
    variants = {
        "baseline_pred": (f0_pred, n_pred),
        "force_f0": (F0_gt_T, n_pred),
        "force_n": (f0_pred, N_gt_T),
        "force_f0_n": (F0_gt_T, N_gt_T),
    }
    for name, (f0, n) in variants.items():
        out = model.decoder(asr, f0, n, style)
        wav = _normalize_wave(out.squeeze().detach().cpu().numpy())
        path = OUT / f"{name}.wav"
        sf.write(str(path), wav, 24000)
        print(f"wrote {path.name}  {len(wav) / 24000:.2f}s")

print("DONE")
