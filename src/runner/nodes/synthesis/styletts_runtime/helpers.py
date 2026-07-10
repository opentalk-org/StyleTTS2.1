from __future__ import annotations

import sys
import importlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from runner.nodes.text.runtime.symbols import TextCleaner


MEL_MEAN = -4.0
MEL_STD = 4.0
TRAINING_ROOT = Path(__file__).resolve().parents[2] / "training" / "styletts" / "finetune" / "training"
MEL: Any | None = None


@contextmanager
def training_import_context():
    prev_path = list(sys.path)
    sys.path.insert(0, str(TRAINING_ROOT.resolve()))
    try:
        yield
    finally:
        sys.path[:] = prev_path


def length_to_mask(lengths: Any) -> Any:
    torch = importlib.import_module("torch")
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def recursive_munch(value: Any) -> Any:
    munch = importlib.import_module("munch")
    if isinstance(value, dict):
        return munch.Munch((key, recursive_munch(item)) for key, item in value.items())
    if isinstance(value, list):
        return [recursive_munch(item) for item in value]
    return value


def compute_style_from_wave(model: Any, wave: Any, *, sr: int, device: Any) -> Any:
    librosa = importlib.import_module("librosa")
    torch = importlib.import_module("torch")
    if sr != 24000:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    mel_tensor = _preprocess_wave(audio).to(device)
    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
    return torch.cat([ref_s, ref_p], dim=1)


def phonemize_line(text: str, *, phoneme_language: str, phoneme_tie: bool) -> str:
    phonemizer = importlib.import_module("phonemizer")
    backend = phonemizer.backend.EspeakBackend(language=phoneme_language, preserve_punctuation=True, with_stress=True)
    ipa = backend.phonemize([text.strip()])[0].strip()
    if not ipa:
        raise ValueError("finetune_test_synth_phonemize_empty")
    return ipa


def tokenize_ipa(ipa: str, symbols_list: list[str], device: Any) -> Any:
    torch = importlib.import_module("torch")
    text_cleaner = TextCleaner(symbols_list)
    tokens = text_cleaner(ipa)
    tokens.insert(0, 0)
    return torch.LongTensor(tokens).to(device).unsqueeze(0)


def _preprocess_wave(wave: Any) -> Any:
    torch = importlib.import_module("torch")
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = _mel_spectrogram()(wave_tensor)
    return (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - MEL_MEAN) / MEL_STD


def _mel_spectrogram() -> Any:
    global MEL
    if MEL is None:
        torchaudio = importlib.import_module("torchaudio")
        MEL = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    return MEL
