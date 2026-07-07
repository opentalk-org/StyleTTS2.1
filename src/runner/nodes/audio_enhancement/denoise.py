from __future__ import annotations

import importlib
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import AUDIO
from runner.nodes.models import Audio, stable_id

_PATCHED_TORCH: Any | None = None
_PATCHED_SF: Any | None = None
_PATCHED_NP: Any | None = None


class DeepFilterNetSettings(StrictSettings):
    model: str = "deepfilternet3"
    strength: float = Field(default=0.8, ge=0.0, le=1.0)


@dataclass(frozen=True)
class DeepFilterNetStack:
    model: Any
    df_state: Any
    sample_rate: int


class DeepFilterNetDenoiseNode(Node):
    NODE_TYPE = "DeepFilterNetDenoise"
    CATEGORY = "Audio / Enhancement"
    SETTINGS = DeepFilterNetSettings
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"audio": Port("audio", AUDIO)}
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 4})

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._stack: DeepFilterNetStack | None = None

    async def setup(self, context: Any) -> None:
        self._stack = load_denoise_stack(self.settings.model)

    async def teardown(self, context: Any) -> None:
        self._stack = None

    async def execute(self, batch, context):
        if self._stack is None:
            raise RuntimeError("DeepFilterNetDenoiseNode.setup() must load the model before execute().")
        outputs = []
        for inputs in batch:
            audio = inputs["audio"]
            assert isinstance(audio, Audio), f"unsupported audio input: {type(audio).__name__}"
            denoised = denoise_wav_bytes(audio.data, self._stack)
            denoised_id = stable_id("audio", audio.id, "denoise", self.settings.model, self.settings.strength)
            outputs.append({
                "audio": replace(
                    audio,
                    data=denoised,
                    id=denoised_id,
                    metadata={
                        **audio.metadata,
                        "denoise": {
                            "model": self.settings.model,
                            "strength": self.settings.strength,
                        },
                    },
                ),
            })
        return outputs


def load_denoise_stack(model_name: str) -> DeepFilterNetStack:
    deps = _load_deepfilternet_dependencies()
    torch = deps["torch"]
    sf = deps["soundfile"]
    np = deps["numpy"]
    _patch_df_io(torch, sf, np)
    enhance = importlib.import_module("df.enhance")
    model_module = importlib.import_module("df.model")
    model, df_state, _suffix = enhance.init_df(
        _deepfilter_model_name(model_name),
        post_filter=False,
        log_level="error",
        log_file=None,
        default_model=_deepfilter_model_name(model_name),
    )
    df_sr = model_module.ModelParams().sr
    return DeepFilterNetStack(model, df_state, int(df_sr))


def denoise_wav_bytes(audio_bytes: bytes, stack: DeepFilterNetStack) -> bytes:
    with tempfile.TemporaryDirectory(prefix="runflow-denoise-") as tmp_dir:
        audio_path = Path(tmp_dir) / "audio.wav"
        audio_path.write_bytes(audio_bytes)
        denoise_wav_with_model(audio_path, stack)
        return audio_path.read_bytes()


def denoise_wav_with_model(audio_path: Path, stack: DeepFilterNetStack) -> None:
    enhance_module = importlib.import_module("df.enhance")
    io_module = importlib.import_module("df.io")
    path_str = str(audio_path.resolve())
    audio, meta = io_module.load_audio(path_str, stack.sample_rate)
    enhanced = enhance_module.enhance(stack.model, stack.df_state, audio, pad=True, atten_lim_db=None)
    enhanced = io_module.resample(enhanced.to("cpu"), stack.sample_rate, meta.sample_rate)
    io_module.save_audio(path_str, enhanced, sr=meta.sample_rate, log=False)


def _load_deepfilternet_dependencies() -> dict[str, Any]:
    _ensure_torchaudio_backend_shim()
    modules = {}
    missing = []
    for name in ["numpy", "soundfile", "torch", "libdf", "df"]:
        try:
            modules[name] = importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        names = ", ".join(missing)
        raise ImportError(f"DeepFilterNetDenoise requires optional audio dependencies: {names}") from None
    return modules


def _ensure_torchaudio_backend_shim() -> None:
    """DeepFilterNet's ``df.io`` does ``from torchaudio.backend.common import
    AudioMetaData`` at import time, but ``torchaudio.backend`` was removed in
    torchaudio>=2.1. We never call the torchaudio-backed IO (it is patched out by
    ``_patch_df_io``), so a lightweight shim is enough to keep the import alive."""
    import sys
    import types

    try:
        __import__("torchaudio.backend.common")
        if hasattr(sys.modules.get("torchaudio.backend.common"), "AudioMetaData"):
            return
    except Exception:
        pass
    backend = sys.modules.get("torchaudio.backend") or types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = _AudioMetaData
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common
    import torchaudio
    torchaudio.backend = backend


def _patch_df_io(torch: Any, sf: Any, np: Any) -> None:
    global _PATCHED_TORCH, _PATCHED_SF, _PATCHED_NP
    _PATCHED_TORCH = torch
    _PATCHED_SF = sf
    _PATCHED_NP = np
    _ensure_torchaudio_backend_shim()
    df_io = importlib.import_module("df.io")
    df_io.load_audio = _load_audio_sf
    df_io.save_audio = _save_audio_soundfile_wav


class _AudioMetaData:
    __slots__ = ("sample_rate",)

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate


def _load_audio_sf(
    file: str,
    sr: int | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> tuple[Any, object]:
    assert _PATCHED_TORCH is not None, "DeepFilterNet audio IO patch was not initialized."
    assert _PATCHED_SF is not None, "DeepFilterNet audio IO patch was not initialized."
    assert _PATCHED_NP is not None, "DeepFilterNet audio IO patch was not initialized."
    df_io = importlib.import_module("df.io")
    df_logger = importlib.import_module("df.logger")
    rkwargs = {}
    if "method" in kwargs:
        rkwargs["method"] = kwargs.pop("method")
    data, orig_sr = _PATCHED_SF.read(file, always_2d=True, dtype="float32")
    audio = _PATCHED_TORCH.from_numpy(_PATCHED_NP.ascontiguousarray(data.T))
    meta = _AudioMetaData(int(orig_sr))
    if sr is not None and int(orig_sr) != sr:
        if verbose:
            df_logger.warn_once(
                f"Audio sampling rate does not match model sampling rate ({orig_sr}, {sr}). Resampling..."
            )
        audio = df_io.resample(audio, orig_sr, sr, **rkwargs)
    return audio.contiguous(), meta


def _save_audio_soundfile_wav(
    file: str,
    audio: Any,
    sr: int,
    output_dir: str | None = None,
    suffix: str | None = None,
    log: bool = False,
    dtype: Any = None,
) -> None:
    del log, dtype, suffix, output_dir
    assert _PATCHED_TORCH is not None, "DeepFilterNet audio IO patch was not initialized."
    assert _PATCHED_SF is not None, "DeepFilterNet audio IO patch was not initialized."
    audio_t = _PATCHED_TORCH.as_tensor(audio)
    if audio_t.ndim == 1:
        audio_t = audio_t.unsqueeze(0)
    wav = audio_t.cpu().float().numpy()
    if wav.shape[0] > 1:
        wav = wav.mean(axis=0)
    else:
        wav = wav[0]
    _PATCHED_SF.write(str(Path(file)), wav, int(sr), subtype="PCM_16")


def _deepfilter_model_name(model_name: str) -> str:
    aliases = {
        "deepfilternet2": "DeepFilterNet2",
        "deepfilternet3": "DeepFilterNet3",
    }
    normalized = model_name.strip().lower()
    if normalized in aliases:
        return aliases[normalized]
    return model_name
