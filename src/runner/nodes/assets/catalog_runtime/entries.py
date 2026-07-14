from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class CatalogKey(StrEnum):
    STYLETTS2_UTILS = "styletts2_utils"
    OFFICIAL_CHECKPOINTS = "official_checkpoints"
    PAPERCUP_MULTILINGUAL_PL_BERT = "papercup_multilingual_pl_bert"
    VOKAN_CHECKPOINT = "vokan_checkpoint"
    ASR_MODELS = "asr_models"
    MOS_MODELS = "mos_models"
    TTS_MODELS = "tts_models"
    TURN_MODELS = "turn_models"


class CatalogEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    file: str
    group: str
    catalog_key: CatalogKey
    item: str


CATALOG_ENTRIES: tuple[CatalogEntry, ...] = (
    CatalogEntry(name="Kokoro · 82M (8 langs, 54 voices)", file="hexgrad/Kokoro-82M", group="TTS", catalog_key=CatalogKey.TTS_MODELS, item="kokoro"),
    CatalogEntry(name="Chatterbox · multilingual (~23 langs, clone)", file="ResembleAI/chatterbox", group="TTS", catalog_key=CatalogKey.TTS_MODELS, item="chatterbox"),
    CatalogEntry(name="F5-TTS · v1 base (EN/ZH, clone)", file="SWivid/F5-TTS", group="TTS", catalog_key=CatalogKey.TTS_MODELS, item="f5_tts"),
    CatalogEntry(name="Orpheus · 3B (EN, 8 voices)", file="unsloth/orpheus-3b-0.1-ft", group="TTS", catalog_key=CatalogKey.TTS_MODELS, item="orpheus"),
    CatalogEntry(name="Dia · 1.6B (EN dialogue, clone)", file="nari-labs/Dia-1.6B-0626", group="TTS", catalog_key=CatalogKey.TTS_MODELS, item="dia"),
    CatalogEntry(name="Fish S2-Pro · dual-AR (80+ langs, clone)", file="fishaudio/s2-pro", group="TTS", catalog_key=CatalogKey.TTS_MODELS, item="fish_speech"),
    CatalogEntry(name="StyleTTS2 · LibriTTS", file="epochs_2nd_00020.pth", group="StyleTTS2", catalog_key=CatalogKey.OFFICIAL_CHECKPOINTS, item="official_styletts2_libritts"),
    CatalogEntry(name="StyleTTS2 · LJSpeech", file="epoch_2nd_00100.pth", group="StyleTTS2", catalog_key=CatalogKey.OFFICIAL_CHECKPOINTS, item="official_styletts2_ljspeech"),
    CatalogEntry(name="StyleTTS2 · Vokan", file="epoch_2nd_00012.pth", group="StyleTTS2", catalog_key=CatalogKey.VOKAN_CHECKPOINT, item="vokan_styletts2"),
    CatalogEntry(name="PL-BERT · multilingual", file="step_1100000.t7", group="Training assets", catalog_key=CatalogKey.PAPERCUP_MULTILINGUAL_PL_BERT, item="papercup_multilingual_pl_bert"),
    CatalogEntry(name="ASR · base aligner", file="epoch_00080.pth", group="Training assets", catalog_key=CatalogKey.STYLETTS2_UTILS, item="styletts2_utils_asr"),
    CatalogEntry(name="F0 · JDC", file="bst.t7", group="Training assets", catalog_key=CatalogKey.STYLETTS2_UTILS, item="styletts2_utils_f0"),
    CatalogEntry(name="PL-BERT · StyleTTS2 utils", file="step_1000000.t7", group="Training assets", catalog_key=CatalogKey.STYLETTS2_UTILS, item="styletts2_utils_plbert"),
    CatalogEntry(name="Wav2Vec2 XLS-R 300M · MOS base", file="facebook/wav2vec2-xls-r-300m", group="Training assets", catalog_key=CatalogKey.MOS_MODELS, item="facebook/wav2vec2-xls-r-300m"),
    CatalogEntry(name="Whisper · tiny", file="tiny.pt", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="whisper:tiny"),
    CatalogEntry(name="Whisper · tiny.en", file="tiny.en.pt", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="whisper:tiny.en"),
    CatalogEntry(name="Whisper · base", file="base.pt", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="whisper:base"),
    CatalogEntry(name="Whisper · base.en", file="base.en.pt", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="whisper:base.en"),
    CatalogEntry(name="Whisper · small", file="small.pt", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="whisper:small"),
    CatalogEntry(name="Whisper · small.en", file="small.en.pt", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="whisper:small.en"),
    CatalogEntry(name="Whisper · medium", file="medium.pt", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="whisper:medium"),
    CatalogEntry(name="Whisper · medium.en", file="medium.en.pt", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="whisper:medium.en"),
    CatalogEntry(name="Whisper · large", file="large.pt", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="whisper:large"),
    CatalogEntry(name="Whisper · large-v1", file="large-v1.pt", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="whisper:large-v1"),
    CatalogEntry(name="Whisper · large-v2", file="large-v2.pt", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="whisper:large-v2"),
    CatalogEntry(name="Whisper · large-v3", file="large-v3.pt", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="whisper:large-v3"),
    CatalogEntry(name="Whisper · turbo", file="large-v3-turbo.pt", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="whisper:turbo"),
    CatalogEntry(name="Parakeet · TDT 0.6B v2", file="parakeet-tdt-0.6b-v2.nemo", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="parakeet:nvidia/parakeet-tdt-0.6b-v2"),
    CatalogEntry(name="Parakeet · TDT 0.6B v3", file="parakeet-tdt-0.6b-v3.nemo", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="parakeet:nvidia/parakeet-tdt-0.6b-v3"),
    CatalogEntry(name="Canary · 1B v2", file="canary-1b-v2.nemo", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="canary:nvidia/canary-1b-v2"),
    CatalogEntry(name="Canary · 1B", file="canary-1b.nemo", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="canary:nvidia/canary-1b"),
    CatalogEntry(name="Canary · 1B Flash", file="canary-1b-flash.nemo", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="canary:nvidia/canary-1b-flash"),
    CatalogEntry(name="Canary · 180M Flash", file="canary-180m-flash.nemo", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="canary:nvidia/canary-180m-flash"),
    CatalogEntry(name="Sortformer · 4spk v1", file="diar_sortformer_4spk-v1.nemo", group="Diarization", catalog_key=CatalogKey.ASR_MODELS, item="sortformer:nvidia/diar_sortformer_4spk-v1"),
    CatalogEntry(name="WhisperX align · English", file="wav2vec2-base-960h", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="whisperx:facebook/wav2vec2-base-960h"),
    CatalogEntry(name="WhisperX align · Polish", file="wav2vec2-large-xlsr-53-polish", group="Transcription", catalog_key=CatalogKey.ASR_MODELS, item="whisperx:jonatasgrosman/wav2vec2-large-xlsr-53-polish"),
    CatalogEntry(name="Raon OpenTTS · 1B", file="KRAFTON/Raon-OpenTTS-1B", group="TTS", catalog_key=CatalogKey.TTS_MODELS, item="raon_opentts"),
    CatalogEntry(name="Smart Turn v3.2 · CPU ONNX", file="smart-turn-v3.2-cpu.onnx", group="Turn detection", catalog_key=CatalogKey.TURN_MODELS, item="pipecat-ai/smart-turn-v3"),
)

assert len({(entry.catalog_key, entry.item) for entry in CATALOG_ENTRIES}) == len(CATALOG_ENTRIES), "catalog key/item pairs must be unique"


def catalog_entries_schema() -> list[dict[str, str]]:
    return [entry.model_dump(mode="json") for entry in CATALOG_ENTRIES]
