import re
from pathlib import Path


TRACKER = Path("imports/dataset-download-groups-1000h.md")
STAGE_ROOT = Path("imports/stage1")
COMPLETE_SLUGS = set(
    Path("imports/stage1-complete-slugs.txt").read_text(encoding="utf-8").splitlines()
)
LEGEND = (
    "State: ❌ `FAIL` · ⛔ `ACCESS_DENIED` · ⏳ `ACTIVE` · "
    "📍 `COMPLETE_LOCAL` · ✅ `COMPLETE`"
)
STATE_LABELS = {
    "ACCESS_DENIED": "⛔ ACCESS_DENIED",
    "FAIL": "❌ FAIL",
}
BACKEND_LABELS = {"☁️ VERIFIED", "🔄 VERIFYING", "📍 LOCAL_ONLY", "—"}
FCBH_COMPLETE = set(
    Path("imports/fcbh-complete-apks.txt").read_text(encoding="utf-8").splitlines()
)
FCBH_EXCLUDED = {
    "Hindi_Sab_Ki-1.0.1.apk",
    "Russian_CARS-1.0.2.apk",
    "Tamil_Contemporary_Bible-1.0.apk",
}
SOURCE_FOLDERS = {
    "ASED": "ased",
    "Crimean Tatar TTS": "crimean_tatar_tts",
    "FCBH South Azerbaijani Bible": "fcbh_south_azerbaijani_bible",
    "SyntAct": "syntact",
    "VoxPopuli accented English": "voxpopuli",
    "Kannada Emotional Speech": "kannada_emotional",
    "MrlolDev/voxtral-emotion-speech": "voxtral_emotion_speech",
    "RAVDESS speech": "ravdess_speech_official",
    "yfish/WESR-Bench": "wesr_bench",
    "synthbot/pony-singing": "pony_singing",
    "IAMCB/elise-clone": "elise_clone",
    "projecte-aina/LaFrescat": "lafrescat",
    "laion/synthetic_vocal_burts_dramabox": "synthetic_vocalbursts_dramabox",
    "NAC-v1.0 / UD Naija Spoken Corpus derivative": "nac_v1",
    "MikhailT/hifi-tts clean": "hifi_tts_clean",
    "DragonLine/ksponspeech_04": "ksponspeech_04",
    "Sh1man/elevenlabs": "elevenlabs",
    "laion/more-synthetic-vocalbursts-raw": "more_synthetic_vocalbursts_raw",
    "ALFFA": "alffa_amharic",
    "jp1924/KoreaSpeech": "koreaspeech",
    "joujiboi/japanese-anime-speech-v2": "japanese_anime_speech_v2",
    "ShoukanLabs/AniSpeech": "anispeech",
    "laion/vocal-burst-db": "vocal_burst_db",
    "alexandrainst/ftspeech": "ftspeech",
    "facebook/multilingual_librispeech": "multilingual_librispeech",
    "simon3000/starrail-voice": "starrail_voice",
    "sleeping-ai/11Labs": "sleeping_ai_11labs",
    "TED-LIUM": "tedlium",
    "TurkmenSpeech": "turkmen_speech",
    "OpenSLR 52 Sinhala ASR": "openslr_52_sinhala_asr",
    "LocalDoc ASR": "localdoc_azerbaijani_asr",
    "Althingi parliamentary corpus": "althingi",
    "LOD_Claude synthetic Luxembourgish": "lod_claude",
    "facebook/voxpopuli": "voxpopuli",
    "DMC-ykfx33/nsfw_tts_dataset_30speakers": "dmc_nsfw_30speakers",
    "fixie-ai/soda-audio": "soda_audio",
    "simon3000/genshin-voice": "genshin_voice",
    "amphion/Emilia": "emilia",
    "Meta Omnilingual ASR": "meta_omnilingual_asr",
    "ylacombe/cml-tts": "cml_tts",
    "espnet/mms_ulab_v2": "mms_ulab_v2",
    "OpenBibleTTS": "open_bible_tts",
    "Mozilla Common Voice — part 1/3": "common_voice_part1",
    "Mozilla Common Voice — part 2/3": "common_voice_part2",
    "Mozilla Common Voice — part 3/3": "common_voice_part3",
    "OWSMv4 cleaned YODAS": "yodas_owsmv4",
    "FCBH Hakka Bible New Testament": "fcbh_hakka_bible",
    "FCBH Burmese MSB New Testament": "fcbh_burmese_msb",
    "FCBH Shan TBS New Testament": "fcbh_shan_tbs",
}


def normalized_folder(source: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
    return value


def source_folder(source: str) -> str | None:
    if source.startswith("FCBH/"):
        return "fcbh_group1"
    if source in SOURCE_FOLDERS:
        return SOURCE_FOLDERS[source]
    candidate = normalized_folder(source)
    if candidate in COMPLETE_SLUGS or (STAGE_ROOT / candidate).is_dir():
        return candidate
    raise ValueError(f"dataset tracker source has no stage mapping: {source}")


def source_state(source: str) -> str:
    if source.startswith("FCBH/"):
        apk_name = source.removeprefix("FCBH/")
        if apk_name in FCBH_EXCLUDED:
            return "❌ FAIL"
        if apk_name in FCBH_COMPLETE:
            return "✅ COMPLETE"
        raise ValueError(f"FCBH tracker source has no terminal result: {apk_name}")
    folder = source_folder(source)
    if folder is None:
        return "⏳ ACTIVE"
    if folder in COMPLETE_SLUGS:
        return "✅ COMPLETE"
    status_path = STAGE_ROOT / folder / "STATUS.md"
    if not status_path.exists():
        return "⏳ ACTIVE"
    state = status_path.read_text(encoding="utf-8").splitlines()[0]
    if state == "COMPLETE":
        backend_path = STAGE_ROOT / folder / "BACKEND.md"
        if backend_path.exists() and backend_path.read_text(encoding="utf-8").splitlines()[0] == "COMPLETE":
            return "✅ COMPLETE"
        return "📍 COMPLETE_LOCAL"
    if state not in STATE_LABELS:
        raise ValueError(f"{status_path}: unknown state {state!r}")
    return STATE_LABELS[state]


def update_tracker() -> tuple[int, dict[str, int]]:
    lines = TRACKER.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    in_dataset_table = False
    states: dict[str, int] = {}
    rows = 0
    for line in lines:
        if line.startswith("State:"):
            continue
        if line == "## Group summary":
            output.extend([LEGEND, ""])
        if line.startswith("| Dataset / download source |"):
            output.append(
                "| Dataset / download source | State | Hours to get | Languages | "
                "Table rows | Included labels/configurations |"
            )
            in_dataset_table = True
            continue
        if in_dataset_table and line.startswith("|---"):
            output.append("|---|---|---:|---:|---:|---|")
            continue
        if in_dataset_table and line.startswith("| "):
            fields = [field.strip() for field in line.strip().strip("|").split("|")]
            source = fields[0]
            if len(fields) >= 6 and fields[1] in set(STATE_LABELS.values()) | {
                "🕒 PENDING", "⏳ ACTIVE", "📍 COMPLETE_LOCAL", "✅ COMPLETE",
            }:
                fields.pop(1)
            if len(fields) >= 6 and fields[1] in BACKEND_LABELS:
                fields.pop(1)
            if len(fields) != 5:
                raise ValueError(f"unexpected dataset tracker row: {line}")
            state = source_state(source)
            fields.insert(1, state)
            output.append("| " + " | ".join(fields) + " |")
            states.setdefault(state, 0)
            states[state] += 1
            rows += 1
            continue
        if in_dataset_table and not line.startswith("|"):
            in_dataset_table = False
        output.append(line)
    TRACKER.write_text("\n".join(output) + "\n", encoding="utf-8")
    return rows, states


def main() -> None:
    rows, states = update_tracker()
    print(f"UPDATED rows={rows} states={states}", flush=True)


if __name__ == "__main__":
    main()
