from __future__ import annotations

import io
import re
import shutil
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import ResourcePolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio, stable_id
from shared.db import database_session
from shared.db.voices.models import Voice
from shared.audio_annotations import AudioAnnotations

_INFO_FIELDS = (
    "id",
    "title",
    "fulltitle",
    "description",
    "uploader",
    "uploader_id",
    "uploader_url",
    "channel",
    "channel_id",
    "channel_url",
    "duration",
    "upload_date",
    "timestamp",
    "view_count",
    "like_count",
    "comment_count",
    "categories",
    "tags",
    "language",
    "ext",
    "acodec",
    "abr",
    "asr",
    "webpage_url",
    "original_url",
    "extractor",
    "extractor_key",
    "availability",
    "age_limit",
    "live_status",
    "thumbnail",
)


class YouTubeAudioSourceSettings(StrictSettings):
    urls: list[str] = Field(default_factory=list, title="YouTube URLs")
    proxies: list[str] = Field(
        default_factory=list,
        title="Proxies",
        description="Optional proxy URLs (e.g. http://host:port). Rotated round-robin across downloads.",
    )
    name_prefix: str = Field(default="youtube", title="Audio name prefix")
    max_parallel: int = Field(default=4, ge=1, le=32, title="Max parallel downloads")
    stagger_seconds: float = Field(default=0.5, ge=0.0, le=10.0, title="Stagger between starts (s)")
    download_retries: int = Field(default=3, ge=1, le=10, title="Download retries")
    create_voices: bool = Field(default=True, title="Create voices from uploader")


class YouTubeAudioSourceNode(Node):
    NODE_TYPE = "YouTubeAudioSource"
    DESCRIPTION = "Download the audio track of one or more YouTube URLs as WAV and stream them into a workflow. Emits one audio clip per URL with rich metadata (title, uploader, channel, duration, language, view/like counts, tags), and can create voices from the uploader names. Downloads run in parallel with staggered starts and optional round-robin proxies to avoid hammering the source. Use it as a starting point to bring YouTube audio into a pipeline."
    CATEGORY = "Inputs"
    SETTINGS = YouTubeAudioSourceSettings
    IS_INPUT = True
    INPUTS = {}
    OUTPUTS = {"audio": AudioPort(mode=PortMode.STREAM)}
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._items: list[Audio] | None = None
        self._cursor = 0

    def remaining_items(self, context: Any) -> int:
        if self._items is None:
            return len(_clean_list(self.settings.urls))
        return len(self._items) - self._cursor

    async def execute(self, batch: list[dict[str, Any]], context: Any) -> list[dict[str, Audio]]:
        if self._items is None:
            self._items = _load_audio_items(self.settings, self.logger)
        end = self._cursor + self.runtime.queue_max_size
        items = self._items[self._cursor:end]
        self._cursor += len(items)
        return [{"audio": item} for item in items]


def _load_audio_items(settings: YouTubeAudioSourceSettings, logger: Any) -> list[Audio]:
    urls = _clean_list(settings.urls)
    if not urls:
        raise ValueError("YouTubeAudioSource requires at least one URL")
    proxies = _clean_list(settings.proxies)
    workdir = Path(tempfile.mkdtemp(prefix="youtube_source_"))
    try:
        max_workers = min(settings.max_parallel, len(urls))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    _staggered_download,
                    index,
                    url,
                    proxies[index % len(proxies)] if proxies else None,
                    workdir,
                    settings,
                    logger,
                )
                for index, url in enumerate(urls)
            ]
            downloads = [future.result() for future in futures]
        voice_ids = _voice_ids_for_uploaders(settings, [info.get("uploader") for _url, _proxy, info, _wav in downloads])
        return [
            _audio_from_download(url, proxy, info, wav_bytes, settings, voice_ids)
            for url, proxy, info, wav_bytes in downloads
        ]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _staggered_download(
    index: int,
    url: str,
    proxy: str | None,
    workdir: Path,
    settings: YouTubeAudioSourceSettings,
    logger: Any,
) -> tuple[str, str | None, dict[str, Any], bytes]:
    # Stagger each download's start so we don't fire every request at once.
    if settings.stagger_seconds > 0 and index > 0:
        time.sleep(settings.stagger_seconds * index)
    logger.info("youtube download start url=%s proxy=%s", url, proxy or "-")
    info, wav_bytes = _download_audio(url, proxy, workdir, settings.download_retries)
    logger.info("youtube download done url=%s bytes=%d", url, len(wav_bytes))
    return url, proxy, info, wav_bytes


def _download_audio(url: str, proxy: str | None, workdir: Path, retries: int) -> tuple[dict[str, Any], bytes]:
    try:
        import yt_dlp
    except ImportError as error:
        raise RuntimeError("yt-dlp is not installed") from error

    options: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": retries,
        "restrictfilenames": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }
    if proxy:
        options["proxy"] = proxy
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.sanitize_info(ydl.extract_info(url, download=True))
    except Exception as error:  # noqa: BLE001 - surface the URL with the yt-dlp failure
        raise RuntimeError(f"yt-dlp failed for {url}: {error}") from error

    video_id = str(info.get("id") or "")
    wav_path = workdir / f"{video_id}.wav"
    if not wav_path.exists():
        candidates = sorted(workdir.glob(f"{video_id}*.wav")) or sorted(workdir.glob("*.wav"))
        if not candidates:
            raise RuntimeError(f"yt-dlp produced no WAV for {url}")
        wav_path = candidates[0]
    return info, wav_path.read_bytes()


def _audio_from_download(
    url: str,
    proxy: str | None,
    info: dict[str, Any],
    wav_bytes: bytes,
    settings: YouTubeAudioSourceSettings,
    voice_ids: dict[str, UUID],
) -> Audio:
    wav = _wav_info(wav_bytes)
    duration = _float_or_none(info.get("duration")) or wav["duration"]
    sample_rate = int(wav["sample_rate"])
    channels = int(wav["channels"])
    video_id = str(info.get("id") or stable_id("youtube", url))
    webpage_url = _string_or_none(info.get("webpage_url")) or url
    audio_file_id = uuid5(NAMESPACE_URL, webpage_url)
    uploader = _string_or_none(info.get("uploader"))
    voice_id = voice_ids.get(uploader) if uploader else None
    name = f"{_safe_prefix(settings.name_prefix)}_{_safe_stem(video_id)}.wav"
    return Audio(
        audio_file_id=audio_file_id,
        name=name,
        data=wav_bytes,
        sample_rate=sample_rate,
        channels=channels,
        start=0.0,
        end=float(duration),
        annotations=AudioAnnotations(
            speaker_id=uploader,
            voice_id=voice_id,
            metadata=_audio_metadata(url, proxy, info, sample_rate, channels, float(duration)),
        ),
        id=stable_id("youtube_audio", webpage_url),
        lineage_id=stable_id("youtube_audio_lineage", webpage_url),
        byte_length=len(wav_bytes),
        virtual=False,
        segments=[],
    )


def _audio_metadata(
    url: str,
    proxy: str | None,
    info: dict[str, Any],
    sample_rate: int,
    channels: int,
    duration: float,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source": "youtube",
        "source_url": url,
        "proxy": proxy,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration": duration,
        "language": _string_or_none(info.get("language")),
    }
    for field in _INFO_FIELDS:
        metadata[field] = _scalar(info.get(field))
    return metadata


def _voice_ids_for_uploaders(settings: YouTubeAudioSourceSettings, uploaders: list[str | None]) -> dict[str, UUID]:
    if not settings.create_voices:
        return {}
    names = sorted({name for raw in uploaders if (name := _string_or_none(raw))})
    if not names:
        return {}
    with database_session() as session:
        session.execute(
            insert(Voice)
            .values([{"name": name} for name in names])
            .on_conflict_do_nothing(index_elements=["name"])
        )
        session.commit()
        voices = session.execute(select(Voice).where(Voice.name.in_(names))).scalars().all()
        return {voice.name: voice.id for voice in voices}


def _wav_info(wav_bytes: bytes) -> dict[str, float | int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        return {
            "sample_rate": sample_rate,
            "channels": wav_file.getnchannels(),
            "duration": frames / float(sample_rate) if sample_rate > 0 else 0.0,
        }


def _clean_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    return [item.strip() for item in values if item and item.strip()]


def _safe_prefix(prefix: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("._") or "youtube"


def _safe_stem(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "video"


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, (str, int, float, bool))]
    return str(value)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
