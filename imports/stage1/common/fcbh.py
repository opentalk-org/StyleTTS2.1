import base64
import gzip
import hashlib
import json
import re
import shutil
import struct
import subprocess
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from xml.etree import ElementTree

from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
from cryptography.hazmat.primitives.ciphers import Cipher, modes
from cryptography.hazmat.primitives.padding import PKCS7
from tqdm import tqdm

from imports.stage1.common.audio import normalize_audio
from imports.stage1.common.schema import AudioRecord, DatasetManifest, DatasetRecord, SegmentRecord


TEA_KEY = struct.unpack("<4I", bytes.fromhex("f9286be50f2869aea6286f9edbf87a2f"))
DES_KEY = b"MJmsLtin"


@dataclass(frozen=True)
class Chapter:
    book: str
    book_name: str
    chapter: int
    audio_asset: str
    published_audio_name: str
    text: str
    text_assets: tuple[str, ...]


def tea_decrypt_chunks(source: bytes) -> bytes:
    output = bytearray(source)
    for chunk in range(0, len(output), 1024):
        chunk_length = min(1024, len(output) - chunk)
        for offset in range(chunk, chunk + chunk_length - 7, 8):
            v0, v1 = struct.unpack_from("<2I", output, offset)
            total = 0xC6EF3720
            for _ in range(32):
                v1 = (v1 - ((((v0 << 4) & 0xFFFFFFFF) + TEA_KEY[2]) ^ ((v0 + total) & 0xFFFFFFFF) ^ ((v0 >> 5) + TEA_KEY[3]))) & 0xFFFFFFFF
                v0 = (v0 - ((((v1 << 4) & 0xFFFFFFFF) + TEA_KEY[0]) ^ ((v1 + total) & 0xFFFFFFFF) ^ ((v1 >> 5) + TEA_KEY[1]))) & 0xFFFFFFFF
                total = (total + 0x61C88647) & 0xFFFFFFFF
            struct.pack_into("<2I", output, offset, v0, v1)
    return bytes(output)


def decrypt_config(source: bytes) -> str:
    try:
        encoded = source.decode("ascii")
    except UnicodeDecodeError:
        return gzip.decompress(tea_decrypt_chunks(source)).decode("utf-8")
    decryptor = Cipher(TripleDES(DES_KEY * 3), modes.ECB()).decryptor()
    padded = decryptor.update(base64.b64decode(encoded)) + decryptor.finalize()
    unpadder = PKCS7(64).unpadder()
    return (unpadder.update(padded) + unpadder.finalize()).decode("utf-8")


def clean_usfm(source: str) -> str:
    text = re.sub(r"\\(?:f|x)\s.*?\\(?:f|x)\*", " ", source, flags=re.DOTALL)
    text = re.sub(r"\\(?:c|v)\s+\d+\s*", " ", text)
    text = re.sub(r"\\[A-Za-z0-9-]+\*?\s*", " ", text)
    return " ".join(text.split())


def config_asset(names: list[str]) -> str:
    matches = [name for name in names if len(name) in (10, 20) and "c" in name and "." not in name]
    if len(matches) != 1:
        raise ValueError(f"expected one FCBH config asset, found {matches}")
    return matches[0]


def chapter_texts(archive: zipfile.ZipFile, names: list[str], filename: str) -> tuple[dict[int, str], dict[int, tuple[str, ...]]]:
    texts: dict[int, str] = {}
    assets: dict[int, tuple[str, ...]] = {}
    for name in names:
        if not name.endswith(filename) or name == filename:
            continue
        try:
            decoded = gzip.decompress(tea_decrypt_chunks(archive.read(f"assets/{name}"))).decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        sections = re.split(r"(?=\\c\s+\d+)", decoded)
        for section in sections:
            match = re.match(r"\\c\s+(\d+)", section)
            if match is None:
                continue
            chapter = int(match.group(1))
            cleaned = clean_usfm(section)
            if cleaned:
                texts[chapter] = cleaned
                assets[chapter] = assets.get(chapter, ()) + (name,)
    return texts, assets


def load_chapters(apk: Path) -> tuple[list[Chapter], str]:
    with zipfile.ZipFile(apk) as archive:
        names = [name.removeprefix("assets/") for name in archive.namelist() if name.startswith("assets/")]
        config = decrypt_config(archive.read(f"assets/{config_asset(names)}"))
        root = ElementTree.fromstring(config)
        chapters: list[Chapter] = []
        for book in root.findall(".//book"):
            book_id = book.attrib["id"]
            book_name = book.findtext("name") or book.findtext("n")
            filename = book.findtext("filename")
            pages = book.findall("page")
            if not filename:
                details_asset = book.findtext("bd")
                details_xml = gzip.decompress(tea_decrypt_chunks(archive.read(f"assets/{details_asset}")))
                details = ElementTree.fromstring(details_xml)
                filename = details.findtext("f")
                pages = details.findall("page")
            texts, text_assets = chapter_texts(archive, names, filename)
            for page in pages:
                audio = page.find("./audio/f")
                if audio is None:
                    continue
                chapter = int(page.attrib["num"])
                if chapter not in texts:
                    continue
                audio_asset = audio.attrib["obf"] if "obf" in audio.attrib else audio.text
                if audio_asset is None:
                    raise ValueError(f"{book_id} chapter {chapter}: audio has no asset reference")
                chapters.append(Chapter(
                    book=book_id,
                    book_name=book_name,
                    chapter=chapter,
                    audio_asset=audio_asset,
                    published_audio_name=audio.text,
                    text=texts[chapter],
                    text_assets=text_assets[chapter],
                ))
    return chapters, config


def select_chapters(chapters: list[Chapter], durations: dict[str, float], target_seconds: float) -> list[Chapter]:
    by_book: dict[str, list[Chapter]] = {}
    for chapter in chapters:
        by_book.setdefault(chapter.book, []).append(chapter)
    ordered: list[Chapter] = []
    while any(by_book.values()):
        for book_chapters in by_book.values():
            if book_chapters:
                ordered.append(book_chapters.pop(0))
    selected: list[Chapter] = []
    total = 0.0
    for chapter in ordered:
        duration = durations[chapter.audio_asset]
        if total + duration <= target_seconds:
            selected.append(chapter)
            total += duration
    if not selected:
        selected.append(min(ordered, key=lambda chapter: durations[chapter.audio_asset]))
    return selected


def source_duration(source: Path) -> float:
    with source.open("rb") as handle:
        input_format = ["-f", "mp3"] if handle.read(3) == b"ID3" else []
    result = subprocess.run(
        ["ffprobe", "-v", "error", *input_format, "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(source)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def chapter_duration(chapter: Chapter, extraction: Path) -> tuple[str, float]:
    return chapter.audio_asset, source_duration(extraction / chapter.audio_asset)


def remote_audio_inventory(config: str) -> dict[str, tuple[str, float]]:
    root = ElementTree.fromstring(config)
    inventory: dict[str, tuple[str, float]] = {}
    for source in root.findall(".//audio-source[@type='fcbh']"):
        key = source.findtext("key")
        fileset = source.findtext("dam-id")
        if key is None or fileset is None:
            raise ValueError("FCBH remote audio source lacks key or fileset ID")
        query = urllib.parse.urlencode({"key": key, "v": 4})
        endpoint = f"https://4.dbt.io/api/bibles/filesets/{fileset}?{query}"
        with urllib.request.urlopen(endpoint, timeout=30) as response:
            document = json.load(response)
        for row in document["data"]:
            path = str(row["path"])
            inventory[Path(urllib.parse.urlsplit(path).path).name] = (path, float(row["duration"]))
    return inventory


def make_record(
    chapter: Chapter,
    extraction: Path,
    stage_root: Path,
    dataset_name: str,
    language: str,
    source_url: str,
    source_bytes: dict[str, bytes],
) -> AudioRecord:
    source = extraction / chapter.audio_asset
    identity = f"{chapter.book}.{chapter.chapter}:{chapter.audio_asset}"
    suffix = hashlib.sha1(identity.encode()).hexdigest()[:8]
    destination = stage_root / "wavs" / f"{chapter.book}_{chapter.chapter:03d}_{suffix}.wav"
    duration = normalize_audio(source, destination)
    return AudioRecord(
        path=destination.relative_to(stage_root).as_posix(), source_id=identity,
        duration=duration, language=language, speaker_id=f"{stage_root.name}_narrator",
        style_prompt="scripture narration", voice_prompt="publisher narrator", score=None, accuracy=None,
        segments=[SegmentRecord(start=0.0, end=duration, text=chapter.text, source="dataset", score=None, accuracy=None, alignment=[])],
        metadata={"source_dataset": dataset_name, "source_url": source_url, "publisher_row": {
            "book": chapter.book, "book_name": chapter.book_name, "chapter": chapter.chapter,
            "published_audio_name": chapter.published_audio_name, "obfuscated_audio_asset": chapter.audio_asset,
            "text_assets": list(chapter.text_assets), "audio": {"path": chapter.audio_asset,
            "byte_length": len(source_bytes[chapter.audio_asset]),
            "sha256": hashlib.sha256(source_bytes[chapter.audio_asset]).hexdigest()}}},
    )


def prepare_fcbh(
    apk: Path,
    stage_root: Path,
    dataset_name: str,
    language: str,
    source_url: str,
    workers: int,
    target_hours: float | None = None,
) -> None:
    chapters, config = load_chapters(apk)
    extraction = stage_root / "tmp" / "audio"
    shutil.rmtree(stage_root / "wavs", ignore_errors=True)
    extraction.mkdir(parents=True, exist_ok=True)
    (stage_root / "wavs").mkdir()
    with zipfile.ZipFile(apk) as archive:
        archive_names = set(archive.namelist())
        missing = [chapter.audio_asset for chapter in chapters if f"assets/{chapter.audio_asset}" not in archive_names]
        remote_inventory = remote_audio_inventory(config) if missing else {}
        if missing:
            durations = {
                chapter.audio_asset: remote_inventory[Path(chapter.published_audio_name).name][1]
                for chapter in chapters
            }
            if target_hours is not None:
                chapters = select_chapters(chapters, durations, target_hours * 3600.0)
        for chapter in chapters:
            destination = extraction / chapter.audio_asset
            destination.parent.mkdir(parents=True, exist_ok=True)
            archive_name = f"assets/{chapter.audio_asset}"
            if archive_name in archive_names:
                payload = archive.read(archive_name)
            else:
                remote_url = remote_inventory[Path(chapter.published_audio_name).name][0]
                with urllib.request.urlopen(remote_url, timeout=60) as response:
                    payload = response.read()
            destination.write_bytes(payload)
        if not missing:
            duration_worker = partial(chapter_duration, extraction=extraction)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                durations = dict(executor.map(duration_worker, chapters))
            if target_hours is not None:
                chapters = select_chapters(chapters, durations, target_hours * 3600.0)
        source_bytes: dict[str, bytes] = {}
        for chapter in chapters:
            payload = (extraction / chapter.audio_asset).read_bytes()
            source_bytes[chapter.audio_asset] = payload

    worker = partial(make_record, extraction=extraction, stage_root=stage_root, dataset_name=dataset_name,
                     language=language, source_url=source_url, source_bytes=source_bytes)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(tqdm(executor.map(worker, chapters), total=len(chapters), desc=dataset_name, unit="chapter"))
    hours = sum(record.duration for record in records) / 3600.0
    manifest = DatasetManifest(dataset=DatasetRecord(name=dataset_name, language_limits_hours={language: hours}, source_url=source_url), audio_files=records)
    temporary = stage_root / "data.json.tmp"
    temporary.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8")
    temporary.replace(stage_root / "data.json")
    shutil.rmtree(stage_root / "tmp")
    (stage_root / "tmp").mkdir()
