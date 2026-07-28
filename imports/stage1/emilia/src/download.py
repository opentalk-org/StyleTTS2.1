from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


DATASET = "amphion/Emilia-Dataset"
LANGUAGES = ("DE", "EN", "FR", "JA", "KO", "ZH")
WORKERS = 6


@dataclass(frozen=True)
class RemoteTar:
    path: str
    size: int


def first_language_tars(siblings: list[Any], languages: tuple[str, ...]) -> list[RemoteTar]:
    paths = sorted((
        RemoteTar(item.rfilename, int(item.size))
        for item in siblings
        if item.rfilename.startswith("Emilia/") and item.rfilename.endswith(".tar")
    ), key=lambda item: item.path)
    selected = []
    for language in languages:
        matches = [item for item in paths if item.path.startswith(f"Emilia/{language}/")]
        assert matches, f"Emilia has no tar files for {language}"
        selected.append(matches[0])
    return selected


def configured_token(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    values = [line.split("=", 1)[1].strip().strip('"').strip("'")
              for line in lines if line.startswith("HF_TOKEN=")]
    assert len(values) == 1 and values[0], "HF_TOKEN is required for gated Emilia access"
    return values[0]


def download_tar(source: RemoteTar, root: Path, token: str) -> Path:
    result = Path(hf_hub_download(
        repo_id=DATASET, repo_type="dataset", filename=source.path,
        token=token, local_dir=root,
    ))
    assert result.stat().st_size == source.size, f"{source.path}: downloaded size differs"
    return result


def main() -> None:
    token = configured_token(Path(".env"))
    info = HfApi(token=token).dataset_info(DATASET, files_metadata=True)
    sources = first_language_tars(info.siblings, LANGUAGES)
    root = Path("imports/stage1/emilia/tmp/repository")
    print(f"files={len(sources)} bytes={sum(item.size for item in sources)}", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        downloads = executor.map(lambda source: download_tar(source, root, token), sources)
        for _ in tqdm(downloads, total=len(sources), desc="emilia", unit="tar"):
            pass


if __name__ == "__main__":
    main()
