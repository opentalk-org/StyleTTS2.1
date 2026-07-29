import argparse
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


BASE_URL = "https://data.mendeley.com"
HEADERS = {
    "Accept": "application/vnd.mendeley-public-dataset.1+json",
    "User-Agent": "Mozilla/5.0 stage1-dataset-importer",
}


def read_json(url: str) -> list[dict]:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
        return json.load(response)


def download_file(item: dict, folder_name: str, destination: Path) -> dict:
    output = destination / folder_name / item["filename"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or output.stat().st_size != item["size"]:
        temporary = output.with_suffix(output.suffix + ".part")
        request = urllib.request.Request(item["content_details"]["download_url"], headers=HEADERS)
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)
        assert temporary.stat().st_size == item["size"], f"size mismatch: {item['filename']}"
        temporary.replace(output)
    return {
        "folder": folder_name,
        "filename": item["filename"],
        "size": item["size"],
        "sha256": item["content_details"]["sha256_hash"],
        "file_id": item["id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected public Mendeley folders")
    parser.add_argument("dataset_id")
    parser.add_argument("version", type=int)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--parent", action="append", default=[])
    parser.add_argument("--folder", action="append", default=[])
    parser.add_argument("--include-folder-files", action="append", default=[])
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    folders = read_json(
        f"{BASE_URL}/public-api/datasets/{arguments.dataset_id}/folders/{arguments.version}"
    )
    by_id = {folder["id"]: folder for folder in folders}
    selected = []
    for folder in folders:
        parent_name = by_id[folder["parent_id"]]["name"] if "parent_id" in folder else None
        parent_matches = not arguments.parent or parent_name in arguments.parent
        folder_matches = not arguments.folder or folder["name"] in arguments.folder
        if parent_matches and folder_matches:
            selected.append((folder, parent_name))
    locations = [("root", "root")] if not folders else [
        (folder["id"], f"{parent}/{folder['name']}" if parent else folder["name"])
        for folder, parent in selected
    ]
    locations.extend(
        (folder["id"], folder["name"])
        for folder in folders
        if folder["name"] in arguments.include_folder_files
    )
    tasks = []
    for folder_id, folder_name in locations:
        url = (
            f"{BASE_URL}/public-api/datasets/{arguments.dataset_id}/files"
            f"?folder_id={folder_id}&version={arguments.version}"
        )
        tasks.extend((item, folder_name) for item in read_json(url))
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        records = list(executor.map(lambda task: download_file(*task, arguments.destination), tasks))
    (arguments.destination / "inventory.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"downloaded={len(records)}")


if __name__ == "__main__":
    main()
