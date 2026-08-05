import json
import shutil
from pathlib import Path


STAGE_ROOT = Path("imports/stage1").resolve()
REGISTRY = Path("imports/stage1-complete-slugs.txt")
FCBH_COMPLETE = Path("imports/fcbh-complete-apks.txt")


def cleanup_fcbh() -> int:
    group = STAGE_ROOT / "fcbh_group1"
    if not group.exists():
        return 0
    assert (group / "BACKEND.md").read_text(encoding="utf-8").splitlines()[0] == "COMPLETE"
    assert (group / "STATUS.md").read_text(encoding="utf-8").splitlines()[0] == "COMPLETE"
    journal = (group / ".backend-verified-source-ids").read_text(encoding="utf-8").splitlines()
    assert len(journal) == 1_792
    assert not list((group / "wavs").glob("*.wav"))
    rows = json.loads((group / "source-audit.json").read_text(encoding="utf-8"))
    assert len(rows) == 79 and all(row["complete"] for row in rows)
    expected_apks = set(FCBH_COMPLETE.read_text(encoding="utf-8").splitlines())
    assert {row["apk_name"] for row in rows} == expected_apks
    for row in rows:
        source = (STAGE_ROOT / row["slug"]).resolve()
        assert source.parent == STAGE_ROOT and source.exists(), row["slug"]
        shutil.rmtree(source)
        print(f"REMOVED {row['slug']}", flush=True)
    shutil.rmtree(group)
    print("REMOVED fcbh_group1", flush=True)
    return 80


def main() -> None:
    removed = cleanup_fcbh()
    for slug in REGISTRY.read_text(encoding="utf-8").splitlines():
        root = (STAGE_ROOT / slug).resolve()
        assert root.parent == STAGE_ROOT, f"invalid completed slug: {slug}"
        if not root.exists():
            continue
        marker = root / "BACKEND.md"
        status = root / "STATUS.md"
        assert marker.read_text(encoding="utf-8").splitlines()[0] == "COMPLETE", slug
        assert status.read_text(encoding="utf-8").splitlines()[0] == "COMPLETE", slug
        assert not (root / ".backend-verified-source-ids").exists(), slug
        shutil.rmtree(root)
        removed += 1
        print(f"REMOVED {slug}", flush=True)
    print(f"CLEANED completed_stage_folders={removed}", flush=True)


if __name__ == "__main__":
    main()
