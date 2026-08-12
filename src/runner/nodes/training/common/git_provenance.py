import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]


def require_clean_git_commit(repository: Path = REPOSITORY_ROOT) -> str:
    status = _git(repository, "status", "--porcelain", "--untracked-files=normal")
    if status:
        changed_paths = "\n".join(
            f"  {line}" for line in status.splitlines()
        )
        raise RuntimeError(
            "Training requires a clean Git worktree. Commit or remove these changes:\n"
            f"{changed_paths}"
        )
    return _git(repository, "rev-parse", "HEAD")


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Unable to determine git revision: {detail}")
    return result.stdout.strip()
