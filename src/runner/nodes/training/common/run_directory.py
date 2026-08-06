import fcntl
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def claim_run_dir(run_dir: Path) -> Iterator[None]:
    lease_dir = run_dir.parent / ".leases"
    lease_dir.mkdir(parents=True, exist_ok=True)
    lease_path = lease_dir / f"{run_dir.name}.lock"
    with lease_path.open("w", encoding="utf-8") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"training run {run_dir.name!r} is already active"
            ) from error
        yield


def remove_run_dir(run_dir: Path) -> None:
    shutil.rmtree(run_dir, ignore_errors=True)
