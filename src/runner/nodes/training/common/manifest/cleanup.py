from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Orphans are per-run dirs a hard-killed run (OOM, segfault) never cleaned up;
# normal success/failure already deletes its own dir. The age guard is large
# because a long-running sibling run keeps its dir's mtime at build time, and we
# must never sweep one that is still training.
ORPHAN_MAX_AGE_SECONDS = 48 * 60 * 60


def remove_run_dir(run_dir: Path) -> None:
    """Delete all data a training run prepared on disk (lists, cache, wavs)."""
    shutil.rmtree(run_dir, ignore_errors=True)


def sweep_orphan_run_dirs(root: Path, keep_run_id: str, max_age_seconds: int = ORPHAN_MAX_AGE_SECONDS) -> None:
    """Remove stale per-run manifest dirs left behind by earlier runs.

    Each run cleans its own dir when training finishes, so anything older than
    the age guard is an orphan. The current run's dir is always kept."""
    if not root.is_dir():
        return
    cutoff = time.time() - max_age_seconds
    for child in root.iterdir():
        if not child.is_dir() or child.name == keep_run_id:
            continue
        if child.stat().st_mtime >= cutoff:
            continue
        logger.info("removing orphan training manifest dir %s", child)
        shutil.rmtree(child, ignore_errors=True)
