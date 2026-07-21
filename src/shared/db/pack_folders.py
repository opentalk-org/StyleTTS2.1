import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session


@dataclass
class PackFolderAllocator:
    session: Session
    model: type[Any]
    prefix: str
    target_files: int = 256
    _database_counts: dict[str, int] | None = field(init=False, default=None)
    _pending: Counter[str] = field(init=False, default_factory=Counter)

    def __post_init__(self) -> None:
        if self.target_files < 1:
            raise ValueError("pack folder target must be positive")

    def path_for(self, pack_id: uuid.UUID) -> str:
        folder = self._folder_with_capacity()
        self._pending[folder] += 1
        return f"{self.prefix}/{folder}/{pack_id}.bin"

    def _folder_with_capacity(self) -> str:
        counts = self._counts()
        candidates = [
            folder
            for folder, count in counts.items()
            if count + self._pending[folder] < self.target_files
        ]
        if candidates:
            return min(candidates, key=lambda folder: (counts[folder] + self._pending[folder], folder))
        folder = str(uuid.uuid4())
        counts[folder] = 0
        return folder

    def _counts(self) -> dict[str, int]:
        if self._database_counts is None:
            statement = select(self.model.path).where(self.model.path.like(f"{self.prefix}/%/%"))
            paths = self.session.execute(statement).scalars()
            self._database_counts = dict(Counter(self._folder(path) for path in paths))
        return self._database_counts

    def _folder(self, path: str) -> str:
        relative = path.removeprefix(f"{self.prefix}/")
        folder, filename = relative.split("/", 1)
        if not folder or "/" in filename or not filename:
            raise ValueError(f"invalid sharded pack path: {path}")
        return folder
