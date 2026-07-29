from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SelectionItem:
    source_id: str
    speaker_id: str | None
    duration: float


def select_duration(items: Iterable[SelectionItem], target_seconds: float) -> list[SelectionItem]:
    if target_seconds <= 0.0:
        raise ValueError("target duration must be positive")
    speakers: dict[str | None, deque[SelectionItem]] = defaultdict(deque)
    order: list[str | None] = []
    for item in items:
        if item.duration <= 0.0:
            raise ValueError(f"{item.source_id}: duration must be positive")
        if item.speaker_id not in speakers:
            order.append(item.speaker_id)
        speakers[item.speaker_id].append(item)
    selected: list[SelectionItem] = []
    duration = 0.0
    while duration < target_seconds and speakers:
        for speaker_id in order:
            queue = speakers.get(speaker_id)
            if queue is None:
                continue
            item = queue.popleft()
            selected.append(item)
            duration += item.duration
            if not queue:
                del speakers[speaker_id]
            if duration >= target_seconds:
                break
    return selected
