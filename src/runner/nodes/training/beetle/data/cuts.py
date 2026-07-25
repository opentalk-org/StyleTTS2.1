import random

from .index import DatabaseSegmentIndex
from .records import CutRange, PlannedExample, SegmentKey


class CutPlanner:
    def __init__(
        self,
        index: DatabaseSegmentIndex,
        maximum_seconds: float,
    ) -> None:
        self.index = index
        self.maximum_seconds = maximum_seconds

    def plan_sentence(self, key: SegmentKey, seed: int) -> PlannedExample:
        item = self.index.records[key]
        return PlannedExample(
            key=key,
            target=CutRange(item.start, item.end),
            target_word_start=0,
            target_word_end=len(item.words),
            sentence=True,
            seed=seed,
        )

    def plan_mid_sentence(self, key: SegmentKey, seed: int) -> PlannedExample:
        item = self.index.records[key]
        candidates = []
        for start_index in range(len(item.words)):
            for end_index in range(start_index + 1, len(item.words) + 1):
                if start_index == 0 and end_index == len(item.words):
                    continue
                duration = item.words[end_index - 1].end - item.words[start_index].start
                if duration <= self.maximum_seconds:
                    candidates.append((start_index, end_index))
        start_index, end_index = random.Random(seed).choice(candidates)
        return PlannedExample(
            key=key,
            target=CutRange(item.words[start_index].start, item.words[end_index - 1].end),
            target_word_start=start_index,
            target_word_end=end_index,
            sentence=False,
            seed=seed,
        )
