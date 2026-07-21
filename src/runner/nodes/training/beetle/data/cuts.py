import random

from .index import DatabaseSegmentIndex
from .records import ContextRange, CutRange, IndexedSegment, PlannedExample, SegmentKey


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
        pre, post = self._boundary_context(item, 0, len(item.words))
        return PlannedExample(
            key=key,
            target=CutRange(item.start, item.end),
            target_word_start=0,
            target_word_end=len(item.words),
            pre_context=pre,
            post_context=post,
            sentence=True,
            seed=seed,
        )

    def plan_mid_sentence(self, key: SegmentKey, seed: int) -> PlannedExample:
        item = self.index.records[key]
        if not item.mid_sentence_eligible:
            raise ValueError(f"segment is not mid-sentence eligible: {key}")
        candidates = []
        for start_index in range(len(item.words)):
            for end_index in range(start_index + 1, len(item.words) + 1):
                if start_index == 0 and end_index == len(item.words):
                    continue
                duration = item.words[end_index - 1].end - item.words[start_index].start
                if duration <= self.maximum_seconds:
                    candidates.append((start_index, end_index))
        if not candidates:
            raise ValueError(f"no valid aligned mid-sentence cut: {key}")
        start_index, end_index = random.Random(seed).choice(candidates)
        pre, post = self._boundary_context(item, start_index, end_index)
        return PlannedExample(
            key=key,
            target=CutRange(item.words[start_index].start, item.words[end_index - 1].end),
            target_word_start=start_index,
            target_word_end=end_index,
            pre_context=pre,
            post_context=post,
            sentence=False,
            seed=seed,
        )

    def _boundary_context(
        self,
        item: IndexedSegment,
        word_start: int,
        word_end: int,
    ) -> tuple[ContextRange | None, ContextRange | None]:
        pre = self._excluded_pre(item, word_start)
        post = self._excluded_post(item, word_end)
        neighbours = self.index.by_audio[item.key.audio_file_id]
        position = next(index for index, value in enumerate(neighbours) if value.key == item.key)
        if pre is None and position > 0:
            pre = self._whole_context(neighbours[position - 1])
        if post is None and position + 1 < len(neighbours):
            post = self._whole_context(neighbours[position + 1])
        return pre, post

    @staticmethod
    def _excluded_pre(item: IndexedSegment, word_start: int) -> ContextRange | None:
        if word_start == 0:
            return None
        return ContextRange(
            key=item.key,
            audio=CutRange(item.start, item.words[word_start - 1].end),
            word_start=0,
            word_end=word_start,
            voice_id=item.voice_id,
        )

    @staticmethod
    def _excluded_post(item: IndexedSegment, word_end: int) -> ContextRange | None:
        if word_end == len(item.words):
            return None
        return ContextRange(
            key=item.key,
            audio=CutRange(item.words[word_end].start, item.end),
            word_start=word_end,
            word_end=len(item.words),
            voice_id=item.voice_id,
        )

    @staticmethod
    def _whole_context(item: IndexedSegment) -> ContextRange:
        return ContextRange(
            key=item.key,
            audio=CutRange(item.start, item.end),
            word_start=0,
            word_end=len(item.words),
            voice_id=item.voice_id,
        )
