from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic import Field

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy
from runner.nodes.audio_segments.break_alignment import annotate_segment
from runner.nodes.audio_segments.silence_detection import detect_silence_intervals
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio


class InsertSilenceBreaksSettings(StrictSettings):
    silence_threshold: float = Field(default=0.01, ge=0.0, le=1.0, title="Silence RMS threshold")
    window_size: int = Field(default=20, gt=0, title="RMS window size (ms)")
    max_silence_gap: int = Field(default=80, ge=0, title="Maximum silence gap (ms)")
    min_break_time: int = Field(default=100, gt=0, title="Minimum break time (ms)")
    word_overlap_drop_ratio: float = Field(default=0.5, ge=0.0, le=1.0, title="Word overlap drop ratio")
    insert_at_start: bool = Field(default=False, title="Insert at start")
    insert_at_end: bool = Field(default=False, title="Insert at end")
    drop_prob: float = Field(default=0.0, ge=0.0, le=1.0, title="Break drop probability")


class InsertSilenceBreaksNode(Node):
    NODE_TYPE = "InsertSilenceBreaks"
    DESCRIPTION = "Detect fixed-window RMS silence and insert timed break tokens into aligned segment text and word alignments. Segments without alignments pass through unchanged."
    CATEGORY = "Audio"
    SETTINGS = InsertSilenceBreaksSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=64)

    async def execute(self, batch: list[dict[str, Any]], context: Any) -> list[dict[str, Audio]]:
        outputs = []
        for inputs in batch:
            context.check_cancel()
            audio = inputs["audio"]
            assert isinstance(audio, Audio), "silence break input must be Audio"
            if audio.data is None:
                raise ValueError(f"audio bytes are required for silence break detection: {audio.id}")
            silences = detect_silence_intervals(
                audio.data,
                audio.start,
                self.settings.silence_threshold,
                self.settings.window_size,
                self.settings.max_silence_gap,
            )
            segments = []
            for segment in audio.segments:
                context.check_cancel()
                segments.append(
                    annotate_segment(
                        segment,
                        silences,
                        self.settings.min_break_time,
                        self.settings.insert_at_start,
                        self.settings.insert_at_end,
                        self.settings.drop_prob,
                        self.settings.word_overlap_drop_ratio,
                    )
                )
            outputs.append({"audio": replace(audio, segments=segments)})
        return outputs
