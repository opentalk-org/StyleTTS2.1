from __future__ import annotations

from dataclasses import replace
from typing import Any

from runflow.core.node import Node
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio
from runner.nodes.text_processing.polish_numbers import normalize_polish_numbers


class NormalizePolishNumbersNode(Node):
    NODE_TYPE = "NormalizePolishNumbers"
    DESCRIPTION = "Conservatively expand recognized Polish dates, years, decimals, percentages, times, ranges, dimensions, and ordinals. If any digit remains after recognition, the complete segment is returned unchanged instead of guessing. This node only changes in-memory graph values; connect an explicit save node to persist results."
    CATEGORY = "Text"
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=256, max_size=512, timeout_ms=25)
    RESOURCE_POLICY = ResourcePolicy(resources={"cpu_workers": 1}, keep_loaded=True)
    QUEUE_MAX_SIZE = 1_024

    async def execute(self, batch: list[dict[str, Audio]], context: Any) -> list[dict[str, Audio]]:
        outputs = []
        for inputs in batch:
            context.check_cancel()
            audio = inputs["audio"]
            if str(audio.language).lower().replace("_", "-").split("-", 1)[0] != "pl":
                raise ValueError(f"polish_normalization_language_mismatch: audio {audio.audio_file_id} has language {audio.language!r}")
            segments = [replace(segment, text=normalize_polish_numbers(segment.text)) for segment in audio.segments]
            outputs.append({"audio": replace(audio, segments=segments)})
        return outputs
