from __future__ import annotations

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runner.nodes.datatypes import TRANSCRIPT
from runner.nodes.models import Transcript, stable_id


class PhonemizeSettings(StrictSettings):
    language: str = "en-us"
    workers: int = Field(default=4, ge=1, le=64)


class PhonemizeTranscriptNode(Node):
    NODE_TYPE = "PhonemizeTranscript"
    CATEGORY = "Text"
    SETTINGS = PhonemizeSettings
    INPUTS = {"transcript": Port("transcript", TRANSCRIPT)}
    OUTPUTS = {"transcript": Port("transcript", TRANSCRIPT)}

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            transcript: Transcript = inputs["transcript"]
            metadata = {**transcript.metadata, "phoneme_language": self.settings.language}
            outputs.append({
                "transcript": Transcript(
                    transcript.text,
                    transcript.model,
                    transcript.source_audio_id,
                    transcript.start,
                    transcript.end,
                    transcript.speaker,
                    stable_id("phon", transcript.id, self.settings.language),
                    transcript.lineage_id,
                    transcript.segments,
                    metadata,
                )
            })
        return outputs
