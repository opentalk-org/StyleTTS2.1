from __future__ import annotations

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.tmp_nodes.audio.datatypes import AUDIO_LIKE, TRANSCRIPT
from runflow.tmp_nodes.audio.models import Transcript, stable_id
from runflow.policies import ResourcePolicy


class ASRNode(Node):
    CATEGORY = "Audio / ASR"
    MODEL_NAME = "asr"

    INPUTS = {
        "audio": Port("audio", AUDIO_LIKE),
    }
    OUTPUTS = {
        "transcript": Port("transcript", TRANSCRIPT),
    }

    RESOURCE_POLICY = ResourcePolicy(
        resources={"accelerator": 1, "vram_gb": 8.0},
        keep_loaded=False,
        exclusive_group="accelerator",
        estimated_vram_gb=8.0,
        unload_after_stage=True,
    )

    def setup(self, context):
        self.model = f"placeholder_{self.MODEL_NAME}_model"

    def teardown(self, context):
        self.model = None

    def execute(self, batch, context):
        outputs = []
        language = self.params.get("language", "auto")
        for inputs in batch:
            audio = inputs["audio"]
            transcript_id = stable_id("transcript", self.MODEL_NAME, audio.id)
            text = (
                f"[{self.MODEL_NAME}] fake transcript for {audio.id} "
                f"speaker={getattr(audio, 'speaker', None)} "
                f"time={getattr(audio, 'start', None)}-{getattr(audio, 'end', None)}"
            )
            transcript = Transcript(
                text=text,
                model=self.MODEL_NAME,
                source_audio_id=audio.source_audio_id,
                start=getattr(audio, "start", None),
                end=getattr(audio, "end", None),
                speaker=getattr(audio, "speaker", None),
                id=transcript_id,
                lineage_id=stable_id("asr_lineage", self.MODEL_NAME, audio.id),
                segments=[{"start": getattr(audio, "start", None), "end": getattr(audio, "end", None), "text": text}],
                metadata={**getattr(audio, "metadata", {}), "asr_model": self.MODEL_NAME, "language": language},
            )
            outputs.append({"transcript": transcript})
        return outputs
