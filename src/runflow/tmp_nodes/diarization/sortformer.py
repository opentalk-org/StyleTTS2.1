from __future__ import annotations

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.tmp_nodes.audio.datatypes import AUDIO_CHUNK, DIARIZATION_RESULT
from runflow.tmp_nodes.audio.models import DiarizationResult, SpeakerTurn, stable_id
from runflow.policies import BatchMode, BatchPolicy
from runflow.policies import ResourcePolicy


class SortformerDiarizationNode(Node):
    NODE_TYPE = "SortformerDiarization"
    CATEGORY = "Audio / Diarization"

    INPUTS = {
        "audio": Port("audio", AUDIO_CHUNK),
    }
    OUTPUTS = {
        "diarization": Port("diarization", DIARIZATION_RESULT),
    }

    BATCH_POLICY = BatchPolicy(
        BatchMode.MICRO_BATCH,
        preferred_size=4,
        max_size=8,
        group_by=("sample_rate", "duration_bucket"),
        sort_by="duration",
    )
    RESOURCE_POLICY = ResourcePolicy(
        resources={"accelerator": 1, "vram_gb": 4.0},
        keep_loaded=False,
        exclusive_group="accelerator",
        estimated_vram_gb=4.0,
        unload_after_stage=True,
    )

    def setup(self, context):
        self.model = "placeholder_sortformer_model"

    def teardown(self, context):
        self.model = None

    def execute(self, batch, context):
        outputs = []
        max_speakers = int(self.params.get("max_speakers", 2))
        for inputs in batch:
            chunk = inputs["audio"]
            midpoint = chunk.start + chunk.duration / 2.0
            turns = [SpeakerTurn(start=chunk.start, end=midpoint, speaker="SPEAKER_00", confidence=0.9)]
            if max_speakers > 1 and midpoint < chunk.end:
                turns.append(SpeakerTurn(start=midpoint, end=chunk.end, speaker="SPEAKER_01", confidence=0.88))

            diarization = DiarizationResult(
                turns=turns,
                source_audio_id=chunk.source_audio_id,
                id=stable_id("diar", chunk.id, max_speakers),
                lineage_id=chunk.lineage_id,
                metadata={**chunk.metadata, "num_speaker_turns": len(turns), "model": "sortformer"},
            )
            outputs.append({"diarization": diarization})
        return outputs
