from __future__ import annotations

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.tmp_nodes.audio.datatypes import DENOISED_AUDIO, SPEAKER_CHUNK
from runflow.tmp_nodes.audio.models import DenoisedAudio, stable_id
from runflow.policies import BatchMode, BatchPolicy
from runflow.policies import ResourcePolicy


class DeepFilterNetNode(Node):
    NODE_TYPE = "DeepFilterNet"
    CATEGORY = "Audio / Enhancement"

    INPUTS = {
        "audio": Port("audio", SPEAKER_CHUNK),
    }
    OUTPUTS = {
        "audio": Port("audio", DENOISED_AUDIO),
    }

    BATCH_POLICY = BatchPolicy(
        BatchMode.MICRO_BATCH,
        preferred_size=16,
        max_size=64,
        sort_by="duration",
    )
    RESOURCE_POLICY = ResourcePolicy(
        resources={"accelerator": 1, "vram_gb": 3.0},
        keep_loaded=False,
        exclusive_group="accelerator",
        estimated_vram_gb=3.0,
        unload_after_stage=True,
    )

    async def setup(self, context):
        self.model = "placeholder_deepfilternet_model"

    async def teardown(self, context):
        self.model = None

    async def execute(self, batch, context):
        outputs = []
        out_dir = context.node_dir(self.id)
        for inputs in batch:
            chunk = inputs["audio"]
            denoised_id = stable_id("denoised", chunk.id)
            out_path = out_dir / f"{denoised_id}.wav"
            out_path.write_text(f"placeholder denoised audio from {chunk.path}\n", encoding="utf-8")
            denoised = DenoisedAudio(
                path=out_path,
                source_audio_id=chunk.source_audio_id,
                speaker=chunk.speaker,
                start=chunk.start,
                end=chunk.end,
                sample_rate=chunk.sample_rate,
                id=denoised_id,
                lineage_id=denoised_id,
                metadata={**chunk.metadata, "enhancement": "deepfilternet"},
            )
            outputs.append({"audio": denoised})
        return outputs
