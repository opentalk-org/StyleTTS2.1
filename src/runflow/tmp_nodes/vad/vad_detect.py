from __future__ import annotations

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.tmp_nodes.audio.datatypes import AUDIO_FILE, VAD_SEGMENTS
from runflow.tmp_nodes.audio.models import VadSegment, VadSegments, stable_id
from runflow.policies import BatchMode, BatchPolicy
from runflow.policies import ResourcePolicy


class VADDetectNode(Node):
    NODE_TYPE = "VADDetect"
    CATEGORY = "Audio / Segmentation"

    INPUTS = {
        "audio": Port("audio", AUDIO_FILE),
    }
    OUTPUTS = {
        "segments": Port("segments", VAD_SEGMENTS),
    }

    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=16, max_size=32, group_by=("sample_rate",))
    RESOURCE_POLICY = ResourcePolicy(resources={"cpu_workers": 1}, keep_loaded=True)

    def execute(self, batch, context):
        outputs = []
        max_segment_sec = float(self.params.get("max_segment_sec", 30.0))
        padding = float(self.params.get("padding_sec", 0.1))

        for inputs in batch:
            audio = inputs["audio"]
            segments = []
            cursor = 0.0
            while cursor < audio.duration:
                end = min(audio.duration, cursor + max_segment_sec)
                segments.append(VadSegment(start=max(0.0, cursor - padding), end=end, confidence=0.95))
                cursor = end

            vad = VadSegments(
                segments=segments,
                source_audio_id=audio.id,
                id=stable_id("vad", audio.id, len(segments)),
                lineage_id=audio.lineage_id,
                metadata={**audio.metadata, "num_vad_segments": len(segments)},
            )
            outputs.append({"segments": vad})
        return outputs
