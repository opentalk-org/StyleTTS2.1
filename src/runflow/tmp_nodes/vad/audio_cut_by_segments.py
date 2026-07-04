from __future__ import annotations

from runflow.core.node import Node
from runflow.core.ports import Port, PortMode
from runflow.tmp_nodes.audio.datatypes import AUDIO_CHUNK, AUDIO_FILE, VAD_SEGMENTS
from runflow.tmp_nodes.audio.models import AudioChunk, stable_id
from runflow.policies import BatchMode, BatchPolicy
from runflow.policies import ResourcePolicy


class AudioCutBySegmentsNode(Node):
    NODE_TYPE = "AudioCutBySegments"
    CATEGORY = "Audio / Segmentation"

    INPUTS = {
        "audio": Port("audio", AUDIO_FILE),
        "segments": Port("segments", VAD_SEGMENTS),
    }
    OUTPUTS = {
        "chunks": Port("chunks", AUDIO_CHUNK, mode=PortMode.STREAM),
    }

    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=32, max_size=64, group_by=("sample_rate",))
    RESOURCE_POLICY = ResourcePolicy(resources={"cpu_workers": 1}, keep_loaded=True)

    def execute(self, batch, context):
        outputs = []
        out_dir = context.node_dir(self.id)
        for inputs in batch:
            audio = inputs["audio"]
            vad = inputs["segments"]
            chunks = []
            for index, segment in enumerate(vad.segments):
                chunk_id = stable_id("chunk", audio.id, index, segment.start, segment.end)
                out_path = out_dir / f"{chunk_id}.wav"
                out_path.write_text(
                    f"placeholder VAD chunk {index} from {audio.path}: {segment.start}-{segment.end}\n",
                    encoding="utf-8",
                )
                chunk = AudioChunk(
                    path=out_path,
                    source_audio_id=audio.id,
                    start=segment.start,
                    end=segment.end,
                    sample_rate=audio.sample_rate,
                    id=chunk_id,
                    lineage_id=chunk_id,
                    metadata={
                        **audio.metadata,
                        "vad_index": index,
                        "start": segment.start,
                        "end": segment.end,
                        "duration": segment.end - segment.start,
                        "duration_bucket": self._duration_bucket(segment.end - segment.start),
                    },
                )
                chunks.append(chunk)
            outputs.append({"chunks": chunks})
        return outputs

    def _duration_bucket(self, duration: float) -> str:
        if duration < 15:
            return "short"
        if duration < 45:
            return "medium"
        return "long"
