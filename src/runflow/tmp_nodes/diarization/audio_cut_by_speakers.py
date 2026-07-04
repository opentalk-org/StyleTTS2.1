from __future__ import annotations

from runflow.core.node import Node
from runflow.core.ports import Port, PortMode
from runflow.tmp_nodes.audio.datatypes import AUDIO_CHUNK, DIARIZATION_RESULT, SPEAKER_CHUNK
from runflow.tmp_nodes.audio.models import SpeakerChunk, stable_id
from runflow.policies import BatchMode, BatchPolicy
from runflow.policies import ResourcePolicy


class AudioCutBySpeakersNode(Node):
    NODE_TYPE = "AudioCutBySpeakers"
    CATEGORY = "Audio / Diarization"

    INPUTS = {
        "audio": Port("audio", AUDIO_CHUNK),
        "diarization": Port("diarization", DIARIZATION_RESULT),
    }
    OUTPUTS = {
        "speaker_chunks": Port("speaker_chunks", SPEAKER_CHUNK, mode=PortMode.STREAM),
    }

    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=32, max_size=64, group_by=("sample_rate",))
    RESOURCE_POLICY = ResourcePolicy(resources={"cpu_workers": 1}, keep_loaded=True)

    def execute(self, batch, context):
        outputs = []
        out_dir = context.node_dir(self.id)
        for inputs in batch:
            audio = inputs["audio"]
            diarization = inputs["diarization"]
            speaker_chunks = []
            for index, turn in enumerate(diarization.turns):
                speaker_id = stable_id("speaker", audio.id, index, turn.speaker, turn.start, turn.end)
                out_path = out_dir / f"{speaker_id}.wav"
                out_path.write_text(
                    f"placeholder speaker chunk {turn.speaker} from {audio.path}: {turn.start}-{turn.end}\n",
                    encoding="utf-8",
                )
                speaker_chunks.append(
                    SpeakerChunk(
                        path=out_path,
                        source_audio_id=audio.source_audio_id,
                        speaker=turn.speaker,
                        start=turn.start,
                        end=turn.end,
                        sample_rate=audio.sample_rate,
                        id=speaker_id,
                        lineage_id=speaker_id,
                        metadata={
                            **audio.metadata,
                            "speaker": turn.speaker,
                            "speaker_index": index,
                            "start": turn.start,
                            "end": turn.end,
                            "duration": turn.end - turn.start,
                            "duration_bucket": self._duration_bucket(turn.end - turn.start),
                        },
                    )
                )
            outputs.append({"speaker_chunks": speaker_chunks})
        return outputs

    def _duration_bucket(self, duration: float) -> str:
        if duration < 15:
            return "short"
        if duration < 45:
            return "medium"
        return "long"
