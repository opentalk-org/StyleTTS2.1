from __future__ import annotations

from pathlib import Path

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.tmp_nodes.audio.datatypes import AUDIO_FILE, FLOAT, INT, JSON, PATH
from runflow.tmp_nodes.audio.models import AudioFile, stable_id
from runflow.policies import BatchMode, BatchPolicy
from runflow.policies import ResourcePolicy


class LoadAudioNode(Node):
    NODE_TYPE = "LoadAudio"
    CATEGORY = "Audio / IO"

    INPUTS = {
        "path": Port("path", PATH),
    }
    OUTPUTS = {
        "audio": Port("audio", AUDIO_FILE),
        "sample_rate": Port("sample_rate", INT),
        "duration": Port("duration", FLOAT),
        "metadata": Port("metadata", JSON),
    }

    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=32, max_size=64)
    RESOURCE_POLICY = ResourcePolicy(resources={"cpu_workers": 1}, keep_loaded=True)

    def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            path = Path(inputs["path"])
            sample_rate = int(self.params.get("sample_rate", 16000))
            channels = int(self.params.get("channels", 1))
            duration = self._fake_probe_duration(path)
            audio_id = stable_id("audio", path.resolve())

            audio = AudioFile(
                path=path,
                sample_rate=sample_rate,
                channels=channels,
                duration=duration,
                id=audio_id,
                lineage_id=audio_id,
                metadata={
                    "source_path": str(path),
                    "sample_rate": sample_rate,
                    "duration": duration,
                    "duration_bucket": self._duration_bucket(duration),
                },
            )
            outputs.append(
                {
                    "audio": audio,
                    "sample_rate": sample_rate,
                    "duration": duration,
                    "metadata": audio.metadata,
                }
            )
        return outputs

    def _fake_probe_duration(self, path: Path) -> float:
        if path.exists():
            return max(12.0, min(180.0, path.stat().st_size / 100.0))
        return 60.0

    def _duration_bucket(self, duration: float) -> str:
        if duration < 30:
            return "short"
        if duration < 90:
            return "medium"
        return "long"
