from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.tmp_nodes.audio.datatypes import AUDIO_LIKE, SAVE_RESULT, TRANSCRIPT
from runflow.tmp_nodes.audio.models import SaveResult, Transcript, stable_id
from runflow.policies import BatchMode, BatchPolicy
from runflow.policies import ResourcePolicy


class SaveArtifactSettings(StrictSettings):
    output_dir: Path | None = None
    sleep_sec: float = 0.0


class SaveAudioNode(Node):
    NODE_TYPE = "SaveAudio"
    CATEGORY = "Audio / IO"
    SETTINGS = SaveArtifactSettings

    INPUTS = {
        "audio": Port("audio", AUDIO_LIKE),
    }
    OUTPUTS = {
        "result": Port("result", SAVE_RESULT),
    }

    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=128)
    RESOURCE_POLICY = ResourcePolicy(resources={"cpu_workers": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        out_dir = self.settings.output_dir or context.output_dir / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for inputs in batch:
            if self.settings.sleep_sec > 0:
                await asyncio.sleep(self.settings.sleep_sec)
            audio = inputs["audio"]
            source_path = Path(audio.path)
            suffix = source_path.suffix or ".wav"
            out_path = out_dir / f"{audio.id}{suffix}"
            if source_path.exists():
                shutil.copyfile(source_path, out_path)
            else:
                out_path.write_text(f"placeholder audio for {audio.id}\n", encoding="utf-8")

            result = SaveResult(
                path=out_path,
                kind="audio",
                id=stable_id("save", out_path),
                lineage_id=getattr(audio, "lineage_id", audio.id),
                metadata={"source_id": audio.id},
            )
            outputs.append({"result": result})
        return outputs


class SaveTranscriptNode(Node):
    NODE_TYPE = "SaveTranscript"
    CATEGORY = "Audio / IO"
    SETTINGS = SaveArtifactSettings

    INPUTS = {
        "transcript": Port("transcript", TRANSCRIPT),
    }
    OUTPUTS = {
        "result": Port("result", SAVE_RESULT),
    }

    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=128)
    RESOURCE_POLICY = ResourcePolicy(resources={"cpu_workers": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        out_dir = self.settings.output_dir or context.output_dir / "transcripts"
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for inputs in batch:
            if self.settings.sleep_sec > 0:
                await asyncio.sleep(self.settings.sleep_sec)
            transcript: Transcript = inputs["transcript"]
            out_path = out_dir / f"{transcript.id}.json"
            payload: dict[str, Any] = {
                "id": transcript.id,
                "model": transcript.model,
                "text": transcript.text,
                "source_audio_id": transcript.source_audio_id,
                "start": transcript.start,
                "end": transcript.end,
                "speaker": transcript.speaker,
                "segments": transcript.segments,
                "metadata": transcript.metadata,
            }
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            result = SaveResult(
                path=out_path,
                kind="transcript",
                id=stable_id("save", out_path),
                lineage_id=transcript.lineage_id,
                metadata={"transcript_id": transcript.id, "model": transcript.model},
            )
            outputs.append({"result": result})
        return outputs
