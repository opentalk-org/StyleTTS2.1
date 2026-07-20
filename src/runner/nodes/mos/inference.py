from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from uuid import UUID

from pydantic import Field
import torch

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import AudioPort, CheckpointRefPort, JsonPort
from runner.nodes.models import Audio, CheckpointRef, typed_checkpoint
from runner.nodes.mos.audio import prepare_audio_batch
from runner.nodes.mos.model import MosModelBundle, load_trained_mos_bundle
from shared.db import database_session
from shared.db.audio import crud as audio_crud


class PredictMosScoreSettings(StrictSettings):
    inference_batch_size: int = Field(default=8, title="Inference batch size", ge=1, le=64)


class PredictMosScoreNode(Node):
    NODE_TYPE = "PredictMosScore"
    DESCRIPTION = "Predict a MOS (mean opinion score) quality rating for each incoming audio clip using a trained MOS model checkpoint. Takes a MOS-model checkpoint and audio, writes the predicted scores back to each audio file in the database, and passes the audio through unchanged alongside a writeback result carrying the score. Use it to automatically rate the perceived quality of generated or recorded speech."
    CATEGORY = "Audio"
    SETTINGS = PredictMosScoreSettings
    INPUTS = {
        "checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST),
        "audio": AudioPort(),
    }
    OUTPUTS = {"audio": AudioPort(), "writeback_result": JsonPort()}
    BATCH_POLICY = BatchPolicy(
        BatchMode.MICRO_BATCH,
        preferred_size=16,
        max_size=32,
        sort_by="duration",
    )
    RESOURCE_POLICY = ResourcePolicy(
        resources={"accelerator": 1, "vram_gb": 6},
        keep_loaded=True,
        exclusive_group="accelerator",
    )
    QUEUE_MAX_SIZE = 64

    def __init__(self, node_id: str | None = None, **params):
        super().__init__(node_id=node_id, **params)
        self._bundle: MosModelBundle | None = None
        self._loaded_checkpoint_id: UUID | None = None
        self._device: torch.device | None = None

    async def teardown(self, context) -> None:
        self._bundle = None
        self._loaded_checkpoint_id = None
        self._device = None
        release_accelerator_memory()

    async def execute(self, batch, context):
        checkpoint = typed_checkpoint(batch[0]["checkpoint"])
        await self._ensure_bundle(checkpoint, torch.device(str(context.device)))
        audios: list[Audio] = [inputs["audio"] for inputs in batch]
        predictions: list[float] = []
        for start in range(0, len(audios), self.settings.inference_batch_size):
            context.check_cancel()
            chunk = audios[start:start + self.settings.inference_batch_size]
            predictions.extend(self._predict(chunk))
            await context.report_progress(
                self.id,
                min(start + len(chunk), len(audios)),
                len(audios),
                f"MOS scored {min(start + len(chunk), len(audios))}/{len(audios)} audio files",
            )
        scores = {audio.audio_file_id: score for audio, score in zip(audios, predictions, strict=True)}
        with database_session() as session:
            audio_crud.bulk_update_audio_scores(session, scores)
        return [
            {
                "audio": replace(
                    audio,
                    annotations=audio.annotations.model_copy(update={"score": score}),
                ),
                "writeback_result": {
                    "audio_file_id": str(audio.audio_file_id),
                    "score": score,
                },
            }
            for audio, score in zip(audios, predictions, strict=True)
        ]

    async def _ensure_bundle(self, checkpoint: CheckpointRef, device: torch.device) -> None:
        if checkpoint.metadata["type"] != "mos_model":
            raise ValueError(f"PredictMosScore requires mos_model checkpoint: {checkpoint.checkpoint_id}")
        if self._bundle is not None and self._loaded_checkpoint_id == checkpoint.checkpoint_id:
            return
        self._bundle = await asyncio.to_thread(self._load_bundle_logged, checkpoint.path, device)
        self._loaded_checkpoint_id = checkpoint.checkpoint_id
        self._device = device

    def _load_bundle_logged(self, checkpoint_dir: Path, device: torch.device) -> MosModelBundle:
        self.logger.info("loading MOS model checkpoint=%s", checkpoint_dir)
        bundle = load_trained_mos_bundle(checkpoint_dir, device)
        bundle.model.eval()
        return bundle

    def _predict(self, audios: list[Audio]) -> list[float]:
        assert self._bundle is not None, "MOS model bundle is not loaded"
        assert self._device is not None, "MOS inference device is not set"
        inputs = prepare_audio_batch(self._bundle.feature_extractor, load_audio_bytes(audios)).to(self._device)
        amp_enabled = self._device.type == "cuda"
        with torch.no_grad(), torch.autocast(
            device_type=self._device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            predictions = self._bundle.model(inputs.input_values, inputs.attention_mask)
        if not torch.isfinite(predictions).all():
            raise RuntimeError("MOS model produced non-finite scores")
        return [float(score) for score in predictions.float().cpu().tolist()]


def load_audio_bytes(audios: list[Audio]) -> list[bytes]:
    missing_ids = [audio.audio_file_id for audio in audios if audio.data is None]
    stored: dict[UUID, bytes] = {}
    if missing_ids:
        with database_session() as session:
            stored = audio_crud.bulk_read_audio_files(session, missing_ids)
    return [audio.data if audio.data is not None else stored[audio.audio_file_id] for audio in audios]
