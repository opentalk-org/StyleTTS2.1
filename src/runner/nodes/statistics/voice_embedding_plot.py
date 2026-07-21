from __future__ import annotations

import asyncio
import importlib
import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import AudioPort, CheckpointRefPort, SaveResultPort
from runner.nodes.models import Audio, CheckpointRef, SaveResult, stable_id, typed_checkpoint
from runner.nodes.statistics.voice_embedding_html import PlotPoint, render_voice_plot_html
from runner.nodes.synthesis.styletts_runtime.checkpoints import latest_weight, resolve_main_checkpoint
from runner.nodes.synthesis.styletts_runtime.helpers import compute_style_from_wave, training_import_context
from shared.db import database_session
from shared.db.assets import crud as asset_crud
from shared.db.assets.schemas import ExtraFileCreate
from shared.db.audio import crud as audio_crud


UNASSIGNED_SPEAKER = "unassigned"


class VoiceEmbeddingPlotSettings(StrictSettings):
    embedding_component: Literal["both", "acoustic", "prosodic"] = Field(default="both", title="Style component")
    title: str = Field(default="Voice style embeddings (PCA)", title="Plot title")
    point_size: int = Field(default=44, title="Point size", ge=4, le=400)
    artifact_name: str = Field(default="voice_embedding_pca.html", title="Artifact name")


@dataclass
class StyleEncoderRuntime:
    style_encoder: Any
    predictor_encoder: Any
    device: Any


@dataclass
class EmbeddedPoint:
    vector: Any
    label: str
    name: str


@dataclass
class LoadedAudioInput:
    inputs: dict[str, Any]
    audio: Audio
    data: bytes
    segments: list[dict[str, Any]]


class EmbedVoicesPcaPlotNode(Node):
    NODE_TYPE = "EmbedVoicesPcaPlot"
    DESCRIPTION = "Embed every incoming audio clip with a StyleTTS2 style encoder, project the style vectors down to 2D with PCA, and produce an interactive HTML scatter plot artifact. Points can be coloured by voice or speaker and use the acoustic, prosodic, or combined style component. Use it to visually inspect how voices cluster and spot outliers or mislabelled samples; wire in the StyleTTS2 checkpoint to embed with on the checkpoint input."
    CATEGORY = "Statistics"
    SETTINGS = VoiceEmbeddingPlotSettings
    INPUTS = {"audio": AudioPort(), "checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST)}
    OUTPUTS = {"artifact": SaveResultPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, exclusive_group="accelerator", keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params: Any):
        super().__init__(node_id=node_id, **params)
        self._runtime: StyleEncoderRuntime | None = None
        self._pending: dict[str, list[EmbeddedPoint]] = {}

    async def teardown(self, context) -> None:
        self._runtime = None
        release_accelerator_memory()

    async def execute(self, batch, context):
        outputs = []
        loaded = _load_audio_batch(list(batch))
        for item in loaded:
            context.check_cancel()
            result = await asyncio.to_thread(
                self._ingest,
                item,
                str(context.run_id),
            )
            if result is not None:
                outputs.append(result)
        return outputs

    def _ingest(
        self,
        item: LoadedAudioInput,
        run_id: str,
    ) -> dict[str, SaveResult] | None:
        audio = item.audio
        if self._runtime is None:
            self._runtime = _load_style_encoder_runtime(
                typed_checkpoint(item.inputs["checkpoint"])
            )
        batch_id = str(audio.metadata["source_batch_id"])
        expected = int(audio.metadata["source_batch_count"])
        label = _speaker_label(item.segments)
        vector = _style_vector(
            self._runtime,
            item.data,
            self.settings.embedding_component,
        )
        point = EmbeddedPoint(vector=vector, label=label, name=audio.name)
        pending = self._pending.setdefault(batch_id, [])
        pending.append(point)
        if len(pending) < expected:
            return None
        del self._pending[batch_id]
        return {"artifact": _render_and_store(pending, self.settings, run_id)}


def _load_audio_batch(batch: list[dict[str, Any]]) -> list[LoadedAudioInput]:
    audios = [inputs["audio"] for inputs in batch]
    assert all(isinstance(audio, Audio) for audio in audios), "voice embedding inputs must be Audio"
    audio_ids = [audio.audio_file_id for audio in audios]
    missing_ids = [audio.audio_file_id for audio in audios if audio.data is None]
    with database_session() as session:
        records = audio_crud.get_audio_files_bulk(session, audio_ids)
        stored_data = (
            audio_crud.bulk_read_audio_files(session, missing_ids)
            if missing_ids
            else {}
        )
    loaded = [
        LoadedAudioInput(
            inputs=inputs,
            audio=audio,
            data=audio.data if audio.data is not None else stored_data[audio.audio_file_id],
            segments=list(records[audio.audio_file_id].segments),
        )
        for inputs, audio in zip(batch, audios, strict=True)
    ]
    return loaded


def _load_style_encoder_runtime(checkpoint: CheckpointRef) -> StyleEncoderRuntime:
    main = resolve_main_checkpoint(checkpoint)
    weights_path = latest_weight(main.root)
    config = yaml.safe_load((main.root / "config.yml").read_text(encoding="utf-8"))
    params = config["model_params"]
    torch = importlib.import_module("torch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with training_import_context():
        style_encoder_cls = importlib.import_module("modules.encoders").StyleEncoder
        style_encoder = style_encoder_cls(dim_in=params["dim_in"], style_dim=params["style_dim"], max_conv_dim=params["hidden_dim"])
        predictor_encoder = style_encoder_cls(dim_in=params["dim_in"], style_dim=params["style_dim"], max_conv_dim=params["hidden_dim"])
        net = torch.load(str(weights_path), map_location="cpu", weights_only=False)["net"]
        style_encoder.load_state_dict(_strip_module_prefix(net["style_encoder"]), strict=False)
        predictor_encoder.load_state_dict(_strip_module_prefix(net["predictor_encoder"]), strict=False)
    style_encoder.eval().to(device)
    predictor_encoder.eval().to(device)
    return StyleEncoderRuntime(style_encoder, predictor_encoder, device)


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key[len("module."):] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def _style_vector(runtime: StyleEncoderRuntime, wav_bytes: bytes, component: str) -> Any:
    import numpy as np
    import soundfile

    wave, sr = soundfile.read(io.BytesIO(wav_bytes))
    if wave.ndim > 1:
        wave = wave[:, 0]
    embedding = compute_style_from_wave(runtime, wave.astype(np.float32), sr=int(sr), device=runtime.device)
    vector = embedding.squeeze(0).detach().cpu().numpy().astype(np.float64)
    half = vector.shape[0] // 2
    if component == "acoustic":
        return vector[:half]
    if component == "prosodic":
        return vector[half:]
    return vector


def _speaker_label(segments: list[dict[str, Any]]) -> str:
    weight: Counter[str] = Counter()
    for segment in segments:
        duration = max(0.0, float(segment["end"]) - float(segment["start"]))
        annotations = segment["annotations"]
        assert isinstance(annotations, dict), "segment annotations must be an object"
        key = str(annotations["speaker_id"]) if annotations["speaker_id"] else UNASSIGNED_SPEAKER
        weight[key] += duration if duration > 0 else 1.0
    if not weight:
        return UNASSIGNED_SPEAKER
    return weight.most_common(1)[0][0]


def _pca_coordinates(vectors: list[Any]) -> Any:
    import numpy as np

    matrix = np.stack(vectors)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ components[:2].T
    if projection.shape[1] == 1:
        projection = np.hstack([projection, np.zeros((projection.shape[0], 1))])
    return projection


def _render_and_store(points: list[EmbeddedPoint], settings: VoiceEmbeddingPlotSettings, run_id: str) -> SaveResult:
    coordinates = _pca_coordinates([point.vector for point in points])
    plot_points = [
        PlotPoint(x=float(coordinates[index, 0]), y=float(coordinates[index, 1]), voice=point.label, name=point.name)
        for index, point in enumerate(points)
    ]
    html_bytes = render_voice_plot_html(plot_points, settings.title, settings.point_size)
    metadata = {
        "content_type": "text/html",
        "kind": "voice_embedding_pca",
        "run_id": run_id,
        "point_count": len(points),
        "voices": sorted({point.label for point in points}),
        "embedding_component": settings.embedding_component,
        "label_source": settings.label_source,
    }
    with database_session() as session:
        artifact = asset_crud.create_extra_file(
            session,
            ExtraFileCreate(name=settings.artifact_name, data=html_bytes, type_="artifact", metadata=metadata),
        )
        artifact_id = str(artifact.id)
        bucket_key = artifact.path
    return SaveResult(Path(bucket_key), "html", stable_id("artifact", artifact_id), artifact_id, {**metadata, "artifact_id": artifact_id, "bucket_key": bucket_key})
