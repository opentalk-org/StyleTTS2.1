from __future__ import annotations

import asyncio
import importlib
import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import yaml
from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import AUDIO, SAVE_RESULT
from runner.nodes.models import Audio, SaveResult, stable_id
from runner.nodes.statistics.voice_embedding_html import PlotPoint, render_voice_plot_html
from runner.nodes.synthesis.styletts_runtime.checkpoints import latest_weight, resolve_slot_checkpoint
from runner.nodes.synthesis.styletts_runtime.helpers import compute_style_from_wave, training_import_context
from shared.db import database_session
from shared.db.assets import crud as asset_crud
from shared.db.assets.schemas import ExtraFileCreate
from shared.db.audio import crud as audio_crud
from shared.db.voices import crud as voice_crud


UNASSIGNED_VOICE = "unassigned"


class VoiceEmbeddingPlotSettings(StrictSettings):
    checkpoint_id: UUID = Field(title="StyleTTS2 checkpoint")
    label_source: Literal["voice", "speaker"] = Field(default="voice", title="Colour by")
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


class EmbedVoicesPcaPlotNode(Node):
    NODE_TYPE = "EmbedVoicesPcaPlot"
    CATEGORY = "Statistics"
    SETTINGS = VoiceEmbeddingPlotSettings
    INPUTS = {"audio": Port("audio", AUDIO)}
    OUTPUTS = {"artifact": Port("artifact", SAVE_RESULT)}
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
        for inputs in batch:
            context.check_cancel()
            result = await asyncio.to_thread(self._ingest, inputs, str(context.run_id), context.output_dir)
            if result is not None:
                outputs.append(result)
        return outputs

    def _ingest(self, inputs: dict[str, Any], run_id: str, output_dir: Path) -> dict[str, SaveResult] | None:
        audio = inputs["audio"]
        assert isinstance(audio, Audio), f"unsupported audio input: {type(audio).__name__}"
        if self._runtime is None:
            self._runtime = _load_style_encoder_runtime(self.settings.checkpoint_id)
        batch_id = str(audio.metadata["source_batch_id"])
        expected = int(audio.metadata["source_batch_count"])
        point = self._embed_audio(audio)
        pending = self._pending.setdefault(batch_id, [])
        pending.append(point)
        if len(pending) < expected:
            return None
        del self._pending[batch_id]
        return {"artifact": _render_and_store(pending, self.settings, run_id, output_dir)}

    def _embed_audio(self, audio: Audio) -> EmbeddedPoint:
        with database_session() as session:
            data = audio.data if audio.data is not None else audio_crud.read_audio_file(session, audio.audio_file_id)
            record = audio_crud.get_audio_file(session, audio.audio_file_id)
            voice_names = {str(voice.id): voice.name for voice in voice_crud.list_voices(session)}
        label = _voice_label(record.segments, voice_names, self.settings.label_source)
        vector = _style_vector(self._runtime, data, self.settings.embedding_component)
        return EmbeddedPoint(vector=vector, label=label, name=audio.name)


def _load_style_encoder_runtime(checkpoint_id: UUID) -> StyleEncoderRuntime:
    main = resolve_slot_checkpoint(checkpoint_id, "styletts2")
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


def _voice_label(segments: list[dict[str, Any]], voice_names: dict[str, str], label_source: str) -> str:
    weight: Counter[str] = Counter()
    for segment in segments:
        duration = max(0.0, float(segment["end"]) - float(segment["start"]))
        if label_source == "speaker":
            key = str(segment["speaker"]) if segment["speaker"] else UNASSIGNED_VOICE
        else:
            voice_id = segment["voice_id"] if "voice_id" in segment else None
            key = voice_names.get(str(voice_id), UNASSIGNED_VOICE) if voice_id else UNASSIGNED_VOICE
        weight[key] += duration if duration > 0 else 1.0
    if not weight:
        return UNASSIGNED_VOICE
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


def _render_and_store(points: list[EmbeddedPoint], settings: VoiceEmbeddingPlotSettings, run_id: str, output_dir: Path) -> SaveResult:
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
    out_dir = output_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{artifact_id}.html"
    out_path.write_bytes(html_bytes)
    return SaveResult(out_path, "html", stable_id("artifact", artifact_id), artifact_id, {**metadata, "artifact_id": artifact_id})
