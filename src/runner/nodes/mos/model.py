from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, Wav2Vec2Model

from runner.nodes.mos.audio import MOS_SAMPLE_RATE, MosFeatureExtractor


class MosCheckpointConfig(BaseModel):
    base_model_id: str = "facebook/wav2vec2-xls-r-300m"
    sample_rate: int = Field(default=MOS_SAMPLE_RATE, gt=0)


class MosRegressor(nn.Module):
    def __init__(self, encoder: Wav2Vec2Model):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size, 1)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.encoder(input_values=input_values, attention_mask=attention_mask).last_hidden_state
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            feature_mask = self.encoder._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
            weights = feature_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


@dataclass(frozen=True)
class MosModelBundle:
    feature_extractor: MosFeatureExtractor
    model: MosRegressor
    config: MosCheckpointConfig


def load_base_mos_bundle(checkpoint_dir: Path, device: torch.device) -> MosModelBundle:
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(checkpoint_dir))
    encoder = Wav2Vec2Model.from_pretrained(str(checkpoint_dir))
    model = MosRegressor(encoder).to(device)
    return MosModelBundle(
        feature_extractor=feature_extractor,
        model=model,
        config=MosCheckpointConfig(),
    )


def save_mos_bundle(folder: Path, bundle: MosModelBundle) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    processor_dir = folder / "processor"
    encoder_dir = folder / "encoder"
    bundle.feature_extractor.save_pretrained(str(processor_dir))
    bundle.model.encoder.save_pretrained(str(encoder_dir))
    torch.save(bundle.model.head.state_dict(), folder / "mos_head.pt")
    (folder / "mos_config.json").write_text(bundle.config.model_dump_json(indent=2), encoding="utf-8")


def load_trained_mos_bundle(checkpoint_dir: Path, device: torch.device) -> MosModelBundle:
    config = MosCheckpointConfig.model_validate_json(
        (checkpoint_dir / "mos_config.json").read_text(encoding="utf-8")
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(checkpoint_dir / "processor"))
    encoder = Wav2Vec2Model.from_pretrained(str(checkpoint_dir / "encoder"))
    model = MosRegressor(encoder)
    head_state = torch.load(checkpoint_dir / "mos_head.pt", map_location=device, weights_only=True)
    model.head.load_state_dict(head_state)
    model.to(device)
    return MosModelBundle(feature_extractor=feature_extractor, model=model, config=config)
