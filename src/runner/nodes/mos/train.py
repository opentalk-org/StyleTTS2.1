from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from runner.nodes.models import TrainingManifest
from runner.nodes.mos.dataset import MosPairBatch, build_mos_dataloader
from runner.nodes.mos.loss import MosLoss, mos_pair_loss
from runner.nodes.mos.model import MosModelBundle, save_mos_bundle


@dataclass(frozen=True)
class MosTrainMetrics:
    train_loss: float
    validation_loss: float
    validation_mos_loss: float
    validation_comparison_loss: float

    def as_dict(self) -> dict[str, float]:
        return {
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "validation_mos_loss": self.validation_mos_loss,
            "validation_comparison_loss": self.validation_comparison_loss,
        }


async def train_mos_model(
    bundle: MosModelBundle,
    manifest: TrainingManifest,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    comparison_weight: float,
    dataloader_workers: int,
    save_interval_epochs: int,
    context,
    node_id: str,
) -> MosTrainMetrics:
    train_count = int(manifest.metadata["train_count"])
    validation_count = int(manifest.metadata["validation_count"])
    train_loader = build_mos_dataloader(
        Path(str(manifest.metadata["train_manifest_path"])),
        train_count,
        bundle.feature_extractor,
        batch_size,
        dataloader_workers,
    )
    validation_loader = build_mos_dataloader(
        Path(str(manifest.metadata["validation_manifest_path"])),
        validation_count,
        bundle.feature_extractor,
        batch_size,
        dataloader_workers,
    )
    optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    steps_per_epoch = math.ceil(train_count / batch_size)
    completed_steps = 0
    final_metrics: MosTrainMetrics | None = None

    for epoch in range(epochs):
        context.check_cancel()
        bundle.model.train()
        training_losses: list[float] = []
        for batch in train_loader:
            context.check_cancel()
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss = _batch_loss(bundle, batch, comparison_weight)
            scaler.scale(loss.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(bundle.model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            training_losses.append(float(loss.total.detach().item()))
            completed_steps += 1
            await context.report_progress(
                node_id,
                completed_steps,
                epochs * steps_per_epoch,
                f"MOS training epoch {epoch + 1}/{epochs}",
            )
        validation = _validate(bundle, validation_loader, device, comparison_weight, context)
        final_metrics = MosTrainMetrics(
            train_loss=float(np.mean(training_losses)),
            validation_loss=validation.total,
            validation_mos_loss=validation.mos,
            validation_comparison_loss=validation.comparison,
        )
        if (epoch + 1) % save_interval_epochs == 0:
            save_mos_bundle(output_dir, bundle)

    assert final_metrics is not None, "MOS training did not complete an epoch"
    save_mos_bundle(output_dir, bundle)
    return final_metrics


@dataclass(frozen=True)
class MosValidationMetrics:
    total: float
    mos: float
    comparison: float


def _validate(bundle, loader, device, comparison_weight, context) -> MosValidationMetrics:
    bundle.model.eval()
    total_losses: list[float] = []
    mos_losses: list[float] = []
    comparison_losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            context.check_cancel()
            loss = _batch_loss(bundle, batch.to(device), comparison_weight)
            total_losses.append(float(loss.total.item()))
            mos_losses.append(float(loss.mos.item()))
            comparison_losses.append(float(loss.comparison.item()))
    return MosValidationMetrics(
        total=float(np.mean(total_losses)),
        mos=float(np.mean(mos_losses)),
        comparison=float(np.mean(comparison_losses)),
    )


def _batch_loss(bundle: MosModelBundle, batch: MosPairBatch, comparison_weight: float) -> MosLoss:
    prediction_a = bundle.model(batch.inputs_a.input_values, batch.inputs_a.attention_mask)
    prediction_b = bundle.model(batch.inputs_b.input_values, batch.inputs_b.attention_mask)
    return mos_pair_loss(
        prediction_a,
        prediction_b,
        batch.score_a,
        batch.score_b,
        batch.preferred_sign,
        comparison_weight,
    )
