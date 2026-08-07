from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import torch
from matplotlib.figure import Figure
from torch import nn

from runflow.runtime.cancellation import check_cancel
from runner.nodes.training.common.mlflow_run import TrackerRun
from runner.nodes.training.styletts.finetune.training.data import build_dataloader
from runner.nodes.training.styletts.finetune.training.modules.asr.models import ASRCNN
from runner.nodes.training.styletts.finetune.training.utils import (
    mask_from_lens,
    maximum_path,
)

logger = logging.getLogger(__name__)


def save_asr_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": step,
        },
        path,
    )


def _load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("asr_checkpoint_invalid")
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as error:
        raise ValueError("asr_checkpoint_is_not_compatible") from error
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("global_step", 0))


def _batch_loss(
    model: ASRCNN,
    batch,
    device: torch.device,
    ctc_loss: nn.CTCLoss,
    sequence_loss: nn.CrossEntropyLoss,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mel_lengths = batch.mel_lengths.to(device)
    text_lengths = batch.input_lengths.to(device)
    encoded_lengths = mel_lengths // (2**model.n_down)
    logits, sequence_logits, attention = model(
        batch.mels,
        src_key_padding_mask=model.length_to_mask(encoded_lengths),
        text_input=batch.texts,
    )
    ctc = ctc_loss(
        logits.log_softmax(dim=2).transpose(0, 1),
        batch.texts,
        encoded_lengths,
        text_lengths,
    )
    sequence = logits.new_zeros(())
    for prediction, target, length in zip(
        sequence_logits,
        batch.texts,
        text_lengths,
        strict=True,
    ):
        sequence = sequence + sequence_loss(
            prediction[:length],
            target[:length],
        )
    sequence = sequence / batch.texts.size(0)
    return ctc + sequence, ctc, sequence, attention


def _save_alignment(path: Path, values: torch.Tensor, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(10, 4), layout="constrained")
    axis = figure.subplots()
    axis.imshow(
        values.detach().float().cpu().numpy(),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_xlabel("mel frame")
    axis.set_ylabel("phoneme")
    axis.set_title(title)
    figure.savefig(path, dpi=120)


def _log_alignment_artifacts(
    run: TrackerRun,
    output_dir: Path,
    step: int,
    soft_alignment: torch.Tensor,
    hard_alignment: torch.Tensor,
    batch,
    n_down: int,
) -> None:
    for index in range(min(4, soft_alignment.size(0))):
        phone_steps = int(batch.input_lengths[index])
        mel_steps = int(batch.mel_lengths[index]) // (2**n_down)
        directory = output_dir / f"step_{step:09d}" / f"sample_{index}"
        for filename, values, title in (
            ("soft_attention.png", soft_alignment[index, :phone_steps, :mel_steps], "Soft attention"),
            ("hard_attention.png", hard_alignment[index, :phone_steps, :mel_steps], "Hard attention"),
        ):
            path = directory / filename
            _save_alignment(path, values, title)
            run.log_artifact(
                path,
                f"validation/step_{step:09d}/alignment/sample_{index}",
            )


@torch.no_grad()
def _validate(
    model: ASRCNN,
    batches,
    device: torch.device,
    ctc_loss: nn.CTCLoss,
    sequence_loss: nn.CrossEntropyLoss,
    run: TrackerRun,
    step: int,
    artifact_dir: Path,
) -> dict[str, float]:
    model.eval()
    losses = []
    ctc_losses = []
    sequence_losses = []
    for batch_index, batch in enumerate(batches):
        check_cancel()
        batch = batch.to(device)
        loss, ctc, sequence, alignment = _batch_loss(
            model,
            batch,
            device,
            ctc_loss,
            sequence_loss,
        )
        losses.append(float(loss.item()))
        ctc_losses.append(float(ctc.item()))
        sequence_losses.append(float(sequence.item()))
        if batch_index == 0:
            soft_alignment = alignment[:, 1:]
            alignment_mask = mask_from_lens(
                soft_alignment,
                batch.input_lengths,
                batch.mel_lengths // (2**model.n_down),
            )
            soft_alignment = soft_alignment.masked_fill(~alignment_mask.bool(), 0.0)
            hard_alignment = maximum_path(soft_alignment, alignment_mask)
            _log_alignment_artifacts(
                run,
                artifact_dir,
                step,
                soft_alignment,
                hard_alignment,
                batch,
                model.n_down,
            )
    if not losses:
        raise ValueError("asr_val_no_batches")
    return {
        "val/loss": float(np.mean(losses)),
        "val/ctc_loss": float(np.mean(ctc_losses)),
        "val/sequence_loss": float(np.mean(sequence_losses)),
    }


def train_asr_model(
    *,
    run: TrackerRun,
    dataset_id: UUID,
    validation_samples: int,
    weights_dir: Path,
    effective_config: dict[str, Any],
    total_steps: int,
    validation_every_steps: int,
    checkpoint_every_steps: int,
    max_audio_seconds: float,
    max_text_tokens: int,
    learning_rate: float,
    warmup_steps: int,
    weight_decay: float,
    gradient_clip_norm: float,
    pretrained_weights_path: str | None,
    num_workers: int = 0,
    seed: int = 1,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    symbols = list(effective_config["data_params"]["phoneme_symbols"])
    model_params = effective_config["model_params"]
    model = ASRCNN(
        input_dim=int(model_params["input_dim"]),
        n_token=len(symbols),
        hidden_dim=int(model_params["hidden_dim"]),
        n_layers=int(model_params["n_layers"]),
        token_embedding_dim=int(model_params["token_embedding_dim"]),
    ).to(device)
    blank_symbol = str(effective_config["data_params"]["ctc_blank_character"])
    try:
        blank_index = symbols.index(blank_symbol)
    except ValueError as error:
        raise ValueError("asr_ctc_blank_not_in_vocab") from error
    ctc_loss = nn.CTCLoss(blank=blank_index, zero_infinity=True)
    sequence_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.0,
        final_div_factor=5.0,
    )
    step = (
        _load_checkpoint(pretrained_weights_path, model, optimizer, scheduler)
        if pretrained_weights_path
        else 0
    )
    if step >= total_steps:
        raise ValueError("asr_checkpoint_already_reached_total_steps")

    dataset_config = {
        "symbols": symbols,
        "max_text_tokens": max_text_tokens,
    }
    train_batches = build_dataloader(
        dataset_id,
        validation_samples,
        max_seconds=max_audio_seconds,
        num_workers=num_workers,
        device=device.type,
        dataset_config=dataset_config,
        seed=seed,
    )
    validation_batches = build_dataloader(
        dataset_id,
        validation_samples,
        validation=True,
        max_seconds=max_audio_seconds,
        num_workers=num_workers,
        device=device.type,
        dataset_config=dataset_config,
        seed=seed,
    )

    weights_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = weights_dir / "snapshots"
    final_path = weights_dir / "final.pth"
    final_saved = False
    try:
        model.train()
        for batch in train_batches:
            check_cancel()
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, ctc, sequence, _ = _batch_loss(
                model,
                batch,
                device,
                ctc_loss,
                sequence_loss,
            )
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            metrics = {
                "train/loss": float(loss.item()),
                "train/ctc_loss": float(ctc.item()),
                "train/sequence_loss": float(sequence.item()),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "performance/items_per_step": float(len(batch.audio_durations)),
                "performance/audio_seconds_per_step": float(sum(batch.audio_durations)),
                "job_progress": 100.0 * step / total_steps,
            }
            run.track_metrics(metrics, step=step)

            if step % validation_every_steps == 0 or step == total_steps:
                run.track_metrics(
                    _validate(
                        model,
                        validation_batches,
                        device,
                        ctc_loss,
                        sequence_loss,
                        run,
                        step,
                        weights_dir / "validation",
                    ),
                    step=step,
                )
                model.train()
            if step % checkpoint_every_steps == 0 or step == total_steps:
                save_asr_checkpoint(
                    snapshots_dir / f"step_{step:09d}.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                )
            if step % 10 == 0:
                logger.info(
                    "ASR step=%d/%d loss=%.5f ctc=%.5f sequence=%.5f lr=%.3e",
                    step,
                    total_steps,
                    loss.item(),
                    ctc.item(),
                    sequence.item(),
                    optimizer.param_groups[0]["lr"],
                )
            if step >= total_steps:
                break

        save_asr_checkpoint(
            final_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
        )
        final_saved = True
    finally:
        if not final_saved and step > 0:
            save_asr_checkpoint(
                final_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
            )
