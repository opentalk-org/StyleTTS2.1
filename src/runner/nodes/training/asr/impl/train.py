from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from runflow.runtime.cancellation import check_cancel
from runner.nodes.training.asr.impl.config_load import blank_index_from_config
from runner.nodes.training.asr.impl.dataset import build_asr_dataloaders
from runner.nodes.training.asr.impl.optimizer import build_asr_optimizer
from runner.nodes.training.common.mlflow_run import TrackerRun
from runner.nodes.training.styletts.finetune.training.asr_train_models import init_ASR_model_from_config, load_ASR_models
from runner.nodes.text.runtime.symbols import build_word_index_dictionary
from runner.nodes.training.styletts.finetune.studio.finetune_mlflow_logger import FinetuneMlflowLogger

logger = logging.getLogger(__name__)


def save_asr_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def train_asr_model(
    *,
    run: TrackerRun,
    dataset_id: UUID,
    validation_samples: int,
    run_dir: Path,
    weights_dir: Path,
    effective_config: dict[str, Any],
    effective_config_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint_save_interval_epochs: int,
    pretrained_weights_path: str | None,
    num_workers: int = 2,
) -> None:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    torch.backends.cudnn.benchmark = True

    symbols = list(effective_config.get("data_params", {}).get("phoneme_symbols") or [])
    sym_dict = build_word_index_dictionary(symbols)
    blank_idx = blank_index_from_config(effective_config, symbol_to_idx=sym_dict)
    ctc_loss_fn = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    ce_loss_fn = nn.CrossEntropyLoss()

    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = weights_dir.resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = weights_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_asr_dataloaders(
        dataset_id=dataset_id,
        validation_samples=validation_samples,
        effective_config=effective_config,
        batch_size=batch_size,
        num_workers=num_workers,
        device_type=device_str,
    )
    if len(train_loader) < 1:
        raise ValueError("asr_train_no_batches")
    if len(val_loader) < 1:
        raise ValueError("asr_val_no_batches")

    steps_per_epoch = len(train_loader)
    mlflow_metrics = FinetuneMlflowLogger(
        run,
        schedule_epochs_total=epochs,
        batches_per_epoch=steps_per_epoch,
        diff_epoch=0,
        joint_epoch=epochs,
    )

    cfg_path_s = str(effective_config_path.resolve())
    n_sym = len(symbols)
    if pretrained_weights_path:
        model = load_ASR_models(
            pretrained_weights_path,
            cfg_path_s,
            target_n_token=n_sym,
        )
        logger.info("Loaded ASR pretrained weights from %s", pretrained_weights_path)
    else:
        model = init_ASR_model_from_config(cfg_path_s, target_n_token=n_sym)
    model.to(device)

    opt_params = effective_config.get("optimizer_params") or {}
    sch_params = effective_config.get("scheduler_params") or {}
    optimizer, scheduler = build_asr_optimizer(
        list(model.parameters()),
        learning_rate=learning_rate,
        optimizer_params=opt_params,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        scheduler_params=sch_params,
    )

    grad_clip = float((effective_config.get("training") or {}).get("grad_clip") or 5.0)

    global_step = 0
    last_epoch_completed = 0
    final_path = weights_dir / "final.pth"
    final_saved = False

    n_down = int(getattr(model, "n_down", 1))

    try:
        epoch_pbar = tqdm(range(epochs), desc="ASR train", unit="epoch")
        for epoch in epoch_pbar:
            check_cancel()
            model.train()
            train_losses: dict[str, list[float]] = {"loss": [], "ctc": [], "s2s": []}
            batch_pbar = tqdm(
                train_loader,
                desc=f"train {epoch + 1}/{epochs}",
                leave=False,
                total=steps_per_epoch,
            )
            for batch_idx, batch in enumerate(batch_pbar, start=1):
                check_cancel()
                batch_d = [b.to(device, non_blocking=True) for b in batch]
                text_input, text_input_length, mel_input, mel_input_length = batch_d
                mel_enc_len = mel_input_length // (2**n_down)
                mel_mask = model.length_to_mask(mel_enc_len)
                optimizer.zero_grad(set_to_none=True)
                ppgs, s2s_pred, _s2s_attn = model(
                    mel_input, src_key_padding_mask=mel_mask, text_input=text_input
                )
                loss_ctc = ctc_loss_fn(
                    ppgs.log_softmax(dim=2).transpose(0, 1),
                    text_input,
                    mel_enc_len,
                    text_input_length,
                )
                loss_s2s = torch.tensor(0.0, device=device)
                for _s2s_pred, _text_input, _text_length in zip(
                    s2s_pred, text_input, text_input_length, strict=True
                ):
                    loss_s2s = loss_s2s + ce_loss_fn(
                        _s2s_pred[: int(_text_length)], _text_input[: int(_text_length)]
                    )
                loss_s2s = loss_s2s / float(text_input.size(0))
                loss = loss_ctc + loss_s2s
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

                train_losses["loss"].append(float(loss.item()))
                train_losses["ctc"].append(float(loss_ctc.item()))
                train_losses["s2s"].append(float(loss_s2s.item()))
                global_step += 1
                lr = optimizer.param_groups[0]["lr"]
                batch_pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

                mlflow_metrics.log_train(
                    epoch + 1,
                    global_step,
                    {
                        "loss": loss.item(),
                        "ctc_loss": loss_ctc.item(),
                        "s2s_loss": loss_s2s.item(),
                        "lr": lr,
                    },
                    batches_per_epoch=steps_per_epoch,
                    batch_in_epoch=batch_idx,
                    schedule_epochs_total=epochs,
                    schedule_diff_epoch=0,
                    schedule_joint_epoch=epochs,
                )

            model.eval()
            eval_losses: dict[str, list[float]] = {"loss": [], "ctc": [], "s2s": []}
            with torch.no_grad():
                for vb in tqdm(val_loader, desc=f"val {epoch + 1}/{epochs}", leave=False, total=len(val_loader)):
                    check_cancel()
                    vb = [b.to(device, non_blocking=True) for b in vb]
                    text_input, text_input_length, mel_input, mel_input_length = vb
                    mel_enc_len = mel_input_length // (2**n_down)
                    mel_mask = model.length_to_mask(mel_enc_len)
                    ppgs, s2s_pred, _s2s_attn = model(
                        mel_input, src_key_padding_mask=mel_mask, text_input=text_input
                    )
                    loss_ctc = ctc_loss_fn(
                        ppgs.log_softmax(dim=2).transpose(0, 1),
                        text_input,
                        mel_enc_len,
                        text_input_length,
                    )
                    loss_s2s = torch.tensor(0.0, device=device)
                    for _s2s_pred, _text_input, _text_length in zip(
                        s2s_pred, text_input, text_input_length, strict=True
                    ):
                        loss_s2s = loss_s2s + ce_loss_fn(
                            _s2s_pred[: int(_text_length)], _text_input[: int(_text_length)]
                        )
                    loss_s2s = loss_s2s / float(text_input.size(0))
                    loss = loss_ctc + loss_s2s
                    eval_losses["loss"].append(float(loss.item()))
                    eval_losses["ctc"].append(float(loss_ctc.item()))
                    eval_losses["s2s"].append(float(loss_s2s.item()))

            val_metrics = {f"val_{k}": float(np.mean(v)) for k, v in eval_losses.items()}
            val_metrics["val_lr"] = optimizer.param_groups[0]["lr"]
            mlflow_metrics.log_val(epoch + 1, global_step, val_metrics)

            tr_l = float(np.mean(train_losses["loss"]))
            logger.info(
                "epoch %d/%d train loss=%.4f ctc=%.4f s2s=%.4f | val loss=%.4f",
                epoch + 1,
                epochs,
                tr_l,
                float(np.mean(train_losses["ctc"])),
                float(np.mean(train_losses["s2s"])),
                val_metrics["val_loss"],
            )
            epoch_pbar.set_postfix(train=f"{tr_l:.4f}", val=f"{val_metrics['val_loss']:.4f}")

            if (epoch + 1) % max(1, checkpoint_save_interval_epochs) == 0:
                ckpt_path = snapshots_dir / f"epoch_{epoch + 1:05d}.pth"
                save_asr_checkpoint(
                    ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    global_step=global_step,
                )
                logger.info("Saved %s", ckpt_path)

            last_epoch_completed = epoch + 1

        save_asr_checkpoint(
            final_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epochs,
            global_step=global_step,
        )
        final_saved = True
        logger.info("Training finished; wrote %s", final_path)
    finally:
        if not final_saved and global_step > 0:
            save_asr_checkpoint(
                final_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=last_epoch_completed,
                global_step=global_step,
            )
            logger.warning("Saved interrupted checkpoint to %s", final_path)
