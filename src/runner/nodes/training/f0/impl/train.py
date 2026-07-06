from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from aim import Run
from tqdm import tqdm

from runner.nodes.training.f0.impl.dataset import build_f0_dataloaders
from runner.nodes.training.f0.impl.optimizer import build_f0_optimizer
from runner.nodes.training.styletts.finetune.training.modules.jdc import JDCNet
from runner.nodes.training.styletts.finetune.studio.finetune_aim_logger import FinetuneAimLogger

logger = logging.getLogger(__name__)


def _load_weights_into_jdc(model: JDCNet, pth_path: str, device: torch.device) -> None:
    blob: dict[str, Any] | Any = torch.load(pth_path, map_location=device, weights_only=False)
    if isinstance(blob, dict):
        state = blob.get("net", blob.get("model", blob))
    else:
        state = blob
    if not isinstance(state, dict):
        return
    model_state = model.state_dict()
    for key, val in state.items():
        if key not in model_state:
            continue
        tgt = model_state[key]
        src = val.data if isinstance(val, nn.Parameter) else val
        if src.shape != tgt.shape:
            try:
                ms = [min(int(a), int(b)) for a, b in zip(src.shape, tgt.shape, strict=True)]
                sl = tuple(slice(0, m) for m in ms)
                tgt[sl].copy_(src[sl])
            except Exception:
                logger.warning("skip weight %s (shape %s vs %s)", key, tuple(src.shape), tuple(tgt.shape))
        else:
            tgt.copy_(src)


def save_f0_checkpoint(
    path: Path,
    *,
    model: JDCNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "net": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def train_f0_model(
    *,
    aim_run: Run,
    train_list_path: str,
    val_list_path: str,
    run_dir: Path,
    weights_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lambda_f0: float,
    checkpoint_save_interval_epochs: int,
    pretrained_pth: str | None,
    weight_decay: float = 5e-4,
    pct_start: float = 0.0,
    num_workers: int = 2,
) -> None:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    torch.backends.cudnn.benchmark = True

    f0_cache_dir = run_dir / "f0_cache"
    f0_cache_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = weights_dir.resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_f0_dataloaders(
        train_list_path=train_list_path,
        val_list_path=val_list_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device_type=device_str,
        f0_cache_dir=f0_cache_dir,
    )

    if len(train_loader) < 1:
        raise ValueError("f0_train_no_batches")
    if len(val_loader) < 1:
        raise ValueError("f0_val_no_batches")

    steps_per_epoch = len(train_loader)
    aim_metrics = FinetuneAimLogger(
        aim_run,
        schedule_epochs_total=epochs,
        batches_per_epoch=steps_per_epoch,
        diff_epoch=0,
        joint_epoch=epochs,
    )

    logger.info("Warming F0 caches (train then val)…")
    for idx, _ in enumerate(train_loader):
        logger.info("Warming F0 caches (train) %d/%d", idx, len(train_loader))
    for idx, _ in enumerate(val_loader):
        logger.info("Warming F0 caches (val) %d/%d", idx, len(val_loader))

    model = JDCNet(num_class=1, seq_len=192)
    model.to(device)
    if pretrained_pth:
        _load_weights_into_jdc(model, pretrained_pth, device)
        logger.info("Loaded pretrained weights from %s", pretrained_pth)

    optimizer, scheduler = build_f0_optimizer(
        list(model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
    )

    criterion_l1 = nn.SmoothL1Loss()
    criterion_sil = nn.BCEWithLogitsLoss()

    global_step = 0
    last_epoch_completed = 0
    final_path = weights_dir / "final.pth"
    final_saved = False
    try:
        epoch_pbar = tqdm(range(epochs), desc="F0 train", unit="epoch")
        for epoch in epoch_pbar:
            model.train()
            train_losses: dict[str, list[float]] = {"loss": [], "f0_loss": [], "sil_loss": []}
            batch_pbar = tqdm(
                train_loader,
                desc=f"train {epoch + 1}/{epochs}",
                leave=False,
                total=steps_per_epoch,
            )
            for batch_idx, batch in enumerate(batch_pbar, start=1):
                x, f0, sil = [b.to(device, non_blocking=True) for b in batch]
                x_t = x.transpose(-1, -2)
                optimizer.zero_grad(set_to_none=True)
                f0_pred, sil_pred = model.forward_pitch_train(x_t)
                loss_f0 = lambda_f0 * criterion_l1(f0_pred, f0)
                loss_sil = criterion_sil(sil_pred, sil)
                loss = loss_f0 + loss_sil
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_losses["loss"].append(float(loss.item()))
                train_losses["f0_loss"].append(float(loss_f0.item()))
                train_losses["sil_loss"].append(float(loss_sil.item()))
                global_step += 1
                batch_pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

                lr = optimizer.param_groups[0]["lr"]
                aim_metrics.log_train(
                    epoch + 1,
                    global_step,
                    {
                        "loss": loss.item(),
                        "f0_loss": loss_f0.item(),
                        "sil_loss": loss_sil.item(),
                        "lr": lr,
                        "lambda_f0": lambda_f0,
                    },
                    batches_per_epoch=steps_per_epoch,
                    batch_in_epoch=batch_idx,
                    schedule_epochs_total=epochs,
                    schedule_diff_epoch=0,
                    schedule_joint_epoch=epochs,
                )

            model.eval()
            eval_losses: dict[str, list[float]] = {"loss": [], "f0_loss": [], "sil_loss": []}
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"val {epoch + 1}/{epochs}", leave=False, total=len(val_loader)):
                    x, f0, sil = [b.to(device, non_blocking=True) for b in batch]
                    x_t = x.transpose(-1, -2)
                    f0_pred, sil_pred = model.forward_pitch_train(x_t)
                    loss_f0 = lambda_f0 * criterion_l1(f0_pred, f0)
                    loss_sil = criterion_sil(sil_pred, sil)
                    loss = loss_f0 + loss_sil
                    eval_losses["loss"].append(float(loss.item()))
                    eval_losses["f0_loss"].append(float(loss_f0.item()))
                    eval_losses["sil_loss"].append(float(loss_sil.item()))

            val_metrics = {f"val_{k}": float(np.mean(v)) for k, v in eval_losses.items()}
            val_metrics["val_lr"] = optimizer.param_groups[0]["lr"]
            aim_metrics.log_val(epoch + 1, global_step, val_metrics)

            tr_l = float(np.mean(train_losses["loss"]))
            logger.info(
                "epoch %d/%d train loss=%.4f f0=%.4f sil=%.4f | val loss=%.4f",
                epoch + 1,
                epochs,
                tr_l,
                float(np.mean(train_losses["f0_loss"])),
                float(np.mean(train_losses["sil_loss"])),
                val_metrics["val_loss"],
            )
            epoch_pbar.set_postfix(train=f"{tr_l:.4f}", val=f"{val_metrics['val_loss']:.4f}")

            if (epoch + 1) % max(1, checkpoint_save_interval_epochs) == 0:
                ckpt_path = weights_dir / f"epoch_{epoch + 1:05d}.pth"
                save_f0_checkpoint(
                    ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    global_step=global_step,
                )
                logger.info("Saved %s", ckpt_path)

            last_epoch_completed = epoch + 1

        save_f0_checkpoint(
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
            save_f0_checkpoint(
                final_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=last_epoch_completed,
                global_step=global_step,
            )
            logger.warning("Saved interrupted checkpoint to %s", final_path)
