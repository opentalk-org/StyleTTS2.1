from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from runner.nodes.training.common.mlflow_run import TrackerRun
from runner.nodes.training.styletts.finetune.studio.metrics import _metrics_for_json


class FinetuneMlflowLogger:
    def __init__(
        self,
        run: TrackerRun,
        *,
        schedule_epochs_total: int | None = None,
        batches_per_epoch: int | None = None,
        diff_epoch: int | None = None,
        joint_epoch: int | None = None,
    ) -> None:
        self._run = run
        self._schedule_epochs_total = schedule_epochs_total
        self._batches_per_epoch = batches_per_epoch
        self._diff_epoch = diff_epoch
        self._joint_epoch = joint_epoch
        self._last_train_ts: float | None = None
        self._last_train_step: int | None = None

    def _track_metric(self, name: str, step: int, v: Any) -> None:
        if isinstance(v, bool):
            self._run.track(float(v), name=name, step=step)
            return
        if isinstance(v, (int, float)):
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                return
            self._run.track(fv, name=name, step=step)
            return
        if isinstance(v, str):
            self._run.track(v, name=name, step=step)
            return
        self._run.track(str(v), name=name, step=step)

    def _emit_progress(
        self,
        *,
        epoch_idx0: int,
        batch_in_epoch: int,
        global_step: int,
    ) -> None:
        if (
            self._schedule_epochs_total is None
            or self._batches_per_epoch is None
            or self._diff_epoch is None
            or self._joint_epoch is None
        ):
            return
        now = time.time()
        sps: float | None = None
        if self._last_train_ts is not None and self._last_train_step is not None and global_step > self._last_train_step:
            dt = now - self._last_train_ts
            ds = global_step - self._last_train_step
            if dt > 1e-3 and ds > 0:
                sps = ds / dt
        self._last_train_ts = now
        self._last_train_step = global_step

        total_e = max(1, int(self._schedule_epochs_total))
        d_e = max(0, int(self._diff_epoch))
        j_e = max(0, int(self._joint_epoch))
        spe = max(1, int(self._batches_per_epoch))
        rem_full = max(0, total_e - epoch_idx0 - 1)
        rem_cur = max(0, spe - batch_in_epoch)
        eta_total: float | None = None
        if sps is not None and sps > 0:
            eta_total = (rem_full * spe + rem_cur) / sps

        total_steps = max(1, total_e * spe)
        done = epoch_idx0 * spe + batch_in_epoch
        job_pct = max(1.0, min(99.0, 1.0 + 98.0 * done / total_steps))
        self._run.track(job_pct, name="job_progress", step=global_step)
        if sps is not None and not math.isnan(sps) and not math.isinf(sps):
            self._run.track(float(sps), name="train/steps_per_sec", step=global_step)
        if eta_total is not None and not math.isnan(eta_total) and not math.isinf(eta_total):
            self._run.track(float(eta_total), name="train/eta_seconds_to_training_end", step=global_step)

    def log_train(
        self,
        epoch: int,
        step: int,
        metrics: dict[str, Any],
        *,
        batches_per_epoch: int | None = None,
        batch_in_epoch: int | None = None,
        schedule_epochs_total: int | None = None,
        schedule_diff_epoch: int | None = None,
        schedule_joint_epoch: int | None = None,
        batch_utterances: int | None = None,
        audio_samples_in_batch: int | None = None,
    ) -> None:
        del schedule_diff_epoch, schedule_joint_epoch, batch_utterances, audio_samples_in_batch
        cleaned = _metrics_for_json(metrics)
        for k, v in cleaned.items():
            if v is None:
                continue
            self._track_metric(f"train/{k}", step, v)
        bpe = batches_per_epoch if batches_per_epoch is not None else self._batches_per_epoch
        ste = schedule_epochs_total if schedule_epochs_total is not None else self._schedule_epochs_total
        if batch_in_epoch is not None and bpe is not None and ste is not None:
            self._emit_progress(
                epoch_idx0=int(epoch) - 1,
                batch_in_epoch=int(batch_in_epoch),
                global_step=int(step),
            )

    def log_val(self, epoch: int, step: int, metrics: dict[str, Any]) -> None:
        cleaned = _metrics_for_json(metrics)
        for k, v in cleaned.items():
            if v is None:
                continue
            self._track_metric(f"val/{k}", step, v)

    def log_val_samples(
        self,
        epoch: int,
        step: int,
        samples: list[dict[str, str]],
        *,
        log_dir: str | Path | None = None,
    ) -> None:
        n = len(samples)
        self._run.track(float(n), name="val/sample_rows", step=step)
        paths = [sample["path"] for sample in samples]
        if paths:
            joined = "\n".join(paths[:32])
            self._run.track(joined, name="val/sample_paths", step=step)
        if log_dir is None:
            return
        base = Path(log_dir)
        for sample in samples:
            role = sample["role"]
            relp = sample["path"]
            idx = sample["index"]
            if role not in ("gt", "pred") or not relp or not str(relp).lower().endswith(".wav"):
                continue
            wav_path = base / relp
            if not wav_path.is_file():
                continue
            self._run.log_artifact(
                wav_path,
                artifact_path=f"validation/epoch_{epoch:05d}/step_{step:09d}/sample_{idx}/{role}",
            )
