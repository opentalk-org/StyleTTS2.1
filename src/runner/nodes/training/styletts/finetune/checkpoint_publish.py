from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

from runner.nodes.training.styletts.finetune.names import slugify_segment
from shared.db.assets import crud as asset_crud
from shared.db.assets.schemas import CheckpointCreate
from shared.db.connection import database_session

logger = logging.getLogger(__name__)


def _checkpoint_tree_to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _checkpoint_tree_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_checkpoint_tree_to_cpu(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_checkpoint_tree_to_cpu(x) for x in obj)
    return obj


def publish_finetune_epoch_bundle_from_training_config(
    config: dict[str, Any],
    *,
    training_state: dict[str, Any],
    epoch_1based: int,
    segment_slug: str,
) -> None:
    _ = segment_slug
    sp = config["studio_publish"]
    job_id = str(sp["finetune_job_id"])
    parent = str(sp["parent_checkpoint_path"])
    run_name = sp["run_name"]
    log_dir = Path(str(config["log_dir"]))

    mp = config["model_params"]
    dec = mp["decoder"]
    meta = {
        "multispeaker": bool(mp["multispeaker"]),
        "decoder_type": str(dec["type"]).strip().lower(),
        "symbols": _symbol_list(config["symbols"]),
    }

    run_seg = slugify_segment((run_name or "").strip(), max_len=64)
    suffix = f"epoch {epoch_1based}"
    title = f"StyleTTS2 finetune {run_seg} {suffix}"
    run_name_opt = str(run_name).strip() if isinstance(run_name, str) else None
    dest = log_dir / "published_checkpoints" / f"epoch_{epoch_1based:04d}"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    target = dest / f"model_{epoch_1based}.pth"
    logger.info("finetune_publish begin job_id=%s epoch=%s bundle=%s", job_id, epoch_1based, dest)
    torch.save(_checkpoint_tree_to_cpu(training_state), target)
    logger.info("finetune_publish saved weights job_id=%s epoch=%s bytes=%s", job_id, epoch_1based, target.stat().st_size)

    with open(dest / "config.yml", "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=True)

    with database_session() as session:
        checkpoint = asset_crud.create_checkpoint(
            session,
            CheckpointCreate(
                name=title,
                folder_path=dest,
                type_="styletts2",
                metadata={
                    **meta,
                    "source": "finetune_epoch",
                    "finetune_job_id": job_id,
                    "finetune_run_name": run_name_opt if run_name_opt else None,
                    "parent_checkpoint_path": parent,
                    "studio_stage": "epoch",
                    "state": {
                        "source": "finetune_epoch",
                        "finetune_job_id": job_id,
                        "finetune_run_name": run_name_opt if run_name_opt else None,
                        "parent_checkpoint_path": parent,
                        "studio_stage": "epoch",
                    },
                },
                job_id=job_id,
            ),
        )
    logger.info("finetune_publish registered row job_id=%s path=%s", job_id, checkpoint.path)


def _symbol_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part for part in value.split(" ") if part]
    if isinstance(value, list):
        return [str(part) for part in value if part is not None and str(part)]
    return []
