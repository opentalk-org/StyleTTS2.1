import logging
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import TrainingConfig
from .setup import TrainingRuntime

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


class CheckpointPublisher:
    def __init__(
        self,
        config: TrainingConfig,
        runtime: TrainingRuntime,
        data_client=None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.data_client = data_client

    def publish(
        self,
        step: int,
        validation_loss,
    ) -> None:
        payload = self.config.model_dump(mode="json")
        state = {
            "net": {
                name: self.runtime.accelerator.get_state_dict(module)
                for name, module in self.runtime.models.modules.items()
            },
            "optimizer": self.runtime.optimizer.state_dict(),
            "iters": step,
            "val_loss": validation_loss,
            "step": step,
        }

        dest = Path(self.config.log_dir) / "published_checkpoints" / f"step_{step:09d}"
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)

        target = dest / f"model_{step}.pth"
        torch.save(_checkpoint_tree_to_cpu(state), target)
        # config.yml alongside the weights makes the folder a self-contained
        # checkpoint the loader (layout.architecture_yaml) can consume directly
        with open(dest / "config.yml", "w") as outfile:
            yaml.safe_dump(payload, outfile, sort_keys=False, allow_unicode=True)
        logger.info(
            "checkpoint published step=%s bytes=%s path=%s",
            step,
            target.stat().st_size,
            dest,
        )
        if self.data_client is not None:
            self.data_client.upload_checkpoint(step, dest)
            logger.info("checkpoint uploaded to givemedata step=%s", step)
