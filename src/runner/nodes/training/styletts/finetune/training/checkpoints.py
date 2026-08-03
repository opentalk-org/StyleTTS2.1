from ..checkpoint_publish import (
    publish_finetune_step_bundle_from_training_config,
)
from .config import TrainingConfig
from .setup import TrainingRuntime


class CheckpointPublisher:
    def __init__(
        self,
        config: TrainingConfig,
        runtime: TrainingRuntime,
    ) -> None:
        self.config = config
        self.runtime = runtime

    def publish(
        self,
        step: int,
        validation_loss,
        running_std: list[float],
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
        publish_finetune_step_bundle_from_training_config(
            payload,
            training_state=state,
            step=step,
            segment_slug=f"step_{step:09d}",
        )
