import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from torch import nn

from .config import BeetleConfig
from .data.pipeline import DataPipelineState
from .setup import TrainingModels, TrainingOptimizers


CONDITIONAL_NAMES = (
    "plbert",
    "phoneme_encoder",
    "latent_phoneme_encoder",
    "duration_phoneme_encoder",
    "context_phoneme_encoder",
    "context_audio_encoder",
    "style_encoder",
    "voice_encoder",
    "language_embedding",
    "condition_bank",
    "duration_predictor",
    "latent_flow",
    "aligner",
    "style_speaker_classifier",
    "style_statistics_head",
    "voice_ge2e",
    "style_ge2e",
)
POSTERIOR_NAMES = (
    "audio_encoder",
    "decoder",
    "generator",
    "feature_linear",
    "discriminators",
)


@dataclass(frozen=True)
class ResumeState:
    step: int
    batch_index: int
    pipeline: DataPipelineState
    mlflow_run_id: str


class CheckpointManager:
    def __init__(
        self,
        output_path: Path,
        config: BeetleConfig,
        accelerator: Accelerator,
    ) -> None:
        self.root = output_path / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.accelerator = accelerator
        self.inherited_model_states: dict[str, dict[str, Any]] = {}

    def initialize(self, path: Path, models: TrainingModels) -> None:
        payload = torch.load(
            self._checkpoint_file(path),
            map_location="cpu",
            weights_only=False,
        )
        states = payload["models"]
        self.inherited_model_states = states
        targets = self._model_targets(models)
        shared = states.keys() & targets.keys()
        for name in shared:
            targets[name].load_state_dict(states[name])
        if not shared:
            raise ValueError(f"checkpoint has no weights for this stage: {path}")

    def resume(
        self,
        path: Path,
        models: TrainingModels,
        optimizers: TrainingOptimizers,
    ) -> ResumeState:
        payload = torch.load(
            self._checkpoint_file(path),
            map_location="cpu",
            weights_only=False,
        )
        self.inherited_model_states = payload["models"]
        targets = self._model_targets(models)
        missing = targets.keys() - self.inherited_model_states.keys()
        if missing:
            raise ValueError(f"resume checkpoint is missing model states: {sorted(missing)}")
        for name, module in targets.items():
            module.load_state_dict(self.inherited_model_states[name])
        optimizers.generator.load_state_dict(payload["generator_optimizer"])
        if optimizers.discriminator is not None:
            optimizers.discriminator.load_state_dict(payload["discriminator_optimizer"])
        scaler = payload["scaler"]
        if self.accelerator.scaler is not None and scaler is not None:
            self.accelerator.scaler.load_state_dict(scaler)
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        return ResumeState(
            step=int(payload["step"]),
            batch_index=int(payload["batch_index"]),
            pipeline=payload["pipeline"],
            mlflow_run_id=str(payload["mlflow_run_id"]),
        )

    def save(
        self,
        step: int,
        batch_index: int,
        pipeline: DataPipelineState,
        mlflow_run_id: str,
        models: TrainingModels,
        optimizers: TrainingOptimizers,
    ) -> Path:
        self.accelerator.wait_for_everyone()
        output = self.root / f"step_{step}"
        if not self.accelerator.is_main_process:
            return output
        known_names = {f"posterior.{name}" for name in POSTERIOR_NAMES}
        known_names.update(f"conditional.{name}" for name in CONDITIONAL_NAMES)
        states = {
            name: state
            for name, state in self.inherited_model_states.items()
            if name in known_names
        }
        states.update(
            {
                name: self.accelerator.unwrap_model(module).state_dict()
                for name, module in self._model_targets(models).items()
            }
        )
        payload = {
            "version": 1,
            "stage": self.config.training.stage.value,
            "step": step,
            "batch_index": batch_index,
            "pipeline": pipeline,
            "mlflow_run_id": mlflow_run_id,
            "models": states,
            "generator_optimizer": optimizers.generator.state_dict(),
            "discriminator_optimizer": (
                optimizers.discriminator.state_dict()
                if optimizers.discriminator is not None
                else None
            ),
            "scaler": (
                self.accelerator.scaler.state_dict()
                if self.accelerator.scaler is not None
                else None
            ),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        temporary = self.root / f".step_{step}.{os.getpid()}.tmp"
        temporary.mkdir(parents=False, exist_ok=False)
        torch.save(payload, temporary / "checkpoint.pt")
        os.replace(temporary, output)
        self.inherited_model_states = states
        checkpoints = sorted(
            (
                item
                for item in self.root.glob("step_*")
                if item.is_dir() and item.name.removeprefix("step_").isdigit()
            ),
            key=lambda item: int(item.name.removeprefix("step_")),
        )
        for expired in checkpoints[: -self.config.checkpoint.keep_last]:
            shutil.rmtree(expired)
        return output

    def _model_targets(self, models: TrainingModels) -> dict[str, nn.Module]:
        acoustic = models.acoustic
        conditional = models.conditional
        posterior = acoustic if acoustic is not None else conditional
        assert posterior is not None
        targets: dict[str, nn.Module] = {
            "posterior.audio_encoder": posterior.audio_encoder,
            "posterior.feature_linear": posterior.feature_linear,
            "posterior.decoder": posterior.decoder,
            "posterior.generator": posterior.generator,
        }
        if acoustic is not None:
            targets.update(
                {
                    "posterior.discriminators": acoustic.discriminators,
                }
            )
        if conditional is not None:
            targets.update(
                {
                    f"conditional.{name}": getattr(conditional, name)
                    for name in CONDITIONAL_NAMES
                }
            )
        return targets

    def _checkpoint_file(self, path: Path) -> Path:
        checkpoint = path / "checkpoint.pt" if path.is_dir() else path
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint file does not exist: {checkpoint}")
        return checkpoint
