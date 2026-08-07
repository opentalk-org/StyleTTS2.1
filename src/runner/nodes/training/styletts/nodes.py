from __future__ import annotations

from enum import Enum
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml
from pydantic import Field

from shared.logging_setup import get_logger

logger = get_logger(__name__)

from runflow.core.node import Node
from runflow.core.ports import JoinMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.accelerator_memory import release_accelerator_memory
from runner.nodes.datatypes import AssetBundlePort, CheckpointRefPort, JsonPort, TrainingResultPort
from runner.nodes.models import AssetBundleRef, CheckpointRef, TrainingResult, stable_id, typed_assets, typed_checkpoint
from runner.nodes.training.common.run_directory import claim_run_dir, remove_run_dir
from runner.nodes.training.common.mlflow_run import TrackerRun, start_mlflow_run
from runflow.runtime.cancellation import check_cancel
from runner.nodes.training.styletts.finetune.node_config import build_node_config
from runner.nodes.training.styletts.finetune.stages import (
    TrainingStageSpec,
    default_training_stages,
)
from shared.db.assets import crud as asset_crud
from shared.db.connection import database_session


class NumericPrecision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class DecoderBackend(str, Enum):
    HIFIGAN = "hifigan"
    ISTFTNET = "istftnet"


class StyleTtsFinetuneSettings(StrictSettings):
    display_name: str = Field(default="styletts_finetune", title="Display name")
    mlflow_run_id: str = Field(default="", title="Existing MLflow run ID")
    seed: int = Field(default=1, title="Random seed", ge=0)
    output_checkpoint_dir: str = Field(default="", title="External output checkpoint folder")
    validation_samples: int = Field(default=32, title="Validation samples", ge=1, le=512)
    learning_rate: float = Field(default=1e-4, title="Learning rate", gt=0)
    numeric_precision: NumericPrecision = Field(default=NumericPrecision.FP32, title="Numeric precision")
    training_stages: list[TrainingStageSpec] = Field(
        default_factory=default_training_stages,
        title="Training stages",
        min_length=1,
    )
    validation_interval_steps: int = Field(
        default=500,
        title="Validation interval",
        ge=1,
    )
    checkpoint_interval_steps: int = Field(
        default=2000,
        title="Checkpoint interval",
        ge=1,
    )
    log_interval_steps: int = Field(default=10, title="Log interval", ge=1)
    distributed_processes: int = Field(
        default=1,
        title="Distributed processes",
        ge=1,
        le=8,
    )
    load_optimizer: bool = Field(default=False, title="Load optimizer state")
    decoder: DecoderBackend = Field(default=DecoderBackend.HIFIGAN, title="Decoder")
    multispeaker: bool = Field(default=True, title="Multi-speaker mode")
    checkpoint_decoder_gradients: bool = Field(default=True, title="Checkpoint decoder gradients")
    checkpoint_discriminator_gradients: bool = Field(default=False, title="Checkpoint discriminator gradients")
    profiling_enabled: bool = Field(default=False, title="Training profiler")
    config_output_dir: str = Field(default="", title="Config output folder")


class StyleTtsFinetuneNode(Node):
    NODE_TYPE = "StyleTtsFinetune"
    DESCRIPTION = "Finetune StyleTTS2 directly from a stored dataset. Metadata, phonemes, and audio are streamed from the database without an intermediate manifest."
    CATEGORY = "Training"
    SETTINGS = StyleTtsFinetuneSettings
    INPUTS = {
        "dataset_ref": JsonPort(),
        "phoneme_alphabet": JsonPort(join_mode=JoinMode.BROADCAST),
        "checkpoint": CheckpointRefPort(join_mode=JoinMode.BROADCAST),
        "assets": AssetBundlePort(join_mode=JoinMode.BROADCAST),
    }
    OUTPUTS = {"training_result": TrainingResultPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 12}, exclusive_group="accelerator")

    async def teardown(self, context) -> None:
        release_accelerator_memory()

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            dataset_id = str(inputs["dataset_ref"]["dataset_id"])
            phoneme_alphabet = [
                str(symbol)
                for symbol in inputs["phoneme_alphabet"]["symbol_list"]
            ]
            run_dir = _config_output_dir(
                self.settings.config_output_dir,
                str(context.run_id),
            )
            training_config = build_styletts_finetune_config(
                dataset_id=dataset_id,
                phoneme_alphabet=phoneme_alphabet,
                base_checkpoint=typed_checkpoint(inputs["checkpoint"]),
                pretrained_assets=typed_assets(inputs["assets"]),
                settings=self.settings,
                output_dir=run_dir / "configs",
            )
            with claim_run_dir(run_dir):
                config_path = _prepare_styletts_config(training_config, str(context.run_id))
                try:
                    _run_styletts_train(config_path)
                    result = _latest_step_result(str(context.run_id))
                finally:
                    remove_run_dir(run_dir)
            outputs.append({"training_result": result})
        return outputs


def build_styletts_finetune_config(
    dataset_id: str,
    phoneme_alphabet: list[str],
    base_checkpoint: CheckpointRef,
    pretrained_assets: AssetBundleRef | None,
    settings: StyleTtsFinetuneSettings,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path, styletts_yaml = build_node_config(
        dataset_id=dataset_id,
        validation_samples=settings.validation_samples,
        phoneme_alphabet=phoneme_alphabet,
        base_checkpoint=base_checkpoint,
        pretrained_assets=pretrained_assets,
        settings=settings,
        output_dir=output_dir,
    )
    return {
        "version": 2,
        "node_type": StyleTtsFinetuneNode.NODE_TYPE,
        "config_path": str(config_path),
        "dataset": {
            "dataset_id": dataset_id,
            "validation_samples": settings.validation_samples,
        },
        "base_checkpoint": _checkpoint_payload(base_checkpoint),
        "pretrained_assets": _assets_payload(pretrained_assets),
        "training": settings.model_dump(mode="json"),
        "styletts_yaml": styletts_yaml,
    }


def _prepare_styletts_config(training_config: dict[str, Any], run_id: str) -> Path:
    config_path = Path(str(training_config["config_path"]))
    styletts_yaml = dict(training_config["styletts_yaml"])
    Path(str(styletts_yaml["log_dir"])).mkdir(parents=True, exist_ok=True)
    publish = dict(styletts_yaml["studio_publish"])
    publish["run_id"] = run_id
    publish["finetune_job_id"] = run_id
    styletts_yaml["studio_publish"] = publish
    config_path.write_text(yaml.safe_dump(styletts_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config_path


def _run_styletts_train(config_path: Path) -> None:
    from runner.nodes.training.styletts.finetune.training.train_finetune import train

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    process_count = int(config["distributed_processes"])
    if process_count > 1:
        precision = {
            "fp32": "no",
            "fp16": "fp16",
            "bf16": "bf16",
        }[str(config["precision"])]
        _run_distributed_training(config_path, process_count, precision)
        return
    run = _make_mlflow_run(config_path)
    try:
        train(str(config_path), run=run)
    finally:
        try:
            run.close()
        except Exception:
            logger.warning("failed to close MLflow run", exc_info=True)
        release_accelerator_memory()


def _run_distributed_training(
    config_path: Path,
    process_count: int,
    precision: str,
) -> None:
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(process_count),
        "--num_machines",
        "1",
        "--mixed_precision",
        precision,
        "--dynamo_backend",
        "no",
        "--multi_gpu",
        "-m",
        "runner.nodes.training.styletts.finetune.training.distributed",
        str(config_path),
    ]
    distributed_log_path = config_path.parent / "distributed.log"
    with distributed_log_path.open(mode="w+") as output:
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            while process.poll() is None:
                check_cancel()
                time.sleep(0.1)
        except BaseException:
            process.terminate()
            process.wait(timeout=10)
            raise
        output.seek(0)
        distributed_log = output.read()
    if process.returncode:
        raise RuntimeError(
            f"distributed training exited with code {process.returncode}\n"
            f"{distributed_log}"
        )
    logger.info("distributed training completed\n%s", distributed_log)


def _make_mlflow_run(config_path: Path) -> TrackerRun:
    """Start an MLflow run named from the training configuration."""
    try:
        styletts_yaml = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        styletts_yaml = {}
    publish = styletts_yaml.get("studio_publish", {}) if isinstance(styletts_yaml, dict) else {}
    run_name = str(publish.get("run_name") or publish.get("run_id") or "styletts_finetune")
    config = {
        "run_id": publish.get("run_id"),
        "finetune_job_id": publish.get("finetune_job_id"),
        "total_steps": styletts_yaml.get("total_steps"),
        "distributed_processes": styletts_yaml.get("distributed_processes"),
        "stage_max_audio_seconds": ",".join(
            str(stage.get("max_audio_seconds"))
            for stage in styletts_yaml.get("training_stages", [])
        ),
        "precision": styletts_yaml.get("precision"),
        "n_token": (styletts_yaml.get("model_params") or {}).get("n_token"),
        "decoder": ((styletts_yaml.get("model_params") or {}).get("decoder") or {}).get("type"),
    }
    return start_mlflow_run(experiment="styletts2_finetune", name=run_name, config=config)


def _latest_step_result(run_id: str) -> TrainingResult:
    with database_session() as session:
        checkpoints = [
            item
            for item in asset_crud.list_checkpoints(session)
            if item.metadata_.get("source") == "finetune_step" and item.metadata_.get("finetune_job_id") == run_id
        ]
        if not checkpoints:
            raise RuntimeError("StyleTTS finetune did not publish any step checkpoints")
        checkpoint = max(checkpoints, key=lambda item: item.updated_at if hasattr(item, "updated_at") else item.name)
        path = asset_crud.get_checkpoint_path(session, checkpoint.id)
    checkpoint_ref_id = stable_id("checkpoint", checkpoint.id, checkpoint.content_hash)
    checkpoint_ref = CheckpointRef(
        checkpoint_id=checkpoint.id,
        name=checkpoint.name,
        path=path,
        id=checkpoint_ref_id,
        lineage_id=checkpoint_ref_id,
        metadata={"type": checkpoint.type_, "content_hash": checkpoint.content_hash, "metadata": checkpoint.metadata_},
    )
    result_id = stable_id("training_result", "StyleTtsFinetune", run_id, checkpoint.id)
    return TrainingResult(
        training_run_id=stable_id("training_run", "StyleTtsFinetune", run_id),
        checkpoint=checkpoint_ref,
        id=result_id,
        lineage_id=result_id,
        metadata={"node_type": "StyleTtsFinetune", "checkpoint_id": str(checkpoint.id)},
    )


def _config_output_dir(configured: str, run_id: str) -> Path:
    if configured:
        return Path(configured)
    return Path("data/training/styletts") / run_id


def _checkpoint_payload(ref: CheckpointRef) -> dict[str, Any]:
    return {"id": str(ref.checkpoint_id), "name": ref.name, "path": str(ref.path), "metadata": ref.metadata}


def _assets_payload(ref: AssetBundleRef | None) -> dict[str, Any] | None:
    if ref is None:
        return None
    return {
        "bundle_key": ref.bundle_key,
        "name": ref.name,
        "paths": [str(path) for path in ref.paths],
        "extra_file_ids": [str(file_id) for file_id in ref.extra_file_ids],
        "metadata": ref.metadata,
    }
