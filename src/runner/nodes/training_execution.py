from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from runflow.core.node import Node
from runflow.core.ports import Port
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import JSON, TRAINING_MANIFEST, TRAINING_RESULT
from runner.nodes.training_config import (
    ASSET_BUNDLE_OR_JSON,
    CHECKPOINT_REF_OR_JSON,
    AsrTrainingSettings,
    F0TrainingSettings,
    StyleTtsFinetuneSettings,
    publish_training_result,
    training_config_output_dir,
)


class StyleTtsFinetuneNode(Node):
    NODE_TYPE = "StyleTtsFinetune"
    CATEGORY = "Training"
    SETTINGS = StyleTtsFinetuneSettings
    INPUTS = {
        "audio_file_ids": Port("audio_file_ids", JSON),
        "base_checkpoint": Port("base_checkpoint", CHECKPOINT_REF_OR_JSON),
        "pretrained_assets": Port("pretrained_assets", ASSET_BUNDLE_OR_JSON),
        "phoneme_alphabet": Port("phoneme_alphabet", JSON),
        "ood_text_sets": Port("ood_text_sets", JSON),
        "training_config": Port("training_config", JSON, optional=True),
    }
    OUTPUTS = {"training_result": Port("training_result", TRAINING_RESULT)}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 12}, exclusive_group="accelerator")

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            payload = _styletts_payload(inputs, self.settings.model_dump(mode="json"))
            output_dir = self.settings.output_checkpoint_dir or _configured_output_dir(payload)
            _run_external_training(self.NODE_TYPE, self.settings.external_command, payload)
            outputs.append(
                {
                    "training_result": publish_training_result(
                        self.NODE_TYPE,
                        self.settings.display_name,
                        "styletts2",
                        output_dir,
                        payload,
                        str(context.run_id),
                    )
                }
            )
        return outputs


class F0ModelTrainingNode(Node):
    NODE_TYPE = "F0ModelTraining"
    CATEGORY = "Training"
    SETTINGS = F0TrainingSettings
    INPUTS = {
        "audio_file_ids": Port("audio_file_ids", JSON),
        "pretrained_checkpoint": Port("pretrained_checkpoint", CHECKPOINT_REF_OR_JSON),
        "training_manifest": Port("training_manifest", TRAINING_MANIFEST, optional=True),
    }
    OUTPUTS = {"training_result": Port("training_result", TRAINING_RESULT)}
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 6}, exclusive_group="accelerator")

    async def execute(self, batch, context):
        return [_training_result(self, inputs, "f0", context.run_id) for inputs in batch]


class AsrModelTrainingNode(Node):
    NODE_TYPE = "AsrModelTraining"
    CATEGORY = "Training"
    SETTINGS = AsrTrainingSettings
    INPUTS = {
        "audio_file_ids": Port("audio_file_ids", JSON),
        "pretrained_checkpoint": Port("pretrained_checkpoint", CHECKPOINT_REF_OR_JSON),
        "phoneme_alphabet": Port("phoneme_alphabet", JSON),
        "training_manifest": Port("training_manifest", TRAINING_MANIFEST, optional=True),
    }
    OUTPUTS = {"training_result": Port("training_result", TRAINING_RESULT)}
    RESOURCE_POLICY = ResourcePolicy(resources={"accelerator": 1, "vram_gb": 8}, exclusive_group="accelerator")

    async def execute(self, batch, context):
        return [_training_result(self, inputs, "asr_bundle", context.run_id) for inputs in batch]


def _training_result(node: Node, inputs: dict[str, Any], checkpoint_type: str, run_id: object) -> dict[str, object]:
    payload = {
        "node_type": node.NODE_TYPE,
        "audio_file_ids": inputs["audio_file_ids"],
        "training_manifest": _manifest_metadata(inputs["training_manifest"]) if inputs["training_manifest"] is not None else None,
        "pretrained_checkpoint": _checkpoint_metadata(inputs["pretrained_checkpoint"]),
        "phoneme_alphabet": inputs["phoneme_alphabet"] if "phoneme_alphabet" in inputs else None,
        "settings": node.settings.model_dump(mode="json"),
    }
    _run_external_training(node.NODE_TYPE, node.settings.external_command, payload)
    return {
        "training_result": publish_training_result(
            node.NODE_TYPE,
            node.settings.display_name,
            checkpoint_type,
            node.settings.output_checkpoint_dir,
            payload,
            str(run_id),
        )
    }


def _styletts_payload(inputs: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_type": StyleTtsFinetuneNode.NODE_TYPE,
        "audio_file_ids": inputs["audio_file_ids"],
        "base_checkpoint": _checkpoint_metadata(inputs["base_checkpoint"]),
        "pretrained_assets": _assets_metadata(inputs["pretrained_assets"]),
        "phoneme_alphabet": inputs["phoneme_alphabet"],
        "ood_text_sets": inputs["ood_text_sets"],
        "training_config": inputs["training_config"],
        "settings": settings,
    }


def _configured_output_dir(payload: dict[str, Any]) -> str:
    training_config = payload["training_config"]
    if training_config is None:
        return ""
    return training_config_output_dir(training_config)


def _run_external_training(node_type: str, command: list[str], payload: dict[str, Any]) -> None:
    if not command:
        raise RuntimeError(f"{node_type} requires external training command")
    with tempfile.TemporaryDirectory(prefix="runflow-training-") as tmp:
        payload_path = Path(tmp) / "payload.json"
        payload_path.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True), encoding="utf-8")
        env = {**os.environ, "RUNFLOW_TRAINING_PAYLOAD": str(payload_path), "RUNFLOW_TRAINING_NODE_TYPE": node_type}
        subprocess.run(command, check=True, env=env)


def _manifest_metadata(value) -> dict[str, object]:
    return {"id": value.id, "dataset_id": str(value.dataset_id), "metadata": value.metadata}


def _checkpoint_metadata(value):
    if value is None:
        return None
    if hasattr(value, "checkpoint_id"):
        return {"checkpoint_id": str(value.checkpoint_id), "name": value.name, "path": str(value.path)}
    return value


def _assets_metadata(value):
    if hasattr(value, "extra_file_ids"):
        return {
            "bundle_key": value.bundle_key,
            "name": value.name,
            "paths": [str(path) for path in value.paths],
            "extra_file_ids": [str(file_id) for file_id in value.extra_file_ids],
            "metadata": value.metadata,
        }
    return value
