from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

from runner.nodes.assets.model_downloads import single_checkpoint_file
from runner.nodes.smart_turn.audio import TARGET_SAMPLE_RATE, WINDOW_SAMPLES


@dataclass(frozen=True)
class SmartTurnBundle:
    feature_extractor: Any
    session: Any


def load_smart_turn_bundle(checkpoint_dir: Path) -> SmartTurnBundle:
    model_path = single_checkpoint_file(checkpoint_dir, (".onnx",))
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return SmartTurnBundle(
        feature_extractor=WhisperFeatureExtractor(chunk_length=8),
        session=ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        ),
    )


def predict_probabilities(bundle: SmartTurnBundle, waveforms: list[np.ndarray]) -> list[float]:
    inputs = bundle.feature_extractor(
        waveforms,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=WINDOW_SAMPLES,
        truncation=True,
        do_normalize=True,
    )
    features = np.asarray(inputs.input_features, dtype=np.float32)
    probabilities = np.asarray(
        bundle.session.run(None, {"input_features": features})[0],
        dtype=np.float32,
    ).reshape(-1)
    if probabilities.size != len(waveforms):
        raise RuntimeError(f"smart_turn_output_count_mismatch:{probabilities.size}:{len(waveforms)}")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("smart_turn_non_finite_probability")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise RuntimeError("smart_turn_probability_out_of_range")
    return [float(probability) for probability in probabilities]


def is_turn_complete(probability: float) -> bool:
    return probability > 0.5
