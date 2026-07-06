from __future__ import annotations

from collections.abc import Mapping

import torch


def merge_state_dict_with_dim0_resize(
    model: torch.nn.Module,
    checkpoint_state_dict: Mapping[str, torch.Tensor],
    resize_dim0_keys: frozenset[str],
    *,
    error_scope: str,
    on_incompatible_shape: str = "raise",
) -> dict[str, torch.Tensor]:
    if on_incompatible_shape not in {"raise", "keep_model"}:
        raise ValueError("on_incompatible_shape_invalid")
    model_state_dict = model.state_dict()
    merged: dict[str, torch.Tensor] = {}

    for key, model_tensor in model_state_dict.items():
        if key not in checkpoint_state_dict:
            merged[key] = model_tensor
            continue

        checkpoint_tensor = checkpoint_state_dict[key]
        if checkpoint_tensor.shape == model_tensor.shape:
            merged[key] = checkpoint_tensor
            continue

        if key in resize_dim0_keys and checkpoint_tensor.shape[1:] == model_tensor.shape[1:]:
            resized = model_tensor.clone()
            n = min(checkpoint_tensor.shape[0], model_tensor.shape[0])
            resized[:n] = checkpoint_tensor[:n]
            merged[key] = resized
            continue

        if on_incompatible_shape == "keep_model":
            merged[key] = model_tensor
            continue

        raise RuntimeError(
            f"{error_scope} checkpoint incompatible key {key}: checkpoint {tuple(checkpoint_tensor.shape)} vs model {tuple(model_tensor.shape)}"
        )

    return merged
