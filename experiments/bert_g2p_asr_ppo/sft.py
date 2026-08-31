from __future__ import annotations

import time
from pathlib import Path

import torch

from .assets import resolve_assets
from .config import ExperimentConfig
from .data import Codec, collate, download_parquets, parquet_rows, shuffled_batches
from .model import BertG2P
from .tracking import ExperimentRun


def train_sft(config: ExperimentConfig) -> tuple[Path, str]:
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assets = resolve_assets(config.assets)
    codec = Codec(assets.bert)
    paths = download_parquets(config.data)
    validation_paths = download_parquets(config.data, validation=True)
    batches = _sft_batches(paths, config)
    model = BertG2P(assets.bert).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.sft.learning_rate,
        weight_decay=config.sft.weight_decay,
    )
    output = config.output_dir / config.run_name / "sft"
    output.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentRun(config.run_name, config.payload())
    started = time.monotonic()
    try:
        for step, rows in enumerate(batches, 1):
            batch = collate(rows, codec).to(device)
            output_values = model(
                batch.input_ids,
                batch.attention_mask,
                batch.decoder_input_ids,
                batch.language_ids,
                batch.labels,
            )
            assert output_values.loss is not None
            optimizer.zero_grad(set_to_none=True)
            output_values.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % config.sft.log_interval == 0 or step == config.sft.steps:
                tracker.metrics(
                    {
                        "sft/loss": output_values.loss.detach().item(),
                        "sft/gradient_norm": gradient_norm.detach().item(),
                        "sft/steps_per_second": step / (time.monotonic() - started),
                        "sft/items_per_second": step * config.sft.batch_size / (time.monotonic() - started),
                    },
                    step,
                )
            if step % config.sft.validation_interval == 0 or step == config.sft.steps:
                tracker.metrics(_validate(model, validation_paths, codec, config, device), step)
            if step % config.sft.checkpoint_interval == 0 or step == config.sft.steps:
                checkpoint = _save(output, model, step)
                tracker.artifact(checkpoint, "sft/checkpoints")
            if step == config.sft.steps:
                tracker.close()
                return checkpoint, tracker.run_id
    except BaseException:
        tracker.close(failed=True)
        raise
    raise RuntimeError("SFT produced no batches")


def _save(output: Path, model: BertG2P, step: int) -> Path:
    path = output / f"step_{step:09d}.pth"
    torch.save({"model": model.state_dict(), "step": step}, path)
    return path


def _sft_batches(paths, config: ExperimentConfig):
    epoch = 0
    while True:
        rows = parquet_rows(paths, config.data)
        yield from shuffled_batches(rows, config.sft.batch_size, config.seed + epoch)
        epoch += 1


@torch.no_grad()
def _validate(model, paths, codec: Codec, config: ExperimentConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    token_count = 0
    edit_distance = 0
    reference_count = 0
    exact_matches = 0
    item_count = 0
    rows = parquet_rows(paths, config.data)
    batches = shuffled_batches(rows, config.sft.batch_size, config.seed)
    for index, items in enumerate(batches):
        batch = collate(items, codec).to(device)
        output = model(
            batch.input_ids,
            batch.attention_mask,
            batch.decoder_input_ids,
            batch.language_ids,
            batch.labels,
        )
        assert output.loss is not None
        losses.append(output.loss.item())
        mask = batch.labels.ne(-100)
        correct += int((output.logits.argmax(-1).eq(batch.labels) & mask).sum())
        token_count += int(mask.sum())
        if index < config.sft.autoregressive_validation_batches:
            max_tokens = min(config.sft.validation_generation_tokens, int(mask.sum(1).max()) + 16)
            generated = model.generate(batch.input_ids, batch.attention_mask, batch.language_ids, max_tokens, sample=False)
            for predicted, target in zip(generated[:, 1:], batch.labels, strict=True):
                predicted_ids = _sequence(predicted.tolist(), codec)
                target_ids = [token for token in target.tolist() if token != -100 and token != codec.eos_id]
                edit_distance += _edit_distance(predicted_ids, target_ids)
                reference_count += len(target_ids)
                exact_matches += int(predicted_ids == target_ids)
                item_count += 1
        if index + 1 == config.sft.validation_batches:
            break
    model.train()
    if not losses:
        raise RuntimeError("validation parquet produced no complete batches")
    return {
        "sft/validation_loss": sum(losses) / len(losses),
        "sft/validation_token_accuracy": correct / token_count,
        "sft/validation_phoneme_error_rate": edit_distance / reference_count,
        "sft/validation_exact_match": exact_matches / item_count,
    }


def _sequence(tokens: list[int], codec: Codec) -> list[int]:
    end = tokens.index(codec.eos_id) if codec.eos_id in tokens else len(tokens)
    return [token for token in tokens[:end] if token != codec.pad_id]


def _edit_distance(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, 1):
        current = [left_index]
        for right_index, right_token in enumerate(right, 1):
            substitution = previous[right_index - 1] + int(left_token != right_token)
            current.append(min(current[-1] + 1, previous[right_index] + 1, substitution))
        previous = current
    return previous[-1]
