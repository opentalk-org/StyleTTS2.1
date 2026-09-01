from __future__ import annotations

import time
from pathlib import Path

import torch

from experiments.bert_g2p_asr_ppo.assets import resolve_assets
from experiments.bert_g2p_asr_ppo.data import Codec, TokenBatch, collate, download_parquets, parquet_rows, shuffled_batches

from .config import ExperimentConfig
from .model import BertBiLstmRnnt
from .tracking import ExperimentRun


def train(
    config: ExperimentConfig,
    checkpoint_path: Path | None = None,
    resume_optimizer: bool = False,
) -> tuple[Path, str]:
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assets = resolve_assets(config.assets)
    codec = Codec(assets.bert)
    train_paths = download_parquets(config.data)
    validation_paths = download_parquets(config.data, validation=True)
    model = BertBiLstmRnnt(assets.bert, config.model).to(device)
    initial_step = 0
    if checkpoint_path is not None:
        checkpoint_payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint_payload["model"], strict=True)
        initial_step = int(checkpoint_payload["step"])
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    if resume_optimizer:
        assert checkpoint_path is not None, "optimizer continuation requires a checkpoint"
        optimizer.load_state_dict(checkpoint_payload["optimizer"])
    output_dir = config.output_dir / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentRun(config.run_name, config.payload())
    started = time.monotonic()
    checkpoint = output_dir / "untrained.pth"
    nonfinite_loss_skips = 0
    nonfinite_gradient_skips = 0
    try:
        for local_step, rows in enumerate(_training_batches(train_paths, config), 1):
            step = initial_step + local_step
            batch = collate(rows, codec).to(device)
            targets, lengths = _rnnt_targets(batch, codec)
            result = model(batch.input_ids, batch.attention_mask, batch.language_ids, targets, lengths)
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(result.loss):
                nonfinite_loss_skips += 1
                tracker.metrics({"train/nonfinite_loss_skips": nonfinite_loss_skips}, step)
                continue
            result.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, config.train.gradient_clip)
            if not torch.isfinite(gradient_norm):
                nonfinite_gradient_skips += 1
                optimizer.zero_grad(set_to_none=True)
                tracker.metrics({"train/nonfinite_gradient_skips": nonfinite_gradient_skips}, step)
                continue
            optimizer.step()
            final_step = initial_step + config.train.steps
            if step % config.train.log_interval == 0 or step == final_step:
                elapsed = time.monotonic() - started
                tracker.metrics({
                    "train/loss": result.loss.detach().item(),
                    "train/gradient_norm": gradient_norm.detach().item(),
                    "train/steps_per_second": local_step / elapsed,
                    "train/items_per_second": local_step * config.train.batch_size / elapsed,
                }, step)
            if step % config.train.validation_interval == 0 or step == final_step:
                tracker.metrics(validate(model, validation_paths, codec, config, device), step)
            if step % config.train.checkpoint_interval == 0 or step == final_step:
                checkpoint = _save(output_dir, model, optimizer, step)
                tracker.artifact(checkpoint, "checkpoints")
            if step == final_step:
                tracker.close()
                return checkpoint, tracker.run_id
    except BaseException:
        tracker.close(failed=True)
        raise
    raise RuntimeError("training parquet produced no complete batches")


@torch.no_grad()
def validate(model, paths, codec: Codec, config: ExperimentConfig, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    edits = 0
    matches = 0
    insertions = 0
    deletions = 0
    substitutions = 0
    reference_tokens = 0
    exact_matches = 0
    item_count = 0
    rows = parquet_rows(paths, config.data)
    batches = shuffled_batches(rows, config.train.batch_size, config.seed)
    for index, items in enumerate(batches):
        batch = collate(items, codec).to(device)
        targets, lengths = _rnnt_targets(batch, codec)
        result = model(batch.input_ids, batch.attention_mask, batch.language_ids, targets, lengths)
        losses.append(result.loss.item())
        predictions = model.greedy_decode(
            batch.input_ids, batch.attention_mask, batch.language_ids, config.train.max_symbols_per_timestep
        )
        for predicted, target, length in zip(predictions, targets.tolist(), lengths.tolist(), strict=True):
            reference = target[:length]
            counts = _alignment_counts(predicted, reference)
            matches += counts[0]
            insertions += counts[1]
            deletions += counts[2]
            substitutions += counts[3]
            edits += sum(counts[1:])
            reference_tokens += length
            exact_matches += int(predicted == reference)
            item_count += 1
        if index + 1 == config.train.validation_batches:
            break
    model.train()
    if not losses:
        raise RuntimeError("validation parquet produced no complete batches")
    f1_denominator = 2 * matches + insertions + deletions + 2 * substitutions
    return {
        "validation/loss": sum(losses) / len(losses),
        "validation/phoneme_error_rate": edits / reference_tokens,
        "validation/accuracy": matches / reference_tokens,
        "validation/f1": 2 * matches / f1_denominator,
        "validation/exact_match": exact_matches / item_count,
    }


def _rnnt_targets(batch: TokenBatch, codec: Codec) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch.labels.clone()
    labels[labels.eq(codec.eos_id)] = -100
    lengths = labels.ne(-100).sum(1)
    targets = labels[:, : int(lengths.max())]
    return targets.masked_fill(targets.eq(-100), 0), lengths


def _training_batches(paths, config: ExperimentConfig):
    epoch = 0
    while True:
        rows = parquet_rows(paths, config.data)
        yield from shuffled_batches(rows, config.train.batch_size, config.seed + epoch)
        epoch += 1


def _save(output: Path, model, optimizer, step: int) -> Path:
    path = output / f"step_{step:09d}.pth"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)
    return path


def _alignment_counts(predicted: list[int], reference: list[int]) -> tuple[int, int, int, int]:
    table = [[(0, 0, column, 0) for column in range(len(reference) + 1)]]
    for row in range(1, len(predicted) + 1):
        table.append([(0, row, 0, 0)])
        for column in range(1, len(reference) + 1):
            if predicted[row - 1] == reference[column - 1]:
                matched = table[row - 1][column - 1]
                table[row].append((matched[0] + 1, *matched[1:]))
                continue
            insertion = _increment(table[row - 1][column], 1)
            deletion = _increment(table[row][column - 1], 2)
            substitution = _increment(table[row - 1][column - 1], 3)
            table[row].append(min((insertion, deletion, substitution), key=lambda counts: sum(counts[1:])))
    return table[-1][-1]


def _increment(counts: tuple[int, int, int, int], index: int) -> tuple[int, int, int, int]:
    values = list(counts)
    values[index] += 1
    return tuple(values)
