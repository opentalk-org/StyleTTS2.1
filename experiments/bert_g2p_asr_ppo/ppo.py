from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import Tensor

from .assets import resolve_assets
from .config import ExperimentConfig
from .data import BackendPartition, Codec, TextPhonemeRow, backend_audio_batches, collate
from .model import BertG2P
from .reward import FrozenAlignerLoss
from .tracking import ExperimentRun


@dataclass(frozen=True)
class Rollout:
    tokens: Tensor
    old_log_probs: Tensor
    reference_log_probs: Tensor
    mask: Tensor
    asr_losses: Tensor


def train_ppo(config: ExperimentConfig, sft_checkpoint: Path, mlflow_run_id: str) -> Path:
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assets = resolve_assets(config.assets)
    codec = Codec(assets.bert)
    policy = BertG2P(assets.bert).to(device)
    policy.load_state_dict(torch.load(sft_checkpoint, map_location="cpu", weights_only=False)["model"])
    policy.eval()
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    aligner_loss = FrozenAlignerLoss(config.assets, assets, device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.ppo.learning_rate)
    tracker = ExperimentRun(config.run_name, config.payload(), mlflow_run_id)
    output = config.output_dir / config.run_name / "ppo"
    output.mkdir(parents=True, exist_ok=True)
    source = iter(
        backend_audio_batches(
            config.ppo.dataset_id,
            config.ppo.batch_size,
            config.data.language,
            BackendPartition.TRAIN,
        )
    )
    try:
        for step in range(1, config.ppo.steps + 1):
            try:
                rows = next(source)
            except StopIteration:
                source = iter(
                    backend_audio_batches(
                        config.ppo.dataset_id,
                        config.ppo.batch_size,
                        config.data.language,
                        BackendPartition.TRAIN,
                    )
                )
                rows = next(source)
            text_rows = [TextPhonemeRow(row.language, row.text, row.phonemes) for row in rows]
            batch = collate(text_rows, codec).to(device)
            rollout = _rollout(policy, reference, aligner_loss, batch, rows, codec, config.ppo.rollout_tokens, sample=True)
            metrics = _update(policy, optimizer, batch, rollout, config)
            if step % config.ppo.log_interval == 0 or step == config.ppo.steps:
                tracker.metrics({"ppo/asr_loss": rollout.asr_losses.mean().item(), **metrics}, step)
            if step % config.ppo.validation_interval == 0 or step == config.ppo.steps:
                tracker.metrics(_validate(policy, reference, aligner_loss, codec, config, device), step)
        checkpoint = output / f"step_{config.ppo.steps:09d}.pth"
        torch.save({"model": policy.state_dict(), "step": config.ppo.steps}, checkpoint)
        tracker.artifact(checkpoint, "ppo/checkpoints")
        tracker.close()
        return checkpoint
    except BaseException:
        tracker.close(failed=True)
        raise


@torch.no_grad()
def _rollout(policy, reference, aligner_loss, batch, rows, codec, max_tokens: int, sample: bool) -> Rollout:
    tokens = policy.generate(batch.input_ids, batch.attention_mask, batch.language_ids, max_tokens, sample=sample)
    old = _token_log_probs(policy, batch, tokens)
    reference_log_probs = _token_log_probs(reference, batch, tokens)
    mask = _token_mask(tokens[:, 1:], codec.eos_id, codec.pad_id)
    phonemes = [codec.decode(row.tolist()) for row in tokens[:, 1:]]
    asr_losses = aligner_loss(rows, phonemes)
    return Rollout(tokens, old, reference_log_probs, mask, asr_losses)


def _update(policy, optimizer, batch, rollout: Rollout, config: ExperimentConfig) -> dict[str, float]:
    rewards = -rollout.asr_losses
    advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-5)
    totals = {"ppo/policy_loss": 0.0, "ppo/kl": 0.0, "ppo/clip_fraction": 0.0}
    for _ in range(config.ppo.update_epochs):
        current = _token_log_probs(policy, batch, rollout.tokens)
        sequence_current = _masked_mean(current, rollout.mask)
        sequence_old = _masked_mean(rollout.old_log_probs, rollout.mask)
        ratio = (sequence_current - sequence_old).exp()
        unclipped = ratio * advantages
        clipped = ratio.clamp(1.0 - config.ppo.clip_ratio, 1.0 + config.ppo.clip_ratio) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        log_ratio = rollout.reference_log_probs - current
        kl = _masked_mean(log_ratio.exp() - log_ratio - 1.0, rollout.mask).mean()
        entropy = _masked_mean(-current, rollout.mask).mean()
        loss = policy_loss + config.ppo.kl_coefficient * kl - config.ppo.entropy_coefficient * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        totals["ppo/policy_loss"] += policy_loss.detach().item()
        totals["ppo/kl"] += kl.detach().item()
        totals["ppo/clip_fraction"] += (ratio.sub(1.0).abs() > config.ppo.clip_ratio).float().mean().item()
    return {key: value / config.ppo.update_epochs for key, value in totals.items()}


def _token_log_probs(model: BertG2P, batch, tokens: Tensor) -> Tensor:
    output = model(batch.input_ids, batch.attention_mask, tokens[:, :-1], batch.language_ids)
    logits = model.mask_generation_logits(output.logits)
    return functional.log_softmax(logits, -1).gather(-1, tokens[:, 1:, None]).squeeze(-1)


def _token_mask(tokens: Tensor, eos_id: int, pad_id: int) -> Tensor:
    positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    first_eos = torch.where(tokens.eq(eos_id), positions, tokens.shape[1]).amin(1)
    return tokens.ne(pad_id) & positions.le(first_eos.unsqueeze(1))


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum(1) / mask.sum(1).clamp_min(1)


@torch.no_grad()
def _validate(policy, reference, aligner_loss, codec, config: ExperimentConfig, device) -> dict[str, float]:
    source = backend_audio_batches(
        config.ppo.dataset_id,
        config.ppo.batch_size,
        config.data.language,
        BackendPartition.VALIDATION,
    )
    asr_losses = []
    divergences = []
    for index, rows in enumerate(source):
        text_rows = [TextPhonemeRow(row.language, row.text, row.phonemes) for row in rows]
        batch = collate(text_rows, codec).to(device)
        rollout = _rollout(policy, reference, aligner_loss, batch, rows, codec, config.ppo.rollout_tokens, sample=False)
        log_ratio = rollout.reference_log_probs - rollout.old_log_probs
        divergence = _masked_mean(log_ratio.exp() - log_ratio - 1.0, rollout.mask)
        asr_losses.extend(rollout.asr_losses.tolist())
        divergences.extend(divergence.tolist())
        if index + 1 == config.ppo.validation_batches:
            break
    if not asr_losses:
        raise RuntimeError("backend validation partition produced no complete batches")
    return {
        "ppo/validation_asr_loss": sum(asr_losses) / len(asr_losses),
        "ppo/validation_kl": sum(divergences) / len(divergences),
    }
