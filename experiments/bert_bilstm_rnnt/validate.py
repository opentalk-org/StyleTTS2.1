from __future__ import annotations

from pathlib import Path

import torch

from experiments.bert_g2p_asr_ppo.assets import resolve_assets
from experiments.bert_g2p_asr_ppo.data import Codec, collate, download_parquets, parquet_rows, shuffled_batches

from .config import ExperimentConfig
from .model import BertBiLstmRnnt
from .train import _alignment_counts, _rnnt_targets


@torch.no_grad()
def validate_checkpoint(
    checkpoint_path: Path,
    config: ExperimentConfig,
    beam_width: int,
    batch_size: int,
    validation_batches: int,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assets = resolve_assets(config.assets)
    codec = Codec(assets.bert)
    model = BertBiLstmRnnt(assets.bert, config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    paths = download_parquets(config.data, validation=True)
    rows = parquet_rows(paths, config.data)
    batches = shuffled_batches(rows, batch_size, config.seed)
    matches = 0
    insertions = 0
    deletions = 0
    substitutions = 0
    references = 0
    exact_matches = 0
    items = 0
    for index, values in enumerate(batches):
        batch = collate(values, codec).to(device)
        targets, lengths = _rnnt_targets(batch, codec)
        predictions = model.beam_decode(
            batch.input_ids,
            batch.attention_mask,
            batch.language_ids,
            beam_width,
            config.train.max_symbols_per_timestep,
        )
        for predicted, target, length in zip(predictions, targets.tolist(), lengths.tolist(), strict=True):
            reference = target[:length]
            counts = _alignment_counts(predicted, reference)
            matches += counts[0]
            insertions += counts[1]
            deletions += counts[2]
            substitutions += counts[3]
            references += length
            exact_matches += int(predicted == reference)
            items += 1
        if index + 1 == validation_batches:
            break
    denominator = 2 * matches + insertions + deletions + 2 * substitutions
    return {
        "accuracy": matches / references,
        "f1": 2 * matches / denominator,
        "exact_match": exact_matches / items,
        "items": float(items),
    }
