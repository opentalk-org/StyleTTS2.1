from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchaudio.functional import rnnt_loss
from transformers import BertConfig, BertModel

from experiments.bert_g2p_asr_ppo.assets import BertAsset

from .config import ModelConfig


@dataclass(frozen=True)
class RnntOutput:
    logits: Tensor
    loss: Tensor


@dataclass(frozen=True)
class BeamHypothesis:
    tokens: tuple[int, ...]
    score: float
    state: tuple[Tensor, Tensor]
    prediction: Tensor


class BertBiLstmRnnt(nn.Module):
    def __init__(self, asset: BertAsset, config: ModelConfig) -> None:
        super().__init__()
        geometry = _bert_geometry(asset.encoder_state)
        bert_config = BertConfig(**geometry, vocab_size=len(asset.symbols), pad_token_id=0)
        self.bert = BertModel(bert_config)
        self.bert.load_state_dict(asset.encoder_state, strict=True)
        self.language_embedding = nn.Embedding.from_pretrained(
            asset.language_weights.clone(), freeze=False, padding_idx=0
        )
        self.modality_embedding = nn.Embedding.from_pretrained(asset.modality_weights.clone(), freeze=False)
        self.encoder = nn.LSTM(
            geometry["hidden_size"],
            config.encoder_hidden_size,
            config.encoder_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.encoder_layers > 1 else 0.0,
        )
        self.blank_id = len(asset.symbols)
        self.vocab_size = self.blank_id + 1
        self.predictor_embedding = nn.Embedding(self.vocab_size, config.predictor_hidden_size)
        self.predictor = nn.LSTM(
            config.predictor_hidden_size,
            config.predictor_hidden_size,
            config.predictor_layers,
            batch_first=True,
            dropout=config.dropout if config.predictor_layers > 1 else 0.0,
        )
        self.encoder_projection = nn.Linear(config.encoder_hidden_size * 2, config.joiner_hidden_size)
        self.predictor_projection = nn.Linear(config.predictor_hidden_size, config.joiner_hidden_size)
        self.joiner = nn.Sequential(nn.Tanh(), nn.Linear(config.joiner_hidden_size, self.vocab_size))

    def encode(self, input_ids: Tensor, attention_mask: Tensor, language_ids: Tensor) -> Tensor:
        embeddings = self.bert.embeddings.word_embeddings(input_ids)
        embeddings = embeddings + self.language_embedding(language_ids[:, None])
        embeddings = embeddings + self.modality_embedding(torch.ones_like(input_ids))
        contextual = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask).last_hidden_state
        encoded, _ = self.encoder(contextual)
        return self.encoder_projection(encoded)

    def predict(self, tokens: Tensor) -> Tensor:
        values, _ = self.predictor(self.predictor_embedding(tokens))
        return self.predictor_projection(values)

    def predict_step(
        self,
        token: int,
        state: tuple[Tensor, Tensor] | None,
        device: torch.device,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        value = torch.tensor([[token]], device=device)
        output, next_state = self.predictor(self.predictor_embedding(value), state)
        return self.predictor_projection(output[:, -1])[0], next_state

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        language_ids: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> RnntOutput:
        encoded = self.encode(input_ids, attention_mask, language_ids)
        starts = torch.full((targets.shape[0], 1), self.blank_id, device=targets.device, dtype=torch.long)
        predicted = self.predict(torch.cat((starts, targets), dim=1))
        logits = self.joiner(encoded[:, :, None, :] + predicted[:, None, :, :])
        loss = rnnt_loss(
            logits.float(),
            targets.to(torch.int32),
            attention_mask.sum(1).to(torch.int32),
            target_lengths.to(torch.int32),
            blank=self.blank_id,
            clamp=1.0,
        )
        return RnntOutput(logits, loss)

    @torch.no_grad()
    def greedy_decode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        language_ids: Tensor,
        max_symbols_per_timestep: int,
    ) -> list[list[int]]:
        encoded = self.encode(input_ids, attention_mask, language_ids)
        results: list[list[int]] = []
        for batch_index, length in enumerate(attention_mask.sum(1).tolist()):
            tokens = [self.blank_id]
            for timestep in range(length):
                for _ in range(max_symbols_per_timestep):
                    predictor = self.predict(torch.tensor(tokens, device=input_ids.device)[None, :])[:, -1]
                    logits = self.joiner(encoded[batch_index, timestep] + predictor[0])
                    token = int(logits.argmax())
                    if token == self.blank_id:
                        break
                    tokens.append(token)
            results.append(tokens[1:])
        return results

    @torch.no_grad()
    def beam_decode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        language_ids: Tensor,
        beam_width: int,
        max_symbols_per_timestep: int,
    ) -> list[list[int]]:
        encoded = self.encode(input_ids, attention_mask, language_ids)
        results: list[list[int]] = []
        for batch_index, length in enumerate(attention_mask.sum(1).tolist()):
            prediction, state = self.predict_step(self.blank_id, None, input_ids.device)
            beams = [BeamHypothesis((), 0.0, state, prediction)]
            for timestep in range(length):
                completed: list[BeamHypothesis] = []
                active = beams
                for _ in range(max_symbols_per_timestep):
                    predictions = torch.stack([hypothesis.prediction for hypothesis in active])
                    logits = self.joiner(encoded[batch_index, timestep][None, :] + predictions)
                    log_probabilities = torch.log_softmax(logits, dim=-1)
                    values, indices = log_probabilities[:, :-1].topk(beam_width, dim=-1)
                    for hypothesis, blank_score in zip(
                        active, log_probabilities[:, self.blank_id].tolist(), strict=True
                    ):
                        completed.append(
                            BeamHypothesis(
                                hypothesis.tokens,
                                hypothesis.score + blank_score,
                                hypothesis.state,
                                hypothesis.prediction,
                            )
                        )
                    parent_indices = torch.arange(len(active), device=input_ids.device).repeat_interleave(beam_width)
                    tokens = indices.flatten()
                    hidden = torch.cat([hypothesis.state[0] for hypothesis in active], dim=1)
                    cells = torch.cat([hypothesis.state[1] for hypothesis in active], dim=1)
                    state = (
                        hidden.index_select(1, parent_indices),
                        cells.index_select(1, parent_indices),
                    )
                    output, next_state = self.predictor(self.predictor_embedding(tokens[:, None]), state)
                    next_predictions = self.predictor_projection(output[:, -1])
                    expanded = [
                        BeamHypothesis(
                            active[parent].tokens + (token,),
                            active[parent].score + value,
                            (next_state[0][:, candidate : candidate + 1], next_state[1][:, candidate : candidate + 1]),
                            next_predictions[candidate],
                        )
                        for candidate, (parent, token, value) in enumerate(
                            zip(parent_indices.tolist(), tokens.tolist(), values.flatten().tolist(), strict=True)
                        )
                    ]
                    active = sorted(expanded, key=lambda item: item.score, reverse=True)[:beam_width]
                    best_completed = max(item.score for item in completed)
                    if best_completed >= active[0].score:
                        break
                beams = sorted(completed, key=lambda item: item.score, reverse=True)[:beam_width]
            results.append(list(beams[0].tokens))
        return results


def _bert_geometry(state: dict[str, Tensor]) -> dict[str, int | float | str]:
    layer_ids = {int(key.split(".")[2]) for key in state if key.startswith("encoder.layer.")}
    hidden = int(state["embeddings.word_embeddings.weight"].shape[1])
    return {
        "hidden_size": hidden,
        "num_hidden_layers": max(layer_ids) + 1,
        "num_attention_heads": 12,
        "intermediate_size": int(state["encoder.layer.0.intermediate.dense.weight"].shape[0]),
        "max_position_embeddings": int(state["embeddings.position_embeddings.weight"].shape[0]),
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "type_vocab_size": 1,
        "attn_implementation": "sdpa",
    }
