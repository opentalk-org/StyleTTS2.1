from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from transformers import BertConfig, BertLMHeadModel, BertModel

from .assets import BertAsset


@dataclass(frozen=True)
class G2POutput:
    logits: Tensor
    loss: Tensor | None


class BertG2P(nn.Module):
    def __init__(self, asset: BertAsset) -> None:
        super().__init__()
        geometry = _geometry(asset.encoder_state)
        encoder_config = BertConfig(
            **geometry,
            vocab_size=len(asset.symbols),
            is_decoder=False,
            pad_token_id=0,
        )
        decoder_config = BertConfig(
            **geometry,
            vocab_size=len(asset.symbols),
            is_decoder=True,
            add_cross_attention=True,
            bos_token_id=len(asset.symbols),
            eos_token_id=len(asset.symbols) + 1,
            pad_token_id=0,
            tie_word_embeddings=False,
        )
        self.encoder = BertModel(encoder_config)
        self.decoder = BertLMHeadModel(decoder_config)
        self.encoder.load_state_dict(asset.encoder_state, strict=True)
        missing, unexpected = self.decoder.bert.load_state_dict(asset.encoder_state, strict=False)
        unexpected_other = [key for key in unexpected if not key.startswith("pooler.")]
        if unexpected_other or any("crossattention" not in key for key in missing):
            raise ValueError(f"decoder initialization mismatch: missing={missing}, unexpected={unexpected}")
        self.encoder.resize_token_embeddings(len(asset.symbols) + 2)
        self.decoder.resize_token_embeddings(len(asset.symbols) + 2)
        with torch.no_grad():
            self.decoder.cls.predictions.decoder.weight[: len(asset.symbols)].copy_(asset.phoneme_head_weight)
            self.decoder.cls.predictions.decoder.bias[: len(asset.symbols)].copy_(asset.phoneme_head_bias)
        self.encoder_language = nn.Embedding.from_pretrained(asset.language_weights.clone(), freeze=False, padding_idx=0)
        self.decoder_language = nn.Embedding.from_pretrained(asset.language_weights.clone(), freeze=False, padding_idx=0)
        self.encoder_modality = nn.Embedding.from_pretrained(asset.modality_weights.clone(), freeze=False)
        self.decoder_modality = nn.Embedding.from_pretrained(asset.modality_weights.clone(), freeze=False)
        self.vocab_size = len(asset.symbols) + 2
        self.bos_id = len(asset.symbols)
        self.eos_id = len(asset.symbols) + 1
        self.pad_id = 0
        allowed_tokens = torch.zeros(self.vocab_size, dtype=torch.bool)
        allowed_tokens[259 : len(asset.symbols)] = True
        allowed_tokens[self.eos_id] = True
        self.register_buffer("generation_tokens", allowed_tokens, persistent=False)

    def encode(self, input_ids: Tensor, attention_mask: Tensor, language_ids: Tensor) -> Tensor:
        inputs = self.encoder.embeddings.word_embeddings(input_ids)
        inputs = inputs + self.encoder_language(language_ids[:, None])
        inputs = inputs + self.encoder_modality(torch.ones_like(input_ids))
        return self.encoder(inputs_embeds=inputs, attention_mask=attention_mask).last_hidden_state

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        decoder_input_ids: Tensor,
        language_ids: Tensor,
        labels: Tensor | None = None,
    ) -> G2POutput:
        memory = self.encode(input_ids, attention_mask, language_ids)
        decoder_embeddings = self.decoder.bert.embeddings.word_embeddings(decoder_input_ids)
        decoder_embeddings = decoder_embeddings + self.decoder_language(language_ids[:, None])
        decoder_embeddings = decoder_embeddings + self.decoder_modality(torch.zeros_like(decoder_input_ids))
        output = self.decoder(
            inputs_embeds=decoder_embeddings,
            encoder_hidden_states=memory,
            encoder_attention_mask=attention_mask,
        )
        loss = None
        if labels is not None:
            loss = functional.cross_entropy(
                output.logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return G2POutput(output.logits, loss)

    def mask_generation_logits(self, logits: Tensor) -> Tensor:
        return logits.masked_fill(~self.generation_tokens, torch.finfo(logits.dtype).min)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        language_ids: Tensor,
        max_tokens: int,
        sample: bool = False,
    ) -> Tensor:
        memory = self.encode(input_ids, attention_mask, language_ids)
        generated = torch.full((input_ids.shape[0], 1), self.bos_id, dtype=torch.long, device=input_ids.device)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        past_key_values = None
        for step in range(max_tokens):
            decoder_ids = generated if past_key_values is None else generated[:, -1:]
            embeddings = self.decoder.bert.embeddings.word_embeddings(decoder_ids)
            embeddings = embeddings + self.decoder_language(language_ids[:, None])
            embeddings = embeddings + self.decoder_modality(torch.zeros_like(decoder_ids))
            output = self.decoder(
                inputs_embeds=embeddings,
                encoder_hidden_states=memory,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            logits = self.mask_generation_logits(output.logits[:, -1])
            allowed_logits = logits[:, self.generation_tokens]
            if step == 0 and not torch.isfinite(allowed_logits).all():
                output = self.decoder(
                    inputs_embeds=embeddings,
                    encoder_hidden_states=memory,
                    encoder_attention_mask=attention_mask,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
                logits = self.mask_generation_logits(output.logits[:, -1])
                allowed_logits = logits[:, self.generation_tokens]
            if not torch.isfinite(allowed_logits).all():
                memory_bad = int((~torch.isfinite(memory)).sum())
                embedding_bad = int((~torch.isfinite(embeddings)).sum())
                raise RuntimeError(
                    f"decoder produced non-finite phoneme logits at generation step {step + 1}; "
                    f"memory_nonfinite={memory_bad}, embedding_nonfinite={embedding_bad}"
                )
            if generated.shape[1] == 1:
                logits[:, self.eos_id] = torch.finfo(logits.dtype).min
            token = torch.distributions.Categorical(logits=logits).sample() if sample else logits.argmax(-1)
            token = torch.where(finished, torch.full_like(token, self.pad_id), token)
            generated = torch.cat((generated, token[:, None]), dim=1)
            finished |= token.eq(self.eos_id)
            if torch.all(finished):
                break
        return generated


def _geometry(state: dict[str, Tensor]) -> dict[str, int | float | str]:
    layer_ids = {int(key.split(".")[2]) for key in state if key.startswith("encoder.layer.")}
    hidden = int(state["embeddings.word_embeddings.weight"].shape[1])
    intermediate = int(state["encoder.layer.0.intermediate.dense.weight"].shape[0])
    positions = int(state["embeddings.position_embeddings.weight"].shape[0])
    return {
        "hidden_size": hidden,
        "num_hidden_layers": max(layer_ids) + 1,
        "num_attention_heads": 12,
        "intermediate_size": intermediate,
        "max_position_embeddings": positions,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "type_vocab_size": 1,
        "attn_implementation": "sdpa",
    }
