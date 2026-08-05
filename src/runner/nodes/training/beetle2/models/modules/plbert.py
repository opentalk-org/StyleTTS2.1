import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor, nn
from transformers import BertConfig, BertModel

from ...phonemes import PHONEME_EXTENSIONS


class PlBertEncoder(nn.Module):
    def __init__(
        self,
        root: Path,
        languages: tuple[str, ...],
        symbols: tuple[str, ...],
    ) -> None:
        super().__init__()
        manifest = json.loads(
            (root / "checkpoint_193088" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        geometry = manifest["model_geometry"]
        config = BertConfig(
            vocab_size=len(symbols),
            hidden_size=geometry["hidden_size"],
            num_hidden_layers=geometry["layers"],
            num_attention_heads=geometry["heads"],
            intermediate_size=geometry["intermediate_size"],
            max_position_embeddings=geometry["max_positions"],
            hidden_dropout_prob=geometry["dropout"],
            attention_probs_dropout_prob=geometry["dropout"],
            type_vocab_size=1,
            pad_token_id=0,
            attn_implementation="sdpa",
        )
        self.encoder = BertModel(config)
        self.language_embeddings = nn.Embedding(
            geometry["num_languages"],
            geometry["hidden_size"],
            padding_idx=0,
        )
        self.modality_embeddings = nn.Embedding(2, geometry["hidden_size"])
        self.symbol_ids = {
            symbol: index for index, symbol in enumerate(symbols)
        }
        language_payload = json.loads(
            (root / "languages.json").read_text(encoding="utf-8")
        )
        language_names = tuple(language_payload["languages"])
        language_ids = tuple(language_names.index(language) + 1 for language in languages)
        self.register_buffer(
            "language_ids",
            torch.tensor(language_ids, dtype=torch.long),
            persistent=False,
        )
        self._load_checkpoint(
            root / "checkpoint_193088" / "accelerate" / "model.safetensors"
        )

    @property
    def output_channels(self) -> int:
        return int(self.encoder.config.hidden_size)

    def _load_checkpoint(self, path: Path) -> None:
        state: dict[str, Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                normalized = key.replace("encoder._orig_mod.", "encoder.")
                if normalized.startswith(
                    ("encoder.", "language_embeddings.", "modality_embeddings.")
                ):
                    value = checkpoint.get_tensor(key)
                    if normalized == "encoder.embeddings.word_embeddings.weight":
                        value = self._resized_word_embeddings(value)
                    state[normalized] = value
        self.load_state_dict(state, strict=True)

    def _resized_word_embeddings(self, checkpoint: Tensor) -> Tensor:
        resized = self.encoder.embeddings.word_embeddings.weight.detach().clone()
        resized[: checkpoint.shape[0]].copy_(checkpoint)
        for symbol, sources in PHONEME_EXTENSIONS.items():
            target_id = self.symbol_ids[symbol]
            source_ids = [self.symbol_ids[source] for source in sources]
            resized[target_id].copy_(resized[source_ids].mean(dim=0))
        return resized

    def forward(
        self,
        input_ids: Tensor,
        mask: Tensor,
        languages: Tensor,
        modality: int = 0,
    ) -> Tensor:
        language_ids = self.language_ids.index_select(0, languages)
        language_ids = language_ids.unsqueeze(1).expand_as(input_ids)
        modality_ids = torch.full_like(input_ids, modality)
        tokens = self.encoder.embeddings.word_embeddings(input_ids)
        tokens = tokens + self.language_embeddings(language_ids)
        tokens = tokens + self.modality_embeddings(modality_ids)
        return self.encoder(
            inputs_embeds=tokens,
            attention_mask=mask,
        ).last_hidden_state
