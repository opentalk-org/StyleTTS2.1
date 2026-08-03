from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor, nn
from transformers import AlbertConfig, AlbertModel, BertConfig, BertModel

from runner.nodes.text.runtime.symbols import END_SYMBOL, START_SYMBOL
from runner.nodes.training.styletts.finetune.training.state_dict_resize import merge_state_dict_with_dim0_resize

_PLBERT_VOCAB_DIM0_KEYS = frozenset({
    "embeddings.word_embeddings.weight",
})
_PHONEME_EXTENSIONS = {
    "ɚ": ("ə", "ɹ"),
    "ᵻ": ("ɨ",),
}


class CustomAlbert(AlbertModel):
    def forward(self, *args, language_ids=None, modality_ids=None, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state


class CustomBert(BertModel):
    def forward(self, *args, language_ids=None, modality_ids=None, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state


class MultilingualPlBert(nn.Module):
    def __init__(
        self,
        encoder: CustomBert,
        language_embeddings: nn.Embedding,
        modality_embeddings: nn.Embedding,
        token_id_map: Tensor,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config
        self.language_embeddings = language_embeddings
        self.modality_embeddings = modality_embeddings
        self.register_buffer("token_id_map", token_id_map, persistent=True)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        language_ids: Tensor,
        modality_ids: Tensor,
    ) -> Tensor:
        artifact_ids = self.token_id_map[input_ids]
        language_ids = _expand_condition_ids(
            language_ids,
            artifact_ids,
            "language",
        )
        modality_ids = _expand_condition_ids(
            modality_ids,
            artifact_ids,
            "modality",
        )
        embeddings = self.encoder.embeddings.word_embeddings(artifact_ids)
        embeddings = embeddings + self.language_embeddings(language_ids)
        embeddings = embeddings + self.modality_embeddings(modality_ids)
        return self.encoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )


def load_plbert(path, config):
    if path is not None and Path(path).suffix == ".safetensors":
        return _load_multilingual_plbert(Path(path), config)

    albert_base_configuration = AlbertConfig(**config['model_params'])
    bert = CustomAlbert(albert_base_configuration)

    if path is None:
        print("No PLBERT path found, using default PLBERT from checkpoint")
        return bert

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["net"]
    stripped = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        if name.startswith("encoder."):
            name = name[8:]
            stripped[name] = v
    stripped.pop("embeddings.position_ids", None)
    merged = merge_state_dict_with_dim0_resize(
        bert,
        stripped,
        _PLBERT_VOCAB_DIM0_KEYS,
        error_scope="PLBERT",
        appended_source_index=0,
    )
    bert.load_state_dict(merged, strict=True)

    return bert


def _load_multilingual_plbert(path: Path, config: dict) -> MultilingualPlBert:
    state_dict: dict[str, torch.Tensor] = {}
    language_weights: Tensor | None = None
    modality_weights: Tensor | None = None
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            normalized = key.removeprefix("encoder._orig_mod.")
            if key.startswith("encoder._orig_mod."):
                state_dict[normalized] = checkpoint.get_tensor(key)
            elif key == "language_embeddings.weight":
                language_weights = checkpoint.get_tensor(key)
            elif key == "modality_embeddings.weight":
                modality_weights = checkpoint.get_tensor(key)

    word_embeddings = state_dict["embeddings.word_embeddings.weight"]
    position_embeddings = state_dict["embeddings.position_embeddings.weight"]
    layer_ids = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("encoder.layer.")
    }
    model_params = config["model_params"]
    artifact_symbols = tuple(config["artifact_symbols"])
    input_symbols = tuple(config["input_symbols"])
    appended_symbols = tuple(
        symbol
        for symbol in input_symbols
        if symbol != "$" and symbol not in artifact_symbols
    )
    all_symbols = artifact_symbols + appended_symbols
    bert = CustomBert(
        BertConfig(
            vocab_size=len(all_symbols),
            hidden_size=word_embeddings.shape[1],
            num_hidden_layers=max(layer_ids) + 1,
            num_attention_heads=int(model_params["num_attention_heads"]),
            intermediate_size=state_dict["encoder.layer.0.intermediate.dense.weight"].shape[0],
            max_position_embeddings=position_embeddings.shape[0],
            hidden_dropout_prob=float(model_params["dropout"]),
            attention_probs_dropout_prob=float(model_params["dropout"]),
            type_vocab_size=state_dict["embeddings.token_type_embeddings.weight"].shape[0],
            pad_token_id=0,
            attn_implementation="sdpa",
        )
    )
    extended_embeddings = bert.embeddings.word_embeddings.weight.detach().clone()
    extended_embeddings[: word_embeddings.shape[0]].copy_(word_embeddings)
    symbol_ids = {symbol: index for index, symbol in enumerate(all_symbols)}
    for symbol in (START_SYMBOL, END_SYMBOL):
        extended_embeddings[symbol_ids[symbol]].copy_(
            extended_embeddings[symbol_ids["[PAD]"]]
        )
    for symbol, sources in _PHONEME_EXTENSIONS.items():
        if symbol in appended_symbols:
            source_ids = [symbol_ids[source] for source in sources]
            extended_embeddings[symbol_ids[symbol]].copy_(
                extended_embeddings[source_ids].mean(dim=0)
            )
    state_dict["embeddings.word_embeddings.weight"] = extended_embeddings
    bert.load_state_dict(state_dict, strict=True)
    assert language_weights is not None
    assert modality_weights is not None
    language_embeddings = nn.Embedding.from_pretrained(
        language_weights,
        freeze=False,
        padding_idx=0,
    )
    modality_embeddings = nn.Embedding.from_pretrained(
        modality_weights,
        freeze=False,
    )
    token_ids = [
        symbol_ids["[PAD]"] if symbol == "$" else symbol_ids[symbol]
        for symbol in input_symbols
    ]
    return MultilingualPlBert(
        bert,
        language_embeddings,
        modality_embeddings,
        torch.tensor(token_ids, dtype=torch.long),
    )


def _expand_condition_ids(
    condition_ids: Tensor,
    input_ids: Tensor,
    name: str,
) -> Tensor:
    if condition_ids.ndim == 1:
        if condition_ids.shape[0] != input_ids.shape[0]:
            raise ValueError(
                f"PLBERT {name} batch does not match input batch"
            )
        return condition_ids.unsqueeze(1).expand_as(input_ids)
    if condition_ids.shape != input_ids.shape:
        raise ValueError(
            f"PLBERT {name} IDs must be per sample or per token"
        )
    return condition_ids
