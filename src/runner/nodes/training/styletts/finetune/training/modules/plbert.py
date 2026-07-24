import os
import yaml
from collections import OrderedDict

import torch
from transformers import AlbertConfig, AlbertModel
from runner.nodes.training.styletts.finetune.training.state_dict_resize import merge_state_dict_with_dim0_resize

_PLBERT_VOCAB_DIM0_KEYS = frozenset({
    "embeddings.word_embeddings.weight",
})
class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state


def load_plbert(path, config):
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
    )
    bert.load_state_dict(merged, strict=True)

    return bert
