from dataclasses import dataclass

from torch import Tensor, nn
from transformers import BertModel


@dataclass(frozen=True)
class PromptEncoding:
    style: Tensor
    voice: Tensor


class TextEncoder(nn.Module):
    def __init__(self, bert: BertModel, output_channels: int) -> None:
        super().__init__()
        self.bert = bert
        self.style_projection = nn.Linear(bert.config.hidden_size, output_channels)
        self.voice_projection = nn.Linear(bert.config.hidden_size, output_channels)

    def forward(self, input_ids: Tensor, mask: Tensor) -> PromptEncoding:
        if input_ids.shape != mask.shape:
            raise ValueError("text ids and mask must have equal shapes")
        tokens = self.bert(
            input_ids=input_ids,
            attention_mask=mask,
            return_dict=True,
        ).last_hidden_state
        numeric_mask = mask.unsqueeze(2).to(dtype=tokens.dtype)
        pooled = (tokens * numeric_mask).sum(dim=1)
        pooled = pooled / numeric_mask.sum(dim=1).clamp_min(1)
        return PromptEncoding(
            style=self.style_projection(pooled),
            voice=self.voice_projection(pooled),
        )
