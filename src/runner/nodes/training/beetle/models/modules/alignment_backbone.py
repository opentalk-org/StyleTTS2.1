import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .alignment_layers import Attention, ConvBlock, ConvNorm, LinearNorm, MelCepstrum


class SequenceAligner(nn.Module):
    def __init__(
        self,
        embedding_channels: int,
        hidden_channels: int,
        token_count: int,
        location_filter_count: int = 32,
        location_kernel_size: int = 63,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(token_count, embedding_channels)
        limit = math.sqrt(6 / hidden_channels)
        self.embedding.weight.data.uniform_(-limit, limit)
        self.decoder_rnn_dim = hidden_channels
        self.project_to_n_symbols = nn.Linear(hidden_channels, token_count)
        self.attention_layer = Attention(
            hidden_channels,
            hidden_channels,
            hidden_channels,
            location_filter_count,
            location_kernel_size,
        )
        self.decoder_rnn = nn.LSTMCell(
            hidden_channels + embedding_channels,
            hidden_channels,
        )
        self.project_to_hidden = nn.Sequential(
            LinearNorm(hidden_channels * 2, hidden_channels),
            nn.Tanh(),
        )
        self.sos = 1
        self.eos = 2

    def forward(
        self,
        memory: Tensor,
        memory_mask: Tensor | None,
        text_input: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self._initialize(memory, memory_mask)
        random_mask = torch.rand(text_input.shape, device=text_input.device) < 0.1
        decoder_ids = text_input.clone().masked_fill_(random_mask, 3)
        inputs = self.embedding(decoder_ids).transpose(0, 1)
        start_ids = torch.full(
            (inputs.shape[1],),
            self.sos,
            dtype=torch.long,
            device=inputs.device,
        )
        inputs = torch.cat((self.embedding(start_ids).unsqueeze(0), inputs), dim=0)
        hidden_outputs = []
        logit_outputs = []
        alignments = []
        for decoder_input in inputs:
            hidden, logits, weights = self._decode(decoder_input)
            hidden_outputs.append(hidden)
            logit_outputs.append(logits)
            alignments.append(weights)
        hidden = torch.stack(hidden_outputs).transpose(0, 1).contiguous()
        logits = torch.stack(logit_outputs).transpose(0, 1).contiguous()
        attention = torch.stack(alignments).transpose(0, 1)
        return hidden, logits, attention

    def _initialize(self, memory: Tensor, mask: Tensor | None) -> None:
        batch, frames, channels = memory.shape
        self.decoder_hidden = memory.new_zeros(batch, self.decoder_rnn_dim)
        self.decoder_cell = memory.new_zeros(batch, self.decoder_rnn_dim)
        self.attention_weights = memory.new_zeros(batch, frames)
        self.attention_weights_cum = memory.new_zeros(batch, frames)
        self.attention_context = memory.new_zeros(batch, channels)
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def _decode(self, decoder_input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        cell_input = torch.cat((decoder_input, self.attention_context), dim=-1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            cell_input,
            (self.decoder_hidden, self.decoder_cell),
        )
        cumulative = torch.cat(
            (
                self.attention_weights.unsqueeze(1),
                self.attention_weights_cum.unsqueeze(1),
            ),
            dim=1,
        )
        self.attention_context, self.attention_weights = self.attention_layer(
            self.decoder_hidden,
            self.memory,
            self.processed_memory,
            cumulative,
            self.mask,
        )
        self.attention_weights_cum = (
            self.attention_weights_cum + self.attention_weights
        )
        hidden = self.project_to_hidden(
            torch.cat((self.decoder_hidden, self.attention_context), dim=-1)
        )
        logits = self.project_to_n_symbols(F.dropout(hidden, 0.5, self.training))
        return hidden, logits, self.attention_weights


class StyleTTSAlignerBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        token_count: int,
        layer_count: int,
        token_embedding_channels: int,
    ) -> None:
        super().__init__()
        self.n_token = token_count
        self.n_down = 1
        self.to_mfcc = MelCepstrum()
        self.init_cnn = ConvNorm(
            input_channels // 2,
            hidden_channels,
            kernel_size=7,
            padding=3,
            stride=2,
        )
        self.cnns = nn.Sequential(
            *[
                nn.Sequential(
                    ConvBlock(hidden_channels),
                    nn.GroupNorm(1, hidden_channels),
                )
                for _ in range(layer_count)
            ]
        )
        self.projection = ConvNorm(hidden_channels, hidden_channels // 2)
        self.ctc_linear = nn.Sequential(
            LinearNorm(hidden_channels // 2, hidden_channels),
            nn.ReLU(),
            LinearNorm(hidden_channels, token_count),
        )
        self.asr_s2s = SequenceAligner(
            token_embedding_channels,
            hidden_channels // 2,
            token_count,
        )

    def forward(
        self,
        mel: Tensor,
        padding_mask: Tensor | None = None,
        text_input: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        features = self.to_mfcc(mel)
        features = self.init_cnn(features)
        features = self.cnns(features)
        features = self.projection(features).transpose(1, 2)
        ctc_logits = self.ctc_linear(features)
        if text_input is None:
            return ctc_logits
        _, sequence_logits, attention = self.asr_s2s(
            features,
            padding_mask,
            text_input,
        )
        return ctc_logits, sequence_logits, attention
