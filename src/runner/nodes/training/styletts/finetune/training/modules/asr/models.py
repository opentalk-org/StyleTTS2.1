import math
import torch
from torch import nn
import torch.nn.functional as F

from ...profiling import profiling_fn
from .layers import MFCC, Attention, LinearNorm, ConvNorm, ConvBlock


class CheckpointCompatibleLSTM(nn.LSTM):
    """cuDNN LSTM with the parameter names used by the original LSTMCell."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__(input_size, hidden_size)
        names = ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
        for current, checkpoint in zip(tuple(self._flat_weights_names), names):
            self.register_parameter(checkpoint, self._parameters.pop(current))
        self._flat_weights_names = list(names)
        self._init_flat_weights()


class ASRCNN(nn.Module):
    def __init__(
        self,
        input_dim=80,
        hidden_dim=256,
        n_token=35,
        n_layers=6,
        token_embedding_dim=256,
    ):
        super().__init__()
        self.n_token = n_token
        self.n_down = 1
        self.to_mfcc = MFCC()
        self.init_cnn = ConvNorm(
            input_dim // 2, hidden_dim, kernel_size=7, padding=3, stride=2
        )
        self.cnns = nn.Sequential(
            *[
                nn.Sequential(
                    ConvBlock(hidden_dim),
                    nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
                )
                for n in range(n_layers)
            ]
        )
        self.projection = ConvNorm(hidden_dim, hidden_dim // 2)
        self.ctc_linear = nn.Sequential(
            LinearNorm(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            LinearNorm(hidden_dim, n_token),
        )
        self.asr_s2s = ASRS2S(
            embedding_dim=token_embedding_dim,
            hidden_dim=hidden_dim // 2,
            n_token=n_token,
        )

    def forward(self, x, src_key_padding_mask=None, text_input=None):
        with profiling_fn("mfcc_projection"):
            x = self.to_mfcc(x)
        with profiling_fn("initial_convolution"):
            x = self.init_cnn(x)
        with profiling_fn("convolution_stack"):
            for index, block in enumerate(self.cnns):
                with profiling_fn(f"convolution_block_{index}"):
                    x = block(x)
        with profiling_fn("feature_projection"):
            x = self.projection(x)
        x = x.transpose(1, 2)
        with profiling_fn("ctc_projection"):
            ctc_logit = self.ctc_linear(x)
        if text_input is not None:
            with profiling_fn("sequence_to_sequence"):
                _, s2s_logit, s2s_attn = self.asr_s2s(
                    x,
                    src_key_padding_mask,
                    text_input,
                )
            return ctc_logit, s2s_logit, s2s_attn
        else:
            return ctc_logit

    def get_feature(self, x):
        x = self.to_mfcc(x.squeeze(1))
        x = self.init_cnn(x)
        x = self.cnns(x)
        x = self.projection(x)
        return x

    def length_to_mask(self, lengths):
        mask = (
            torch.arange(lengths.max())
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths)
        )
        mask = torch.gt(mask + 1, lengths.unsqueeze(1)).to(lengths.device)
        return mask

    def get_future_mask(self, out_length, unmask_future_steps=0):
        index_tensor = torch.arange(out_length).unsqueeze(0).expand(out_length, -1)
        mask = torch.gt(index_tensor, index_tensor.T + unmask_future_steps)
        return mask


class ASRS2S(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        hidden_dim=512,
        n_location_filters=32,
        location_kernel_size=63,
        n_token=40,
    ):
        super(ASRS2S, self).__init__()
        self.embedding = nn.Embedding(n_token, embedding_dim)
        val_range = math.sqrt(6 / hidden_dim)
        self.embedding.weight.data.uniform_(-val_range, val_range)

        self.decoder_rnn_dim = hidden_dim
        self.project_to_n_symbols = nn.Linear(self.decoder_rnn_dim, n_token)
        self.attention_layer = Attention(
            self.decoder_rnn_dim,
            hidden_dim,
            hidden_dim,
            n_location_filters,
            location_kernel_size,
        )
        self.decoder_rnn = CheckpointCompatibleLSTM(
            self.decoder_rnn_dim + embedding_dim, self.decoder_rnn_dim
        )
        self.project_to_hidden = nn.Sequential(
            LinearNorm(self.decoder_rnn_dim * 2, hidden_dim), nn.Tanh()
        )
        self.sos = 1
        self.eos = 2
        self.unk_index = 3
        self.random_mask = 0.1

    def forward(self, memory, memory_mask, text_input):
        batch_size, memory_steps, memory_dim = memory.shape
        processed_memory = self.attention_layer.memory_layer(memory)
        random_mask = (
            torch.rand(
                text_input.shape,
                device=text_input.device,
            )
            < self.random_mask
        )
        _text_input = text_input.clone()
        _text_input.masked_fill_(random_mask, self.unk_index)
        decoder_inputs = self.embedding(_text_input).transpose(0, 1)
        start_embedding = self.embedding(
            torch.full(
                (decoder_inputs.size(1),),
                self.sos,
                dtype=torch.long,
                device=decoder_inputs.device,
            )
        )
        decoder_inputs = torch.cat(
            (start_embedding.unsqueeze(0), decoder_inputs),
            dim=0,
        )
        zero_context = memory.new_zeros(
            (decoder_inputs.size(0), batch_size, memory_dim)
        )
        recurrent_inputs = torch.cat((decoder_inputs, zero_context), dim=-1)
        initial_hidden = memory.new_zeros(
            (1, batch_size, self.decoder_rnn_dim)
        )
        initial_cell = torch.zeros_like(initial_hidden)
        recurrent, _ = self.decoder_rnn(
            recurrent_inputs,
            (initial_hidden, initial_cell),
        )
        recurrent = recurrent.transpose(0, 1)
        queries = self.attention_layer.query_layer(recurrent)
        scale = queries.size(-1) ** -0.5
        scores = torch.bmm(
            queries,
            processed_memory.transpose(1, 2),
        ) * scale
        if memory_mask is not None:
            scores = scores.masked_fill(memory_mask.unsqueeze(1), -float("inf"))
        alignment = F.softmax(scores, dim=-1)
        attention_context = torch.bmm(alignment, memory)
        hidden = self.project_to_hidden(
            torch.cat((recurrent, attention_context), dim=-1)
        )
        logit = self.project_to_n_symbols(
            F.dropout(hidden, 0.5, self.training)
        )
        return hidden, logit, alignment
