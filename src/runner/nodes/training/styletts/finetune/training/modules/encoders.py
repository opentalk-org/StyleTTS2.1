from torch import nn
from torch.nn.utils import weight_norm
from runner.nodes.training.styletts.finetune.training.modules.normalizations import (
    LayerNorm,
)


class TextEncoder(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        depth,
        n_symbols,
        actv=nn.LeakyReLU(0.2),
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)

        for convolution in self.cnn:
            x = convolution(x)
            x.masked_fill_(m, 0.0)

        x = x.transpose(1, 2)
        input_lengths = input_lengths.numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = x.transpose(-1, -2)
        padded = x.new_zeros([x.shape[0], x.shape[1], m.shape[-1]])
        padded[:, :, :x.shape[-1]] = x
        return padded.masked_fill(m, 0.0)
