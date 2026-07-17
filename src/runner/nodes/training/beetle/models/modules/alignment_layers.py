import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchaudio.functional import create_dct


class LinearNorm(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        bias: bool = True,
        gain: str = "linear",
    ) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(input_channels, output_channels, bias=bias)
        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(gain),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.linear_layer(values)


class ConvNorm(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        bias: bool = True,
        gain: str = "linear",
    ) -> None:
        super().__init__()
        resolved_padding = (
            dilation * (kernel_size - 1) // 2 if padding is None else padding
        )
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            stride=stride,
            padding=resolved_padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(gain))

    def forward(self, values: Tensor) -> Tensor:
        return self.conv(values)


class ConvBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        convolution_count: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNorm(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=3,
                        padding=3**index,
                        dilation=3**index,
                    ),
                    nn.ReLU(),
                    nn.GroupNorm(8, hidden_channels),
                    nn.Dropout(dropout),
                    ConvNorm(hidden_channels, hidden_channels, kernel_size=3),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for index in range(convolution_count)
            ]
        )

    def forward(self, values: Tensor) -> Tensor:
        for block in self.blocks:
            values = values + block(values)
        return values


class LocationLayer(nn.Module):
    def __init__(
        self,
        filter_count: int,
        kernel_size: int,
        attention_channels: int,
    ) -> None:
        super().__init__()
        self.location_conv = ConvNorm(
            2,
            filter_count,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.location_dense = LinearNorm(
            filter_count,
            attention_channels,
            bias=False,
            gain="tanh",
        )

    def forward(self, weights: Tensor) -> Tensor:
        return self.location_dense(self.location_conv(weights).transpose(1, 2))


class Attention(nn.Module):
    def __init__(
        self,
        recurrent_channels: int,
        embedding_channels: int,
        attention_channels: int,
        location_filter_count: int,
        location_kernel_size: int,
    ) -> None:
        super().__init__()
        self.query_layer = LinearNorm(
            recurrent_channels,
            attention_channels,
            bias=False,
            gain="tanh",
        )
        self.memory_layer = LinearNorm(
            embedding_channels,
            attention_channels,
            bias=False,
            gain="tanh",
        )
        self.v = LinearNorm(attention_channels, 1, bias=False)
        self.location_layer = LocationLayer(
            location_filter_count,
            location_kernel_size,
            attention_channels,
        )
        self.score_mask_value = -float("inf")

    def forward(
        self,
        hidden: Tensor,
        memory: Tensor,
        processed_memory: Tensor,
        cumulative_weights: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        query = self.query_layer(hidden.unsqueeze(1))
        location = self.location_layer(cumulative_weights)
        energies = self.v(torch.tanh(query + location + processed_memory)).squeeze(-1)
        if mask is not None:
            energies = energies.masked_fill(mask, self.score_mask_value)
        weights = F.softmax(energies, dim=1)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        return context, weights


class MelCepstrum(nn.Module):
    def __init__(self, coefficient_count: int = 40, mel_channels: int = 80) -> None:
        super().__init__()
        matrix = create_dct(coefficient_count, mel_channels, "ortho")
        self.register_buffer("dct_mat", matrix)

    def forward(self, mel: Tensor) -> Tensor:
        return torch.matmul(mel.transpose(1, 2), self.dct_mat).transpose(1, 2)
