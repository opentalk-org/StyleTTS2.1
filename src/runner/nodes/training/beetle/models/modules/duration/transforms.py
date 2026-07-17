import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .spline import unconstrained_rational_quadratic_spline


def _validate_transform_input(values: Tensor, mask: Tensor) -> Tensor:
    if values.ndim != 3 or mask.shape != (values.shape[0], 1, values.shape[2]):
        raise ValueError("flow transforms require [B,C,T] values and [B,1,T] mask")
    return mask.to(dtype=values.dtype)


class LogTransform(nn.Module):
    def forward(
        self,
        values: Tensor,
        mask: Tensor,
        condition: Tensor | None = None,
        reverse: bool = False,
    ) -> tuple[Tensor, Tensor]:
        numeric_mask = _validate_transform_input(values, mask)
        if reverse:
            transformed = torch.exp(values)
            logdet = (values * numeric_mask).sum(dim=(1, 2))
        else:
            if torch.any(values.masked_select(mask) <= 0):
                raise ValueError("log transform requires positive valid values")
            valid_values = torch.where(mask, values, torch.ones_like(values))
            transformed = torch.log(valid_values)
            logdet = -(transformed * numeric_mask).sum(dim=(1, 2))
        output = torch.where(mask, transformed, values)
        return output, logdet


class ElementwiseAffine(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(channels, 1))

    def forward(
        self,
        values: Tensor,
        mask: Tensor,
        condition: Tensor | None = None,
        reverse: bool = False,
    ) -> tuple[Tensor, Tensor]:
        numeric_mask = _validate_transform_input(values, mask)
        scale = torch.exp(self.log_scale)
        if reverse:
            transformed = (values - self.offset) / scale
            sign = -1
        else:
            transformed = self.offset + scale * values
            sign = 1
        output = torch.where(mask, transformed, values)
        logdet = sign * (self.log_scale * numeric_mask).sum(dim=(1, 2))
        return output, logdet


class Flip(nn.Module):
    def forward(
        self,
        values: Tensor,
        mask: Tensor,
        condition: Tensor | None = None,
        reverse: bool = False,
    ) -> tuple[Tensor, Tensor]:
        _validate_transform_input(values, mask)
        transformed = torch.flip(values, dims=(1,))
        output = torch.where(mask, transformed, values)
        return output, values.new_zeros(values.shape[0])


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, values: Tensor) -> Tensor:
        return self.normalization(values.transpose(1, 2)).transpose(1, 2)


class DepthwiseSeparableStack(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        layer_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.depthwise = nn.ModuleList()
        self.pointwise = nn.ModuleList()
        self.depthwise_norms = nn.ModuleList()
        self.pointwise_norms = nn.ModuleList()
        for layer_index in range(layer_count):
            dilation = kernel_size**layer_index
            padding = (kernel_size * dilation - dilation) // 2
            self.depthwise.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                    groups=channels,
                )
            )
            self.pointwise.append(nn.Conv1d(channels, channels, 1))
            self.depthwise_norms.append(ChannelLayerNorm(channels))
            self.pointwise_norms.append(ChannelLayerNorm(channels))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: Tensor,
        mask: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        numeric_mask = mask.to(dtype=values.dtype)
        if condition is not None:
            if condition.shape != values.shape:
                raise ValueError("flow condition must match hidden values")
            values = values + condition
        for depthwise, pointwise, depth_norm, point_norm in zip(
            self.depthwise,
            self.pointwise,
            self.depthwise_norms,
            self.pointwise_norms,
            strict=True,
        ):
            hidden = F.gelu(depth_norm(depthwise(values * numeric_mask)))
            hidden = F.gelu(point_norm(pointwise(hidden)))
            values = values + self.dropout(hidden)
        return values * numeric_mask


class ConvFlow(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        layer_count: int,
        spline_bins: int,
        spline_tail_bound: float,
    ) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("convolutional flow channels must be even")
        self.half_channels = channels // 2
        self.hidden_channels = hidden_channels
        self.spline_bins = spline_bins
        self.spline_tail_bound = spline_tail_bound
        self.input_projection = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.conditioner = DepthwiseSeparableStack(
            hidden_channels,
            kernel_size,
            layer_count,
            dropout=0.0,
        )
        parameter_channels = self.half_channels * (spline_bins * 3 - 1)
        self.projection = nn.Conv1d(hidden_channels, parameter_channels, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        values: Tensor,
        mask: Tensor,
        condition: Tensor | None = None,
        reverse: bool = False,
    ) -> tuple[Tensor, Tensor]:
        numeric_mask = _validate_transform_input(values, mask)
        first, second = values.split(self.half_channels, dim=1)
        hidden = self.input_projection(first * numeric_mask)
        hidden = self.conditioner(hidden, mask, condition)
        parameters = self.projection(hidden) * numeric_mask
        batch, _, frames = first.shape
        parameters = parameters.view(
            batch,
            self.half_channels,
            self.spline_bins * 3 - 1,
            frames,
        ).permute(0, 1, 3, 2)
        widths = parameters[..., : self.spline_bins] / math.sqrt(self.hidden_channels)
        heights = parameters[..., self.spline_bins : 2 * self.spline_bins]
        heights = heights / math.sqrt(self.hidden_channels)
        derivatives = parameters[..., 2 * self.spline_bins :]
        transformed, element_logdet = unconstrained_rational_quadratic_spline(
            second,
            widths,
            heights,
            derivatives,
            inverse=reverse,
            bound=self.spline_tail_bound,
        )
        transformed = torch.where(mask, transformed, second)
        output = torch.cat((first, transformed), dim=1)
        logdet = (element_logdet * numeric_mask).sum(dim=(1, 2))
        return output, logdet
