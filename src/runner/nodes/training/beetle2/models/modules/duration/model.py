"""Duration likelihood derived from ``papers/duration-flow.md`` and Piper 73c04d8."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ....config.architecture import DurationFlowConfig
from .transforms import (
    ConvFlow,
    DurationConvolutionStack,
    ElementwiseAffine,
    Flip,
    LogTransform,
)


def standard_normal_negative_log_likelihood(
    values: Tensor,
    logdet: Tensor,
    mask: Tensor,
) -> Tensor:
    numeric_mask = mask.to(dtype=values.dtype)
    base_nll = 0.5 * (math.log(2 * math.pi) + values.square())
    return (base_nll * numeric_mask).sum(dim=(1, 2)) - logdet


class DurationConditionEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        config: DurationFlowConfig,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_channels, config.hidden_channels, 1)
        self.convolutions = DurationConvolutionStack(
            config.hidden_channels,
            config.kernel_size,
            config.convolution_layers,
            config.dropout,
        )
        self.output_projection = nn.Conv1d(
            config.hidden_channels,
            config.hidden_channels,
            1,
        )

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        hidden = self.input_projection(values)
        hidden = self.convolutions(hidden, mask)
        return self.output_projection(hidden) * mask.to(dtype=hidden.dtype)


def _build_flows(config: DurationFlowConfig, count: int) -> nn.ModuleList:
    flows = nn.ModuleList((ElementwiseAffine(2),))
    for _ in range(count):
        flows.append(
            ConvFlow(
                channels=2,
                hidden_channels=config.hidden_channels,
                kernel_size=config.kernel_size,
                layer_count=config.convolution_layers,
                spline_bins=config.spline_bins,
                spline_tail_bound=config.spline_tail_bound,
            )
        )
        flows.append(Flip())
    return flows


class DurationPredictor(nn.Module):
    def __init__(self, config: DurationFlowConfig) -> None:
        super().__init__()
        self.config = config
        self.condition_encoder = DurationConditionEncoder(
            config.condition_channels,
            config,
        )
        self.duration_encoder = DurationConditionEncoder(1, config)
        self.log_transform = LogTransform()
        self.flows = _build_flows(config, config.flow_count)
        self.posterior_flows = _build_flows(config, config.posterior_flow_count)

    def forward(
        self,
        duration: Tensor,
        condition: Tensor,
        mask: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        return self.log_prob(duration, condition, mask, generator)

    def _run_flows(
        self,
        values: Tensor,
        flows: list[nn.Module] | nn.ModuleList,
        mask: Tensor,
        condition: Tensor,
        reverse: bool,
    ) -> tuple[Tensor, Tensor]:
        total_logdet = values.new_zeros(values.shape[0])
        for flow in flows:
            values, logdet = flow(values, mask, condition, reverse=reverse)
            total_logdet = total_logdet + logdet
        return values, total_logdet

    def log_prob(
        self,
        duration: Tensor,
        condition: Tensor,
        mask: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        numeric_mask = mask.to(dtype=condition.dtype)
        encoded_condition = self.condition_encoder(condition, mask)
        encoded_duration = self.duration_encoder(duration, mask)
        posterior_condition = encoded_condition + encoded_duration

        posterior_base = (
            torch.randn(
                duration.shape[0],
                2,
                duration.shape[2],
                dtype=duration.dtype,
                device=duration.device,
                generator=generator,
            )
            * numeric_mask
        )
        posterior_value, posterior_logdet = self._run_flows(
            posterior_base,
            self.posterior_flows,
            mask,
            posterior_condition,
            reverse=False,
        )
        dequantization_logit, auxiliary = posterior_value.chunk(2, dim=1)
        dequantization = torch.sigmoid(dequantization_logit)
        continuous_duration = (duration - dequantization) * numeric_mask
        sigmoid_logdet = (
            (F.logsigmoid(dequantization_logit) + F.logsigmoid(-dequantization_logit))
            * numeric_mask
        ).sum(dim=(1, 2))
        posterior_log_density = (
            (
                -0.5 * (math.log(2 * math.pi) + posterior_base.square()) * numeric_mask
            ).sum(dim=(1, 2))
            - posterior_logdet
            - sigmoid_logdet
        )

        log_duration, log_logdet = self.log_transform(
            continuous_duration,
            mask,
        )
        main_value = torch.cat((log_duration, auxiliary), dim=1)
        main_value, main_logdet = self._run_flows(
            main_value,
            self.flows,
            mask,
            encoded_condition,
            reverse=False,
        )
        main_nll = standard_normal_negative_log_likelihood(
            main_value,
            log_logdet + main_logdet,
            mask,
        )
        return main_nll + posterior_log_density

    def sample(
        self,
        condition: Tensor,
        mask: Tensor,
        generator: torch.Generator,
    ) -> Tensor:
        encoded_condition = self.condition_encoder(condition, mask)
        values = torch.randn(
            condition.shape[0],
            2,
            condition.shape[2],
            dtype=condition.dtype,
            device=condition.device,
            generator=generator,
        )
        reverse_flows = list(reversed(self.flows))
        reverse_flows = reverse_flows[:-2] + [reverse_flows[-1]]
        values, _ = self._run_flows(
            values,
            reverse_flows,
            mask,
            encoded_condition,
            reverse=True,
        )
        log_duration = values[:, :1]
        duration = torch.ceil(torch.exp(log_duration))
        return duration * mask.to(dtype=duration.dtype)
