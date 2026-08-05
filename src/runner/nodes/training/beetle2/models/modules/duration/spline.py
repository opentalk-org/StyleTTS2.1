import math

import torch
from torch import Tensor
from torch.nn import functional as F


MIN_BIN_WIDTH = 1e-3
MIN_BIN_HEIGHT = 1e-3
MIN_DERIVATIVE = 1e-3


def _searchsorted(bin_locations: Tensor, inputs: Tensor) -> Tensor:
    adjusted_end = bin_locations[..., -1:] + 1e-6
    locations = torch.cat((bin_locations[..., :-1], adjusted_end), dim=-1)
    return torch.sum(inputs.unsqueeze(-1) >= locations, dim=-1) - 1


def _gather(values: Tensor, indices: Tensor) -> Tensor:
    return values.gather(-1, indices.unsqueeze(-1)).squeeze(-1)


def rational_quadratic_spline(
    inputs: Tensor,
    unnormalized_widths: Tensor,
    unnormalized_heights: Tensor,
    unnormalized_derivatives: Tensor,
    inverse: bool,
    bound: float,
) -> tuple[Tensor, Tensor]:
    bin_count = unnormalized_widths.shape[-1]
    widths = MIN_BIN_WIDTH + (1 - MIN_BIN_WIDTH * bin_count) * torch.softmax(
        unnormalized_widths,
        dim=-1,
    )
    cumwidths = F.pad(torch.cumsum(widths, dim=-1), (1, 0))
    cumwidths = (2 * bound) * cumwidths - bound
    cumwidths = torch.cat(
        (
            cumwidths[..., :1] * 0 - bound,
            cumwidths[..., 1:-1],
            cumwidths[..., -1:] * 0 + bound,
        ),
        dim=-1,
    )
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = MIN_BIN_HEIGHT + (1 - MIN_BIN_HEIGHT * bin_count) * torch.softmax(
        unnormalized_heights,
        dim=-1,
    )
    cumheights = F.pad(torch.cumsum(heights, dim=-1), (1, 0))
    cumheights = (2 * bound) * cumheights - bound
    cumheights = torch.cat(
        (
            cumheights[..., :1] * 0 - bound,
            cumheights[..., 1:-1],
            cumheights[..., -1:] * 0 + bound,
        ),
        dim=-1,
    )
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    derivatives = MIN_DERIVATIVE + F.softplus(unnormalized_derivatives)
    bin_indices = _searchsorted(cumheights if inverse else cumwidths, inputs)
    input_cumwidths = _gather(cumwidths, bin_indices)
    input_widths = _gather(widths, bin_indices)
    input_cumheights = _gather(cumheights, bin_indices)
    input_heights = _gather(heights, bin_indices)
    deltas = heights / widths
    input_delta = _gather(deltas, bin_indices)
    input_derivative = _gather(derivatives, bin_indices)
    next_derivative = _gather(derivatives[..., 1:], bin_indices)

    if inverse:
        offset = inputs - input_cumheights
        derivative_sum = input_derivative + next_derivative - 2 * input_delta
        a = offset * derivative_sum + input_heights * (input_delta - input_derivative)
        b = input_heights * input_derivative - offset * derivative_sum
        c = -input_delta * offset
        discriminant = b.square() - 4 * a * c
        theta = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = theta * input_widths + input_cumwidths
    else:
        theta = (inputs - input_cumwidths) / input_widths
        theta_complement = theta * (1 - theta)
        numerator = input_heights * (
            input_delta * theta.square() + input_derivative * theta_complement
        )
        denominator = (
            input_delta
            + (input_derivative + next_derivative - 2 * input_delta) * theta_complement
        )
        outputs = input_cumheights + numerator / denominator

    theta_complement = theta * (1 - theta)
    denominator = (
        input_delta
        + (input_derivative + next_derivative - 2 * input_delta) * theta_complement
    )
    derivative_numerator = input_delta.square() * (
        next_derivative * theta.square()
        + 2 * input_delta * theta_complement
        + input_derivative * (1 - theta).square()
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
    return outputs, -logabsdet if inverse else logabsdet


def unconstrained_rational_quadratic_spline(
    inputs: Tensor,
    unnormalized_widths: Tensor,
    unnormalized_heights: Tensor,
    unnormalized_derivatives: Tensor,
    inverse: bool,
    bound: float,
) -> tuple[Tensor, Tensor]:
    boundary_value = math.log(math.expm1(1 - MIN_DERIVATIVE))
    derivatives = F.pad(unnormalized_derivatives, (1, 1), value=boundary_value)
    inside = (inputs >= -bound) & (inputs <= bound)
    safe_inputs = torch.where(inside, inputs, torch.zeros_like(inputs))
    spline_output, spline_logabsdet = rational_quadratic_spline(
        safe_inputs,
        unnormalized_widths,
        unnormalized_heights,
        derivatives,
        inverse,
        bound,
    )
    outputs = torch.where(inside, spline_output, inputs)
    logabsdet = torch.where(inside, spline_logabsdet, torch.zeros_like(inputs))
    return outputs, logabsdet
