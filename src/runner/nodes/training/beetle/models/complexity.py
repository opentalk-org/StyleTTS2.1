from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from ..config.training import BeetleConfig, ComplexityConfig
from .decoder import Decoder
from .features import FeatureLinear
from .generator import Generator


@dataclass(frozen=True)
class ComplexityReport:
    parameter_count: int
    flops: int
    generated_samples: int
    generated_seconds: float
    gflops_per_second: float
    over_budget: bool


def profile_latent_audio(
    feature_linear: FeatureLinear,
    decoder: Decoder,
    generator: Generator,
    config: BeetleConfig,
) -> ComplexityReport:
    roots: tuple[nn.Module, ...] = (feature_linear, decoder, generator)
    module_states: list[tuple[nn.Module, bool]] = []
    seen_modules: set[int] = set()
    for root in roots:
        for module in root.modules():
            identity = id(module)
            if identity not in seen_modules:
                seen_modules.add(identity)
                module_states.append((module, module.training))

    seen_parameters: set[int] = set()
    parameter_count = 0
    for root in roots:
        for parameter in root.parameters():
            identity = id(parameter)
            if identity not in seen_parameters:
                seen_parameters.add(identity)
                parameter_count += parameter.numel()

    feature_parameter = next(feature_linear.parameters())
    output_hop = (
        feature_linear.config.upsample_rate * generator.config.output_hop()
    )
    benchmark_samples = round(
        config.audio.sample_rate * config.complexity.benchmark_seconds
    )
    latent_frames, remainder = divmod(benchmark_samples, output_hop)
    if remainder:
        raise ValueError("complexity benchmark must contain whole latent frames")
    input_generator = torch.Generator(device=feature_parameter.device).manual_seed(
        config.runtime.seed
    )
    latent = torch.randn(
        1,
        feature_linear.config.latent_channels,
        latent_frames,
        device=feature_parameter.device,
        dtype=feature_parameter.dtype,
        generator=input_generator,
    )
    latent_mask = torch.ones(
        1,
        1,
        latent_frames,
        device=latent.device,
        dtype=torch.bool,
    )
    frame_mask = torch.ones(
        1,
        1,
        latent_frames * feature_linear.config.upsample_rate,
        device=latent.device,
        dtype=torch.bool,
    )
    source_generator = torch.Generator(device=feature_parameter.device).manual_seed(
        config.runtime.seed
    )

    try:
        for root in roots:
            root.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            acoustic = feature_linear(latent, latent_mask, frame_mask)
            decoded = decoder(
                latent,
                acoustic.f0,
                acoustic.n,
                latent_mask,
                frame_mask,
            )
            waveform = generator(
                decoded.features,
                decoded.f0,
                decoded.mask,
                source_generator,
            )
        flops = int(counter.get_total_flops())
    finally:
        for module, training in module_states:
            module.training = training

    generated_samples = waveform.shape[-1]
    generated_seconds = generated_samples / config.audio.sample_rate
    if parameter_count <= 0 or flops <= 0:
        raise ValueError("complexity profile must measure positive parameters and FLOPs")
    if generated_samples != benchmark_samples:
        raise ValueError(
            f"complexity profile generated {generated_samples} samples; "
            f"expected {benchmark_samples}"
        )
    gflops_per_second = flops / 1e9 / generated_seconds
    ceiling = config.complexity.latent_audio_max_gflops_per_second
    return ComplexityReport(
        parameter_count=parameter_count,
        flops=flops,
        generated_samples=generated_samples,
        generated_seconds=generated_seconds,
        gflops_per_second=gflops_per_second,
        over_budget=gflops_per_second >= ceiling,
    )


def require_complexity_budget(
    report: ComplexityReport,
    complexity_config: ComplexityConfig,
) -> None:
    ceiling = complexity_config.latent_audio_max_gflops_per_second
    if report.gflops_per_second >= ceiling:
        raise ValueError(
            f"latent-to-audio complexity is {report.gflops_per_second:.6f} "
            f"GFLOPs/s and must be strictly below {ceiling:.6f} GFLOPs/s; "
            "do not deploy without an approved complexity budget"
        )
