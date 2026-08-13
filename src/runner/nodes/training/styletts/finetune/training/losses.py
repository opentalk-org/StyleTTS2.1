from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor, nn


@dataclass(frozen=True)
class WaveformDiscriminatorLosses:
    total: Tensor
    real_lsgan: Tensor
    generated_lsgan: Tensor
    tprls: Tensor
    real_accuracy: Tensor
    generated_accuracy: Tensor
    accuracy: Tensor


@dataclass(frozen=True)
class WaveformGeneratorLosses:
    total: Tensor
    feature_matching: Tensor
    lsgan: Tensor
    tprls: Tensor


class STFTLoss(nn.Module):
    def __init__(
        self,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window=torch.hann_window,
    ):
        super().__init__()
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=fft_size,
            win_length=win_length,
            hop_length=shift_size,
            window_fn=window,
        )

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction = (torch.log(1e-5 + self.to_mel(prediction)) + 4) / 4
        target = (torch.log(1e-5 + self.to_mel(target)) + 4) / 4
        return torch.norm(target - prediction, p=1) / torch.norm(target, p=1)


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=(1024, 2048, 512),
        hop_sizes=(120, 240, 50),
        win_lengths=(600, 1200, 240),
        window=torch.hann_window,
    ):
        super().__init__()
        self.stft_losses = nn.ModuleList(
            STFTLoss(fft_size, hop_size, win_length, window)
            for fft_size, hop_size, win_length in zip(
                fft_sizes,
                hop_sizes,
                win_lengths,
                strict=True,
            )
        )

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.stack(
            [loss(prediction, target) for loss in self.stft_losses]
        ).mean()


def discriminator_tprls_loss(real_scores, generated_scores) -> Tensor:
    loss = real_scores[0].new_zeros(())
    for real, generated in zip(real_scores, generated_scores, strict=True):
        tau = 0.04
        median = torch.median(real - generated)
        relative = torch.mean(
            (((real - generated) - median) ** 2)[real < generated + median]
        )
        loss = loss + tau - F.relu(tau - relative)
    return loss


def generator_tprls_loss(real_scores, generated_scores) -> Tensor:
    loss = generated_scores[0].new_zeros(())
    for real, generated in zip(real_scores, generated_scores, strict=True):
        tau = 0.04
        median = torch.median(generated - real)
        relative = torch.mean(
            (((generated - real) - median) ** 2)[
                generated < real + median
            ]
        )
        loss = loss + tau - F.relu(tau - relative)
    return loss


def waveform_discriminator_losses(
    real_scores,
    generated_scores,
) -> WaveformDiscriminatorLosses:
    real_lsgan = real_scores[0].new_zeros(())
    generated_lsgan = real_scores[0].new_zeros(())
    real_accuracy = real_scores[0].new_zeros(())
    generated_accuracy = real_scores[0].new_zeros(())
    for real, generated in zip(real_scores, generated_scores, strict=True):
        real_lsgan = real_lsgan + torch.mean((1 - real) ** 2)
        generated_lsgan = generated_lsgan + torch.mean(generated**2)
        real_accuracy = real_accuracy + (real > 0.5).float().mean()
        generated_accuracy = (
            generated_accuracy + (generated < 0.5).float().mean()
        )
    discriminator_count = len(real_scores)
    real_accuracy = real_accuracy / discriminator_count
    generated_accuracy = generated_accuracy / discriminator_count
    tprls = discriminator_tprls_loss(real_scores, generated_scores)
    return WaveformDiscriminatorLosses(
        total=(real_lsgan + generated_lsgan + tprls).mean(),
        real_lsgan=real_lsgan.mean(),
        generated_lsgan=generated_lsgan.mean(),
        tprls=tprls.mean(),
        real_accuracy=real_accuracy,
        generated_accuracy=generated_accuracy,
        accuracy=(real_accuracy + generated_accuracy) / 2,
    )


def waveform_generator_losses(
    real_scores,
    generated_scores,
    real_features,
    generated_features,
) -> WaveformGeneratorLosses:
    feature = generated_scores[0].new_zeros(())
    for real_maps, generated_maps in zip(
        real_features,
        generated_features,
        strict=True,
    ):
        for real, generated in zip(real_maps, generated_maps, strict=True):
            feature = feature + torch.mean(torch.abs(real - generated))
    feature = feature * 2
    lsgan = sum(torch.mean((1 - score) ** 2) for score in generated_scores)
    tprls = generator_tprls_loss(real_scores, generated_scores)
    return WaveformGeneratorLosses(
        total=(feature + lsgan + tprls).mean(),
        feature_matching=feature.mean(),
        lsgan=lsgan.mean(),
        tprls=tprls.mean(),
    )


def wavlm_feature_loss(real_features, generated_features) -> Tensor:
    return sum(
        torch.mean(torch.abs(real - generated))
        for real, generated in zip(real_features, generated_features, strict=True)
    ).mean()


def slm_discriminator_loss(real_scores: Tensor, generated_scores: Tensor) -> Tensor:
    return (
        torch.mean((1 - real_scores) ** 2)
        + torch.mean(generated_scores**2)
    ).mean()


def slm_generator_loss(generated_scores: Tensor) -> Tensor:
    return torch.mean((1 - generated_scores) ** 2)


def speaker_losses(
    real_values: tuple[Tensor, ...],
    generated_values: tuple[Tensor, ...],
    real_embedding: Tensor,
    generated_embedding: Tensor,
) -> tuple[Tensor, Tensor]:
    feature = generated_embedding.new_zeros(())
    for real, generated in zip(
        real_values,
        generated_values,
        strict=True,
    ):
        feature = feature + F.l1_loss(generated, real)
    similarity = 1 - F.cosine_similarity(
        F.normalize(generated_embedding, dim=-1),
        F.normalize(real_embedding, dim=-1),
        dim=-1,
    ).mean()
    return feature, similarity


def prosody_discriminator_loss(
    real_scores: Tensor,
    generated_scores: Tensor,
    lengths: Tensor,
) -> Tensor:
    losses = []
    for index, length_value in enumerate(lengths):
        length = int(length_value.item())
        generated = generated_scores[index, :length]
        real = real_scores[index, :length]
        item = (
            torch.mean(generated.square())
            + torch.mean((1 - real).square())
            + discriminator_tprls_loss(
                [real.unsqueeze(0)],
                [generated.unsqueeze(0)],
            )
        )
        if not torch.isnan(item):
            losses.append(item)
    if losses:
        return torch.stack(losses).mean()
    return (generated_scores.sum() + real_scores.sum()) * 0


def prosody_generator_losses(
    real_scores: Tensor,
    generated_scores: Tensor,
    real_features,
    generated_features,
    lengths: Tensor,
) -> tuple[Tensor, Tensor]:
    adversarial = []
    feature_matching = []
    for index, length_value in enumerate(lengths):
        length = int(length_value.item())
        generated = generated_scores[index, :length]
        real = real_scores[index, :length]
        item = torch.mean((1 - generated).square()) + generator_tprls_loss(
            [real.unsqueeze(0)],
            [generated.unsqueeze(0)],
        )
        if not torch.isnan(item):
            adversarial.append(item)
        feature_matching.append(
            sum(
                F.l1_loss(real_map[index, :length], generated_map[index, :length])
                for real_map, generated_map in zip(
                    real_features,
                    generated_features,
                    strict=True,
                )
            )
        )
    if adversarial:
        adversarial_loss = torch.stack(adversarial).mean()
    else:
        adversarial_loss = generated_scores.sum() * 0
    return adversarial_loss, torch.stack(feature_matching).mean()


def reconstruction_loss(
    target: Tensor,
    prediction: Tensor,
    lengths: Tensor | list[int],
    divisor: float = 1.0,
) -> Tensor:
    length_total = torch.as_tensor(lengths, device=target.device).sum()
    valid_ratio = target.numel() / length_total
    return F.smooth_l1_loss(target, prediction) * valid_ratio / divisor


def acoustic_losses(
    losses: list[Tensor],
) -> Tensor:
    return torch.stack(losses).mean()
