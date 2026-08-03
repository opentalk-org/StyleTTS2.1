from typing import TYPE_CHECKING

import torch

from ..data import TrainingBatch
from ..gradient_sync import synchronize_gradients

if TYPE_CHECKING:
    from ..setup import TrainingRuntime
    from .training_forward import ForwardOutput


def discriminator_step(
    runtime: "TrainingRuntime",
    output: "ForwardOutput",
    batch: TrainingBatch,
    waveform_active: bool,
    prosody_active: bool,
    slm_active: bool,
) -> tuple[torch.Tensor | float, torch.Tensor | float, torch.Tensor | float]:
    modules = runtime.models.modules
    if not waveform_active and not prosody_active and not slm_active:
        for name in (
            "mpd",
            "msd",
            "prosody_discriminator",
            "duration_discriminator",
            "wd",
        ):
            modules[name].requires_grad_(False)
        return 0.0, 0.0, 0.0
    accelerator = runtime.accelerator
    optimizer = runtime.optimizer
    optimizer.zero_grad()
    waveform_total = output.reconstructed.new_zeros(())
    prosody_total = output.reconstructed.new_zeros(())
    slm_total = output.reconstructed.new_zeros(())
    waveform_items = (("mpd", modules.mpd), ("msd", modules.msd))
    for name, discriminator in waveform_items if waveform_active else ():
        with accelerator.autocast():
            loss = runtime.losses.discriminator(
                output.waveform.detach(),
                output.reconstructed.detach(),
                discriminator,
            )
        accelerator.backward(loss)
        synchronize_gradients(accelerator, modules, (name,))
        optimizer.step(name)
        discriminator.requires_grad_(False)
        waveform_total = waveform_total + loss.detach()
    prosody_items = (
        (
            "prosody_discriminator",
            runtime.losses.prosody_discriminator,
            output.prosody_fake,
            output.prosody_real,
            batch.mel_lengths.to(output.reconstructed.device) // 2,
        ),
        (
            "duration_discriminator",
            runtime.losses.duration_discriminator,
            output.duration_fake,
            output.duration_real,
            batch.input_lengths.to(output.reconstructed.device),
        ),
    )
    for name, objective, fake, real, lengths in prosody_items if prosody_active else ():
        loss = objective(fake, real, output.style_target, lengths, real.size(-1))
        accelerator.backward(loss)
        synchronize_gradients(accelerator, modules, (name,))
        optimizer.step(name)
        modules[name].requires_grad_(False)
        prosody_total = prosody_total + loss.detach()
    if slm_active:
        with accelerator.autocast():
            slm_total = runtime.losses.wavlm.discriminator(
                output.waveform.detach().squeeze(1),
                output.reconstructed.detach().squeeze(1),
            )
        accelerator.backward(slm_total)
        synchronize_gradients(accelerator, modules, ("wd",))
        optimizer.step("wd")
        modules.wd.requires_grad_(False)
    return waveform_total, prosody_total, slm_total.detach()


def prosody_generator_loss(
    runtime: "TrainingRuntime",
    output: "ForwardOutput",
    batch: TrainingBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mel_lengths = batch.mel_lengths.to(output.reconstructed.device) // 2
    prosody, prosody_features = runtime.losses.prosody_generator(
        output.prosody_fake,
        output.prosody_real,
        output.style_target,
        mel_lengths,
        output.prosody_real.size(-1),
    )
    text_lengths = batch.input_lengths.to(output.reconstructed.device)
    duration, duration_features = runtime.losses.duration_generator(
        output.duration_fake,
        output.duration_real,
        output.style_target,
        text_lengths,
        output.duration_real.size(-1),
    )
    feature_matching = prosody_features + duration_features
    adversarial = prosody + duration
    return adversarial + feature_matching, adversarial, feature_matching
