from pathlib import Path

import torch
import torch.nn.functional as F

from runflow.runtime.cancellation import check_cancel

from ...studio.val_sample_export import (
    ValidationArtifactRenderer,
    ValidationSample,
    ValidationSampleArtifacts,
)
from ...stages import TrainingStageSpec, stage_for_step
from ..config import TrainingConfig
from ..data import TrainingBatch, ValidationResult
from ..setup import TrainingRuntime
from ..utils import (
    length_to_mask,
    mask_from_lens,
    maximum_path,
)
from .validation_batch import (
    ValidationBatch,
    acoustic_losses,
)
from .validation_synthesis import synthesize_validation


class Validator:
    def __init__(
        self,
        config: TrainingConfig,
        runtime: TrainingRuntime,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.artifacts = ValidationArtifactRenderer(
            Path(config.log_dir),
            config.preprocess_params.sr,
        )

    def run(
        self,
        batches,
        step: int,
    ) -> ValidationResult:
        modules = self.runtime.models.modules
        for module in modules.values():
            module.eval()
        totals = {
            "mel_loss": torch.zeros(
                (),
                device=self.runtime.accelerator.device,
            ),
            "dur_loss": torch.zeros(
                (),
                device=self.runtime.accelerator.device,
            ),
            "F0_loss": torch.zeros(
                (),
                device=self.runtime.accelerator.device,
            ),
        }
        samples: list[ValidationSampleArtifacts] = []
        count = 0
        stage = stage_for_step(self.config.training_stages, step - 1)
        with torch.no_grad():
            for batch in batches:
                check_cancel()
                batch = batch.to(self.runtime.accelerator.device)
                values, exported = self._validate_batch(
                    batch,
                    step,
                    stage,
                    export_samples=not samples,
                )
                for name, value in values.items():
                    totals[name] += value
                samples.extend(exported)
                count += 1
        if count == 0:
            raise ValueError("validation loader produced no batches")
        reduced = {
            name: value / count
            for name, value in totals.items()
        }
        return ValidationResult(
            reduced,
            samples,
        )

    def _validate_batch(
        self,
        batch: TrainingBatch,
        step: int,
        stage: TrainingStageSpec,
        export_samples: bool,
    ) -> tuple[dict[str, torch.Tensor], list[ValidationSampleArtifacts]]:
        modules = self.runtime.models.modules
        n_down = self.runtime.models.n_down
        with self.runtime.accelerator.autocast():
            mask = length_to_mask(
                batch.mel_lengths // (2**n_down),
                batch.mels.device,
            )
            text_mask = length_to_mask(
                batch.input_lengths,
                batch.texts.device,
            )
            _, _, soft_alignment = modules.text_aligner(
                batch.mels,
                mask,
                batch.texts,
            )
            soft_alignment = soft_alignment.transpose(-1, -2)
            soft_alignment = soft_alignment[..., 1:].transpose(-1, -2)
            alignment_mask = mask_from_lens(
                soft_alignment,
                batch.input_lengths,
                batch.mel_lengths // (2**n_down),
            )
            monotonic = maximum_path(soft_alignment, alignment_mask)
            text_encoding = modules.text_encoder(
                batch.texts,
                batch.input_lengths,
                text_mask,
            )
            aligned_text = text_encoding @ monotonic
            duration_targets = monotonic.sum(axis=-1).detach()
            duration_styles = []
            acoustic_styles = []
            for index, length in enumerate(batch.mel_lengths):
                mel = batch.mels[index, :, :length]
                duration_styles.append(
                    modules.predictor_encoder(
                        mel.unsqueeze(0).unsqueeze(1)
                    )
                )
                acoustic_styles.append(
                    modules.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                )
            duration_style = torch.stack(duration_styles).squeeze(1)
            style = torch.stack(acoustic_styles).squeeze(1)
            bert = modules.bert(
                batch.texts,
                attention_mask=(~text_mask).int(),
            )
            duration_encoding = modules.bert_encoder(bert).transpose(-1, -2)
            validation_batch = self._full_batch(batch, aligned_text)
            (
                reconstructed,
                duration_predictions,
                predicted_f0,
                predicted_norm,
                target_f0,
                target_norm,
                decode_alignment,
                decode_lengths,
            ) = synthesize_validation(
                self.runtime,
                batch,
                stage,
                validation_batch,
                text_encoding,
                duration_encoding,
                bert,
                text_mask,
                monotonic,
                duration_style,
                style,
            )
            waveform = validation_batch.waveform.unsqueeze(1)
            prediction_sample_lengths = [
                min(length * 600, reconstructed.size(-1))
                for length in decode_lengths
            ]
            mel_loss, f0_loss = acoustic_losses(
                self.runtime.losses.stft,
                reconstructed,
                waveform,
                predicted_f0,
                target_f0,
                decode_lengths,
                validation_batch.sample_lengths,
                prediction_sample_lengths,
            )
            duration_loss = self._duration_loss(
                duration_predictions,
                duration_targets,
                batch.input_lengths,
            )
        samples = []
        if export_samples:
            sample_count = min(4, waveform.size(0))
            validation_samples = [
                ValidationSample(
                    ground_truth=waveform[
                        index,
                        :,
                        : prediction_sample_lengths[index],
                    ],
                    prediction=reconstructed[
                        index,
                        :,
                        : validation_batch.sample_lengths[index],
                    ],
                    target_f0=target_f0[
                        index,
                        : decode_lengths[index],
                    ],
                    predicted_f0=predicted_f0[
                        index,
                        : decode_lengths[index],
                    ],
                    target_n=target_norm[
                        index,
                        : decode_lengths[index],
                    ],
                    predicted_n=predicted_norm[
                        index,
                        : decode_lengths[index],
                    ],
                    soft_attention=soft_alignment[
                        index,
                        : batch.input_lengths[index],
                        : batch.mel_lengths[index] // (2**n_down),
                    ],
                    hard_attention=decode_alignment[
                        index,
                        : batch.input_lengths[index],
                        : decode_lengths[index],
                    ],
                )
                for index in range(sample_count)
            ]
            samples = self.artifacts.render(
                step,
                validation_samples,
            )
        return {
            "mel_loss": mel_loss,
            "dur_loss": duration_loss,
            "F0_loss": f0_loss,
        }, samples

    @staticmethod
    def _full_batch(
        batch: TrainingBatch,
        aligned_text: torch.Tensor,
    ) -> ValidationBatch:
        lengths = [int(length.item() // 2) for length in batch.mel_lengths]
        sample_lengths = [
            min(length * 600, batch.waves[index].shape[0])
            for index, length in enumerate(lengths)
        ]
        frames = max(lengths)
        waveform = batch.mels.new_zeros((len(lengths), frames * 600))
        for index, samples in enumerate(sample_lengths):
            waveform[index, :samples] = torch.from_numpy(
                batch.waves[index][:samples]
            ).to(batch.mels.device)
        return ValidationBatch(
            aligned_text[:, :, :frames],
            [0] * len(lengths),
            frames,
            batch.mels[:, :, : frames * 2].detach(),
            waveform.detach(),
            lengths,
            sample_lengths,
        )

    @staticmethod
    def _duration_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.zeros((), device=predictions.device)
        for prediction, target, length in zip(predictions, targets, lengths):
            prediction = prediction[:length, :]
            target = target[:length].long()
            predicted = torch.sigmoid(prediction).sum(axis=1)
            loss += F.l1_loss(
                predicted[1 : length - 1],
                target[1 : length - 1],
            )
        return loss / predictions.size(0)
