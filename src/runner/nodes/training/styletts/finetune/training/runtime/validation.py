import numpy as np
import torch
import torch.nn.functional as F

from runflow.runtime.cancellation import check_cancel

from ...studio.val_sample_export import export_finetune_val_wavs_for_studio
from ..config import TrainingConfig
from ..data import TrainingBatch, ValidationResult
from ..setup import TrainingRuntime
from ..utils import (
    length_to_mask,
    mask_from_lens,
    maximum_path,
)


class Validator:
    def __init__(
        self,
        config: TrainingConfig,
        runtime: TrainingRuntime,
    ) -> None:
        self.config = config
        self.runtime = runtime

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
        samples: list[dict[str, str]] = []
        count = 0
        with torch.no_grad():
            for batch in batches:
                check_cancel()
                batch = batch.to(self.runtime.accelerator.device)
                values, exported = self._validate_batch(
                    batch,
                    step,
                    export_samples=not samples,
                )
                for name, value in values.items():
                    totals[name] += value
                samples.extend(exported)
                count += 1
        if count == 0:
            raise ValueError("validation loader produced no batches")
        count_tensor = torch.tensor(
            float(count),
            device=self.runtime.accelerator.device,
        )
        total_count = self.runtime.accelerator.reduce(
            count_tensor,
            reduction="sum",
        )
        reduced = {
            name: self.runtime.accelerator.reduce(
                value,
                reduction="sum",
            )
            / total_count
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
        export_samples: bool,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, str]]]:
        modules = self.runtime.models.modules
        n_down = self.runtime.models.n_down
        with self.runtime.accelerator.autocast():
            mask = length_to_mask(
                batch.mel_lengths // (2**n_down)
            ).to(batch.mels.device)
            text_mask = length_to_mask(batch.input_lengths).to(
                batch.texts.device
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
            styles = []
            for index, length in enumerate(batch.mel_lengths):
                mel = batch.mels[index, :, :length]
                styles.append(
                    modules.predictor_encoder(
                        mel.unsqueeze(0).unsqueeze(1)
                    )
                )
            duration_style = torch.stack(styles).squeeze(1)
            bert = modules.bert(
                batch.texts,
                attention_mask=(~text_mask).int(),
            )
            duration_encoding = modules.bert_encoder(bert).transpose(-1, -2)
            crops = self._crop(batch, aligned_text)
            (
                duration_predictions,
                prosody,
                predicted_f0,
                predicted_norm,
            ) = modules.predictor(
                duration_encoding,
                duration_style,
                batch.input_lengths,
                monotonic,
                text_mask,
                crops.starts,
                crops.frames,
                modules.predictor_encoder(crops.mel.unsqueeze(1)),
            )
            style = modules.style_encoder(crops.mel.unsqueeze(1))
            target_f0, _, _ = modules.pitch_extractor(
                crops.mel.unsqueeze(1)
            )
            target_f0 = target_f0.squeeze(-1)
            reconstructed = modules.decoder(
                crops.aligned_text,
                predicted_f0,
                predicted_norm,
                style,
            )
            waveform = crops.waveform.unsqueeze(1)
            mel_loss = self.runtime.losses.stft(
                reconstructed.squeeze(1),
                waveform.squeeze(1),
            ).mean()
            f0_loss = F.l1_loss(target_f0, predicted_f0) / 10
            duration_loss = self._duration_loss(
                duration_predictions,
                duration_targets,
                batch.input_lengths,
            )
        samples = []
        if export_samples:
            samples = export_finetune_val_wavs_for_studio(
                self.config.log_dir,
                sample_rate=self.config.preprocess_params.sr,
                step=step,
                y_pred=reconstructed,
                y_gt=waveform,
            )
        return {
            "mel_loss": mel_loss,
            "dur_loss": duration_loss,
            "F0_loss": f0_loss,
        }, samples

    @staticmethod
    def _crop(
        batch: TrainingBatch,
        aligned_text: torch.Tensor,
    ) -> "_ValidationCrops":
        length = int(batch.mel_lengths.min().item() / 2 - 1)
        aligned, starts, mels, waveforms = [], [], [], []
        for index, mel_frames in enumerate(batch.mel_lengths):
            half_frames = int(mel_frames.item() / 2)
            start = int(np.random.randint(0, half_frames - length))
            starts.append(start)
            aligned.append(aligned_text[index, :, start : start + length])
            mels.append(batch.mels[index, :, start * 2 : (start + length) * 2])
            waveforms.append(
                torch.from_numpy(
                    batch.waves[index][start * 600 : (start + length) * 600]
                ).to(batch.mels.device)
            )
        return _ValidationCrops(
            torch.stack(aligned),
            starts,
            length,
            torch.stack(mels).detach(),
            torch.stack(waveforms).float().detach(),
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

class _ValidationCrops:
    def __init__(
        self,
        aligned_text: torch.Tensor,
        starts: list[int],
        frames: int,
        mel: torch.Tensor,
        waveform: torch.Tensor,
    ) -> None:
        self.aligned_text = aligned_text
        self.starts = starts
        self.frames = frames
        self.mel = mel
        self.waveform = waveform
