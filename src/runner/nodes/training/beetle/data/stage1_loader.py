import torch

from shared.db.audio.ranges import SegmentReadRequest
from shared.db.audio.ranges.wav import WavClip

from .audio import AudioPreprocessor
from .index import DatabaseSegmentIndex
from .records import SegmentKey
from .source import ClipBulkLoader, SharedClipBulkLoader
from .stage1_records import (
    FetchedStage1Batch,
    FetchedStage1Source,
    Stage1Batch,
    Stage1PlannedBatch,
    Stage1WindowGeometry,
)


class Stage1WindowLoader:
    def __init__(
        self,
        index: DatabaseSegmentIndex,
        clips: ClipBulkLoader,
        preprocessor: AudioPreprocessor,
        geometry: Stage1WindowGeometry,
    ) -> None:
        self.index = index
        self.clips = clips
        self.preprocessor = preprocessor
        self.geometry = geometry

    @classmethod
    def from_database(
        cls,
        index: DatabaseSegmentIndex,
        cache_bytes: int,
        fetch_workers: int,
        preprocessor: AudioPreprocessor,
        geometry: Stage1WindowGeometry,
    ) -> "Stage1WindowLoader":
        return cls(
            index,
            SharedClipBulkLoader(cache_bytes, fetch_workers),
            preprocessor,
            geometry,
        )

    def fetch(self, planned: Stage1PlannedBatch) -> FetchedStage1Batch:
        keys = tuple(dict.fromkeys(plan.key for plan in planned.windows))
        requests = tuple(self._request(key) for key in keys)
        clips = self.clips.load(requests)
        if len(clips) != len(keys):
            raise ValueError("Stage 1 bulk loader returned the wrong clip count")
        sources = tuple(
            self._prepare(key, clip) for key, clip in zip(keys, clips, strict=True)
        )
        return FetchedStage1Batch(planned.windows, sources)

    def collate(self, fetched: FetchedStage1Batch) -> Stage1Batch:
        sources = {source.item.key: source for source in fetched.sources}
        encoder_mels = []
        encoder_masks = []
        target_mels = []
        waveforms = []
        for plan in fetched.plans:
            source = sources[plan.key]
            target_start = plan.latent_start * self.geometry.posterior_rate
            target_end = target_start + self.geometry.target_mel_frames
            target = source.mel[:, target_start:target_end]
            sample_start = target_start * self.geometry.hop_length
            sample_end = sample_start + self.geometry.target_samples
            waveform = source.waveform[:, sample_start:sample_end]
            if target.shape[-1] != self.geometry.target_mel_frames:
                raise ValueError(f"Stage 1 target mel is incomplete: {plan.key}")
            if waveform.shape[-1] != self.geometry.target_samples:
                raise ValueError(f"Stage 1 target waveform is incomplete: {plan.key}")
            target_mels.append(target)
            waveforms.append(waveform)
            encoder, mask = self._encoder_window(source.mel, target_start)
            encoder_mels.append(encoder)
            encoder_masks.append(mask)
        batch_size = len(fetched.plans)
        return Stage1Batch(
            encoder_mel=torch.stack(encoder_mels),
            encoder_mask=torch.stack(encoder_masks),
            target_mel=torch.stack(target_mels),
            frame_mask=torch.ones(
                batch_size,
                1,
                self.geometry.target_mel_frames,
                dtype=torch.bool,
            ),
            waveform=torch.stack(waveforms),
            sample_keys=tuple(plan.key for plan in fetched.plans),
            window_indices=tuple(plan.window_index for plan in fetched.plans),
        )

    def close(self) -> None:
        self.clips.close()

    def _request(self, key: SegmentKey) -> SegmentReadRequest:
        item = self.index.records[key]
        return SegmentReadRequest(key.audio_file_id, item.start, item.end)

    def _prepare(self, key: SegmentKey, clip: WavClip) -> FetchedStage1Source:
        item = self.index.records[key]
        if clip.sample_rate != item.sample_rate:
            raise ValueError(f"Stage 1 source sample rate changed: {key}")
        processed = self.preprocessor.decode(clip, key)
        expected = self.geometry.mel_frames(item)
        if processed.mel.shape[-1] != expected:
            raise ValueError(
                f"Stage 1 planned {expected} mel frames but decoded "
                f"{processed.mel.shape[-1]} for {key}"
            )
        return FetchedStage1Source(item, processed.waveform, processed.mel)

    def _encoder_window(
        self,
        mel: torch.Tensor,
        target_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frames = self.geometry.encoder_mel_frames
        output = torch.zeros(mel.shape[0], frames, dtype=mel.dtype)
        mask = torch.zeros(1, frames, dtype=torch.bool)
        source_start = target_start - self.geometry.context_mel_frames
        source_end = source_start + frames
        copied_start = max(0, source_start)
        copied_end = min(mel.shape[-1], source_end)
        output_start = copied_start - source_start
        output_end = output_start + copied_end - copied_start
        output[:, output_start:output_end] = mel[:, copied_start:copied_end]
        mask[:, output_start:output_end] = True
        return output, mask
