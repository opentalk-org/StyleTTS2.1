from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AlignedSegments:
    frame_starts: Tensor
    frame_count: int
    frame_alignment: int
    sample_hop: int

    @classmethod
    def random(
        cls,
        frame_mask: Tensor,
        frame_count: int,
        frame_alignment: int,
        sample_hop: int,
        generator: torch.Generator,
    ) -> "AlignedSegments":
        lengths = frame_mask[:, 0].sum(dim=1)
        available = lengths - frame_count
        torch._assert_async(
            torch.all(available >= 0),
            "utterance is shorter than adversarial segment",
        )
        aligned_choices = torch.div(
            available,
            frame_alignment,
            rounding_mode="floor",
        ) + 1
        samples = torch.rand(
            lengths.shape,
            device=frame_mask.device,
            generator=generator,
        )
        starts = torch.floor(samples * aligned_choices).to(dtype=torch.long)
        return cls(starts * frame_alignment, frame_count, frame_alignment, sample_hop)

    @classmethod
    def reference_chunks(
        cls,
        frame_mask: Tensor,
        frame_count: int,
        sample_hop: int,
        generator: torch.Generator,
    ) -> "AlignedSegments":
        lengths = frame_mask[:, 0].sum(dim=1)
        torch._assert_async(
            torch.all(lengths >= frame_count),
            "utterance is shorter than adversarial segment",
        )
        complete = torch.div(lengths, frame_count, rounding_mode="floor")
        remainder = lengths.remainder(frame_count)
        choices = complete + (remainder > 0)
        selected = torch.floor(
            torch.rand(lengths.shape, device=lengths.device, generator=generator)
            * choices
        ).to(dtype=torch.long)
        starts = selected * frame_count
        tail = (remainder > 0) & (selected == complete)
        starts = torch.where(tail, lengths - frame_count, starts)
        return cls(starts, frame_count, frame_count, sample_hop)

    def frames(self, values: Tensor) -> Tensor:
        return self._take(values, self.frame_starts, self.frame_count)

    def latents(self, values: Tensor) -> Tensor:
        starts = torch.div(
            self.frame_starts,
            self.frame_alignment,
            rounding_mode="floor",
        )
        count = self.frame_count // self.frame_alignment
        return self._take(values, starts, count)

    def samples(self, values: Tensor) -> Tensor:
        starts = self.frame_starts * self.sample_hop
        count = self.frame_count * self.sample_hop
        return self._take(values, starts, count)

    def context_frames(self, values: Tensor, context_frame_count: int) -> Tensor:
        context_side = context_frame_count // 2
        starts = self.frame_starts - context_side
        count = self.frame_count + context_frame_count
        positions = starts.unsqueeze(1) + torch.arange(count, device=values.device)
        valid = (positions >= 0) & (positions < values.shape[-1])
        positions = positions.clamp(0, values.shape[-1] - 1)
        for _ in range(values.ndim - 2):
            positions = positions.unsqueeze(1)
            valid = valid.unsqueeze(1)
        indices = positions.expand(*values.shape[:-1], count)
        selected = torch.gather(values, -1, indices)
        return torch.where(valid.expand_as(selected), selected, torch.zeros_like(selected))

    def _take(self, values: Tensor, starts: Tensor, count: int) -> Tensor:
        positions = starts.unsqueeze(1) + torch.arange(count, device=values.device)
        for _ in range(values.ndim - 2):
            positions = positions.unsqueeze(1)
        indices = positions.expand(*values.shape[:-1], count)
        return torch.gather(values, -1, indices)
