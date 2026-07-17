from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VocoderProfile(str, Enum):
    NATIVE_300 = "native_300"
    PAPER_256 = "paper_256"


@dataclass(frozen=True)
class MelGeometry:
    n_fft: int
    win_length: int
    hop_length: int
    n_mels: int
    fmin: float
    fmax: float


@dataclass(frozen=True)
class SignalGeometry:
    sample_rate: int
    segment_samples: int
    synthesis_hop: int
    conditioning: MelGeometry
    reconstruction: tuple[MelGeometry, ...]
    target_steps: int | None


NATIVE_300 = SignalGeometry(
    sample_rate=24_000,
    segment_samples=9_600,
    synthesis_hop=300,
    conditioning=MelGeometry(2048, 1200, 300, 80, 0.0, 8_000.0),
    reconstruction=(
        MelGeometry(1024, 600, 120, 80, 0.0, 8_000.0),
        MelGeometry(2048, 1200, 240, 80, 0.0, 8_000.0),
        MelGeometry(512, 240, 50, 80, 0.0, 8_000.0),
    ),
    target_steps=None,
)

PAPER_256 = SignalGeometry(
    sample_rate=22_050,
    segment_samples=8_192,
    synthesis_hop=256,
    conditioning=MelGeometry(1024, 1024, 256, 80, 80.0, 7_600.0),
    reconstruction=(MelGeometry(1024, 1024, 256, 80, 0.0, 11_025.0),),
    target_steps=2_500_000,
)

PROFILE_GEOMETRY = {
    VocoderProfile.NATIVE_300: NATIVE_300,
    VocoderProfile.PAPER_256: PAPER_256,
}


def profile_geometry(profile: VocoderProfile) -> SignalGeometry:
    """Resolve the immutable signal contract selected by the CLI profile."""
    return PROFILE_GEOMETRY[profile]
