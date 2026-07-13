from dataclasses import dataclass, field, replace
from random import SystemRandom
from threading import Lock
from typing import Literal

from pydantic import Field

from runflow.core.node import Node
from runflow.core.ports import PortMode
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AudioPort
from runner.nodes.models import Audio, stable_id


class RandomAudioSubsetSettings(StrictSettings):
    selection: Literal["all", "random"] = "all"
    count: int = Field(default=100, ge=1)


@dataclass
class Reservoir:
    expected: int
    seen: int = 0
    items: list[Audio] = field(default_factory=list)


class RandomAudioSubsetNode(Node):
    NODE_TYPE = "RandomAudioSubset"
    DESCRIPTION = "Pass all incoming audio references through, or select a uniformly random fixed-size subset without loading audio bytes. Random mode uses reservoir sampling and emits the chosen references after the complete source batch arrives."
    CATEGORY = "Audio"
    SETTINGS = RandomAudioSubsetSettings
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"audio": AudioPort(mode=PortMode.STREAM)}
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    def __init__(self, node_id: str | None = None, **params):
        super().__init__(node_id=node_id, **params)
        self._random = SystemRandom()
        self._lock = Lock()
        self._reservoirs: dict[str, Reservoir] = {}

    async def execute(self, batch, context):
        assert len(batch) == 1, f"{self.id} requires disabled batching"
        context.check_cancel()
        audio = batch[0]["audio"]
        assert isinstance(audio, Audio), f"unsupported audio input: {type(audio).__name__}"
        if self.settings.selection == "all":
            return [
                {
                    "audio": replace(
                        audio,
                        metadata={
                            **audio.metadata,
                            "sample_selection": "all",
                            "sample_requested_count": None,
                        },
                    )
                }
            ]

        batch_id, expected = _source_batch(audio)
        with self._lock:
            reservoir = self._reservoirs.setdefault(batch_id, Reservoir(expected=expected))
            assert reservoir.expected == expected, f"source batch count changed for {batch_id}"
            reservoir.seen += 1
            if len(reservoir.items) < self.settings.count:
                reservoir.items.append(audio)
            else:
                slot = self._random.randrange(reservoir.seen)
                if slot < self.settings.count:
                    reservoir.items[slot] = audio
            if reservoir.seen < reservoir.expected:
                return []
            selected = list(reservoir.items)
            del self._reservoirs[batch_id]
        subset_id = stable_id("audio_subset", self.id, batch_id, *(item.audio_file_id for item in selected))
        count = len(selected)
        return [
            {
                "audio": [
                    replace(
                        item,
                        metadata={
                            **item.metadata,
                            "source_batch_id": subset_id,
                            "source_batch_count": count,
                            "sample_selection": "random",
                            "sample_requested_count": self.settings.count,
                        },
                    )
                    for item in selected
                ]
            }
        ]


def _source_batch(audio: Audio) -> tuple[str, int]:
    assert "source_batch_id" in audio.metadata, "RandomAudioSubset requires source_batch_id metadata"
    assert "source_batch_count" in audio.metadata, "RandomAudioSubset requires source_batch_count metadata"
    return str(audio.metadata["source_batch_id"]), int(audio.metadata["source_batch_count"])
