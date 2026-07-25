from dataclasses import replace

from .pipeline import DataPipelineState
from .records import BeetleBatch
from .sampling import PlannerState, PoolState
from .validation_records import ValidationRecording


class RepeatedBatchPipeline:
    def __init__(
        self,
        batch: BeetleBatch,
        data_fingerprint: str,
        world_size: int,
        initial_state: DataPipelineState | None,
    ) -> None:
        self.batch = batch
        self.data_fingerprint = data_fingerprint
        self.world_size = world_size
        self.batch_index = 0
        self.in_flight = False
        if initial_state is not None:
            self.batch_index = initial_state.planner.batch_index

    def next_batch(self) -> BeetleBatch:
        if self.in_flight:
            raise RuntimeError("mark the repeated batch consumed before requesting it again")
        self.in_flight = True
        return self.batch

    def mark_consumed(self) -> None:
        if not self.in_flight:
            raise RuntimeError("no repeated batch is awaiting consumption")
        self.batch_index += 1
        self.in_flight = False

    def state_dict(self) -> DataPipelineState:
        key = self.batch.sample_keys[0]
        index = self.batch_index
        planner = PlannerState(
            PoolState(index, (key,), 0),
            index,
            index,
            (),
            index,
        )
        return DataPipelineState(self.data_fingerprint, planner, self.world_size)

    def close(self) -> None:
        self.in_flight = False


def repeat_validation_embedding_groups(
    recording: ValidationRecording,
) -> ValidationRecording:
    batch = recording.batch
    repeated = replace(
        batch,
        style_views=batch.style_views.repeat(2, 1, 1, 1),
        voice_views=batch.voice_views.repeat(2, 1, 1, 1),
        style_view_lengths=batch.style_view_lengths.repeat(2, 1),
        voice_view_lengths=batch.voice_view_lengths.repeat(2, 1),
        style_distances=batch.style_distances.repeat(2, 1),
        style_group_ids=(
            f"{batch.style_group_ids[0]}:0",
            f"{batch.style_group_ids[0]}:1",
        ),
        voice_group_ids=(
            f"{batch.voice_group_ids[0]}:0",
            f"{batch.voice_group_ids[0]}:1",
        ),
    )
    return replace(recording, batch=repeated)
