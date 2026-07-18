from ..config.training import BeetleConfig
from .audio import AudioPreprocessor
from .collate import BatchCollator, Tokenizer
from .index import DatabaseSegmentIndex
from .prefetch import (
    BoundedBatchPrefetcher,
    DataPipelineState,
    PlannedBatchLoader,
    PrefetchCallbacks,
)
from .records import BeetleBatch, PlannedBatch
from .sampling import ContinuousBatchPlanner
from .sampling import DistributedShard
from .source import DatabaseBatchSource, FetchedBatch


class DatabaseBatchLoader:
    def __init__(self, source: DatabaseBatchSource, collator: BatchCollator) -> None:
        self.source = source
        self.collator = collator

    def fetch(self, planned: PlannedBatch) -> FetchedBatch:
        return self.source.fetch(planned)

    def collate(self, fetched: FetchedBatch) -> BeetleBatch:
        return self.collator.collate(fetched)

    def close(self) -> None:
        self.source.close()


def build_data_pipeline(
    config: BeetleConfig,
    stage: int,
    callbacks: PrefetchCallbacks,
    index: DatabaseSegmentIndex,
    phoneme_tokenizer: Tokenizer,
    text_tokenizer: Tokenizer,
    initial_state: DataPipelineState,
    shard: DistributedShard,
) -> BoundedBatchPrefetcher:
    stage_config = {1: config.stage1, 2: config.stage2, 3: config.stage3}[stage]
    planner = ContinuousBatchPlanner(
        index=index,
        stage=stage,
        batch_size=stage_config.batch_size,
        sentence_probability=config.data.sentence_probability,
        seed=config.runtime.seed,
        grouping=config.data.grouping,
        shard=shard,
    )
    source = DatabaseBatchSource.from_database(
        index,
        config.data.prefetch.audio_cache_bytes,
        config.data.prefetch.audio_fetch_workers,
    )
    audio = config.audio
    preprocessor = AudioPreprocessor(
        audio.sample_rate,
        audio.n_fft,
        audio.win_length,
        audio.hop_length,
        audio.mel_channels,
        audio.f_min,
        audio.f_max,
    )
    loader: PlannedBatchLoader = DatabaseBatchLoader(
        source,
        BatchCollator(
            preprocessor,
            phoneme_tokenizer,
            text_tokenizer,
            config.data.augmentation,
            config.architecture.language.values,
            stage,
            config.runtime.compile_frame_count if stage == 1 else None,
        ),
    )
    return BoundedBatchPrefetcher(
        planner=planner,
        loader=loader,
        callbacks=callbacks,
        maximum_batches=config.data.prefetch.planned_batches,
        maximum_decoded_bytes=config.data.prefetch.decoded_bytes,
        sample_rate=audio.sample_rate,
        initial_state=initial_state,
    )
