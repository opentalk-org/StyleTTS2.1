from ..config.training import BeetleConfig
from .audio import AudioPreprocessor
from .collate import BatchCollator, Tokenizer
from .index import DatabaseSegmentIndex
from .loader import DatabaseBatchLoader
from .prefetch import (
    BoundedBatchPrefetcher,
    DataPipelineState,
    PrefetchCallbacks,
)
from .sampling import ContinuousBatchPlanner
from .sampling import DistributedShard


def build_data_pipeline(
    config: BeetleConfig,
    callbacks: PrefetchCallbacks,
    index: DatabaseSegmentIndex,
    phoneme_tokenizer: Tokenizer,
    text_tokenizer: Tokenizer,
    initial_state: DataPipelineState,
    shard: DistributedShard,
) -> BoundedBatchPrefetcher:
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
    planner = ContinuousBatchPlanner(
        index=index,
        batch_size=config.training.batch_size,
        seed=config.runtime.seed,
        maximum_seconds=config.data.maximum_seconds,
        grouping=config.data.grouping,
        shard=shard,
    )
    loader = DatabaseBatchLoader.from_database(
        index,
        BatchCollator(
            preprocessor,
            phoneme_tokenizer,
            text_tokenizer,
            config.data.augmentation,
            config.architecture.language.values,
            config.adversarial.segment_samples // config.audio.hop_length,
        ),
        config.data.prefetch.audio_cache_bytes,
        config.data.prefetch.audio_fetch_workers,
    )
    return BoundedBatchPrefetcher(
        planner=planner,
        loader=loader,
        callbacks=callbacks,
        window_size=config.data.prefetch.window_size,
        maximum_decoded_bytes=config.data.prefetch.decoded_bytes,
        sample_rate=audio.sample_rate,
        initial_state=initial_state,
    )
